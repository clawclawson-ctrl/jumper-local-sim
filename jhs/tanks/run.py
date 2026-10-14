"""1v1 Crab Tanks in the local sim. SIM-ONLY VISION CONCEPT.

    python jhs/tanks/run.py --app jumper_tanks_vision.app --scene RUN/scene.map --backend native --device cpu --headless --steps 999999999
Env: TK_SETUP (json {A, B, props}), VS_SEED, VS_TMAX, VS_OUT, VS_SAVEVIS, TK_LIVE=1, TK_HITS (hits to win, default 3),
     TK_BOUNCE (1 = shots bounce off walls), TK_PUSHABLE, JUMPER_REPO.
Each crab = the official walking app driven by its own tanks Brain through a virtual gamepad, from ITS OWN camera +
dToF + pose. The turret and projectiles are a sim add-on (drawn into every camera, so the crabs SEE the shots). The
REFEREE (projectile flight, hits, stun, score, walk-back resets) is the only code that uses sim truth."""
from __future__ import annotations
import importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from jhs.soccer import world as WD, vision  # noqa: E402
from jhs.tanks import field as F  # noqa: E402
from jhs.tanks import arena as A, brain as TB, fx as FXD  # noqa: E402

REPO = Path(os.environ.get("JUMPER_REPO", str(HERE.parent.parent / "toolkit")))
SEED = int(os.environ.get("VS_SEED", "1")); TMAX = float(os.environ.get("VS_TMAX", "300"))
OUTD = Path(os.environ.get("VS_OUT", "tanks_run")); OUTD.mkdir(parents=True, exist_ok=True)
SAVEVIS = os.environ.get("VS_SAVEVIS", "1") == "1"
SETUP = json.load(open(os.environ["TK_SETUP"])) if os.environ.get("TK_SETUP") else {}
START = {k: [float(v) for v in (SETUP.get(k) or A.START[k])] for k in ("A", "B")}
# randomized per-crab openings (same menu/odds for both). Seeded -> reproducible; no seed -> differs every match.
OPEN_CFG = dict(SETUP.get("openings") or {}); OPEN_CFG.setdefault("mode", "random")
OPEN_SEED = int(OPEN_CFG["seed"]) if OPEN_CFG.get("seed") not in (None, "") else int.from_bytes(os.urandom(4), "little")
ORNG = np.random.default_rng(OPEN_SEED)
ROUND = TB.roll_round(ORNG, OPEN_CFG); OPEN_LOG = []
for _k in ("A", "B"):
    if [round(v, 3) for v in START[_k]] == [round(float(v), 3) for v in A.START[_k]]: START[_k] = list(TB.jittered_start(_k, ROUND[_k]["jit"]))
VW, VH, VIS_DT = 640, 480, 0.1
RETURN_TIMEOUT, RET_POS, RET_YAW = 25.0, 0.15, math.radians(15)
HITS_TO_WIN = max(1, min(10, int(os.environ.get("TK_HITS", "3"))))
BOUNCE = os.environ.get("TK_BOUNCE", "1") == "1"

sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("play", REPO / "scripts" / "play.py")
play = importlib.util.module_from_spec(_spec); sys.modules["play"] = play; _spec.loader.exec_module(play)
import controller  # noqa: E402
import tasks  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("VS_THREADS", "2"))); torch.manual_seed(SEED)
WD.install(tasks, START["A"], START["B"], SEED)
ap = WD.share_controller()


class VirtualPad:
    connected = True; path = "virtual"
    def __init__(self, name): self.name = name; self.s = controller.GamepadState()
    def state(self): return self.s
    def close(self): pass
    def set(self, c):
        btn = set(); hx = hy = 0
        for part in (c.get("press", "") or "").split("+"):
            if part == "dpad_up": hy = -1
            elif part == "dpad_down": hy = 1
            elif part: btn.add(part)
        self.s = controller.GamepadState(lx=float(c.get("lx", 0)), ly=float(c.get("ly", 0)), rx=float(c.get("rx", 0)), ry=0.0, lt=0.0, rt=0.0,
                                         hat_x=hx, hat_y=hy, buttons=frozenset(btn))


class Celebration:
    """the shooter celebrates with the official app's own moves: a hit = crab dance ~3 s; the winning hit = bow + dance."""
    def __init__(self, t, mode_of, win):
        self.mode_of = mode_of; self.t0 = t; self.done = False
        steps = [("wait", 0.3)]
        if win: steps += [("press", "dpad_down", 0.2), ("until", "gesture_bow", 3.0), ("until", "locomotion", 12.0), ("wait", 0.4)]
        steps += [("press", "menu+dpad_up", 0.2), ("until", "dance_crab", 3.0), ("wait", 6.0 if win else 3.0),
                  ("press", "menu", 0.15), ("until", "locomotion", 6.0), ("wait", 0.5)]
        self.steps = steps; self.i = 0; self.ts = t; self.label = "celebrating: " + ("bow + crab dance" if win else "crab dance")

    def cmd(self, t):
        while self.i < len(self.steps):
            st = self.steps[self.i]
            if st[0] == "wait" and t - self.ts < st[1]: return {}
            if st[0] == "press" and t - self.ts < st[2]: return {"press": st[1]}
            if st[0] == "until" and self.mode_of() != st[1] and t - self.ts < st[2]: return {}
            self.i += 1; self.ts = t
        self.done = True; return {}


PADS = {"A": VirtualPad("virtual pad RED (tanks brain, sim-only)"), "B": VirtualPad("virtual pad BLUE (tanks brain, sim-only)")}
controller.open = lambda *a, **k: PADS["A"]


def yaw_of(q): return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
    global ROUND
    import mujoco, imageio
    from scipy import ndimage
    un = env.unwrapped; sim = un.sim
    robs = {"A": un.scene["robot"], "B": un.scene["robot2"]}
    tofs = {"A": un.scene["tof"], "B": un.scene["tof2"]}
    rb = robs["B"]; rs = rb.data.default_root_state.clone(); rs[:, :3] += un.scene.env_origins
    rb.write_root_state_to_sim(rs); rb.write_joint_state_to_sim(rb.data.default_joint_pos.clone(), torch.zeros_like(rb.data.default_joint_vel))
    pB = ap.AppPlayer(ap.open_app(Path(sys.argv[sys.argv.index("--app") + 1])), WD.EnvView(env, rb), pad=PADS["B"])
    pA = policy
    mgr = un.action_manager; names_ = list(mgr.active_terms)
    def both(): pA._apply(); pB._apply()
    mgr.get_term(names_[0]).apply_actions = both
    model, live = sim.env_mjdata(0)
    nsk = WD.skin(model)
    rmodel = getattr(sim, "render_model", None) or model
    if rmodel is not model: WD.skin(rmodel)
    model.vis.global_.offwidth = max(VW, model.vis.global_.offwidth); model.vis.global_.offheight = max(VH, model.vis.global_.offheight)
    rend = mujoco.Renderer(model, VH, VW)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
    def body(suffix):
        m = [i for i, n in enumerate(names) if n.endswith(suffix)]; return m[0] if m else None
    def _root(b):
        while model.body_parentid[b] != 0: b = model.body_parentid[b]
        return b
    groot = [names[_root(model.geom_bodyid[g])] for g in range(model.ngeom)]
    PFX = {"A": "robot/", "B": "robot2/"}
    own_geom = {k: np.array([n.startswith(p) for n in groot] + [False]) for k, p in PFX.items()}
    cams = {k: (model.camera(PFX[k] + "onboard").id, model.camera(PFX[k] + "tof").id) for k in PFX}
    cid, tid = cams["A"]
    cam = vision.Camera(VW, VH, float(model.cam_fovy[cid]))
    tw, th = (int(v) for v in model.cam_resolution[tid]); tcam = vision.Camera(tw, th, float(model.cam_fovy[tid]))
    hfov = 2 * math.atan(VW / 2 / cam.f); TB.FOV = hfov - math.radians(8)       # turret stays inside the camera's view
    base = {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PFX[k] + "base_link") for k in PFX}
    ball_b = body("prop:football"); bq = model.jnt_qposadr[model.body_jntadr[ball_b]]; bd = model.jnt_dofadr[model.body_jntadr[ball_b]]
    pb = {k: body(v[0]) for k, v in F.PROPS.items() if body(v[0]) is not None}
    DUE = {"due": True}
    for s in tofs.values():
        _k = s.raycast_kernel
        def _when(*a, _k=_k, _s=s, **kw):
            if DUE["due"]: _s._force[:] = True; return _k(*a, **kw)
        s.raycast_kernel = _when

    def park_ball():      # the field's football is not used: it waits outside the arena
        model, live = sim.env_mjdata(0)
        live.qpos[bq:bq + 3] = [A.BALL_PARK[0], A.BALL_PARK[1], F.BALL_R + 0.003]; live.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]; live.qvel[bd:bd + 6] = 0
        mujoco.mj_forward(model, live)
        if hasattr(sim, "_gather"): sim._gather()

    def place_robot(k, x, y, deg):
        model, live = sim.env_mjdata(0); jn = model.body_jntadr[base[k]]
        qa, da = model.jnt_qposadr[jn], model.jnt_dofadr[jn]; h = math.radians(deg) / 2
        live.qpos[qa:qa + 2] = [x, y]; live.qpos[qa + 2] = max(float(live.qpos[qa + 2]), 0.11)
        live.qpos[qa + 3:qa + 7] = [math.cos(h), 0.0, 0.0, math.sin(h)]; live.qvel[da:da + 6] = 0
        mujoco.mj_forward(model, live)
        if hasattr(sim, "_gather"): sim._gather()

    def cur_props():
        model, live = sim.env_mjdata(0); out = {}
        for k, b in pb.items():
            x, y = float(live.xpos[b][0]), float(live.xpos[b][1])
            if abs(x) < F.FX + 0.2 and abs(y) < F.FY + 0.2: out[k] = (x, y, math.degrees(yaw_of(live.xquat[b])))
        return out

    P0 = {k: tuple(float(c) for c in (list(v) + [0.0])[:3]) for k, v in (SETUP.get("props") or {}).items() if k in F.PROPS}
    for k in ("A", "B"):
        fx_, fy_ = F.free_spot(START[k][0], START[k][1], P0, 0.30)
        if (fx_, fy_) != (START[k][0], START[k][1]):
            print(f"[setup] {F.TEAM[k]} crab overlaps an obstacle -> starts at ({fx_}, {fy_})"); START[k] = [fx_, fy_, START[k][2]]
    _m, _l = sim.env_mjdata(0); ks = list(P0)
    for i, k in enumerate(ks):
        if any(F._box_dist((P0[k][0], P0[k][1]), k2, *P0[k2]) < max(F.PROPS[k][2:]) for k2 in ks[:i]):
            j = _m.body_jntadr[pb[k]]
            if j >= 0 and _m.jnt_type[j] == 0: _l.qpos[_m.jnt_qposadr[j] + 2] += 0.45; print(f"[setup] {k} overlaps another obstacle -> lifted")
    mujoco.mj_forward(_m, _l)
    park_ball()
    _pbset = set(pb.values()); _gn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    obst_geom = np.array([(_root(model.geom_bodyid[g]) in _pbset) or _gn[g].startswith(("wall_", "corner_", "goal")) for g in range(model.ngeom)])
    bumps = {"A": 0, "B": 0}; bump_prev = {"A": False, "B": False}
    brains = {k: TB.Brain(k, np.random.default_rng(SEED * 31 + i), pushable=os.environ.get("TK_PUSHABLE", "1") == "1", bounce=BOUNCE) for i, k in enumerate(("A", "B"))}
    for k in ("A", "B"): brains[k].set_opening(ROUND[k], 0.0)
    OPEN_LOG.append(dict(round=1, **{F.TEAM[k]: ROUND[k]["opener"] for k in ("A", "B")}, start={F.TEAM[k]: START[k] for k in ("A", "B")}, contact=None))
    print("[openings] seed=%d mode=%s round 1: RED %s, BLUE %s" % (OPEN_SEED, OPEN_CFG["mode"], ROUND["A"]["opener"], ROUND["B"]["opener"]), flush=True)
    HOOK = None
    if os.environ.get("TK_LIVE", "0") == "1" or os.environ.get("JHS_LIVE_DIR"):
        from jhs.tanks.live import TanksHook
        HOOK = TanksHook(sim=sim, outd=OUTD, dt=un.step_dt)
    print(f"[setup] tanks seed={SEED} tmax={TMAX} A={START['A']} B={START['B']} props={SETUP.get('props', {})} bounce={BOUNCE} hits_to_win={HITS_TO_WIN} "
          f"skins={nsk} cam {VW}x{VH} hfov {math.degrees(hfov):.0f} deg tof {tw}x{th}", flush=True)
    dt = float(un.step_dt); vis_every = max(1, round(VIS_DT / dt)); q_every = max(1, round(0.04 / dt))
    obs = env.get_observations(); step = 0
    score = {"A": 0, "B": 0}; hits = []; events = []; falls = {"A": 0, "B": 0}; fallen = {"A": False, "B": False}
    cele = None; final = None; winner = None; end_at = None
    modes = {"A": lambda: pA.fsm.mode, "B": lambda: pB.fsm.mode}
    settle_until = 1.0; paused_s = 0.0; prev_t = 0.0; returns = []
    pending = None; ref_line = "fight!"
    shots = []; turret = {"A": math.radians(START["A"][2]), "B": math.radians(START["B"][2])}
    stun_until = {"A": 0.0, "B": 0.0}; fired = {"A": 0, "B": 0}; shot_end = {"hit": 0, "cover": 0, "wall": 0, "expired": 0}
    dodges_ok = {"A": 0, "B": 0}; hit_seen = {"A": 0, "B": 0}; survived_seen = {"A": 0, "B": 0}
    qbuf, qt, vis_meta, tofA, tofB, fxbuf = [], [], [], [], [], []
    truth = open(OUTD / "truth.jsonl", "w"); t_wall = time.time(); stopped = False; t = 0.0; mt = 0.0
    cmds = {"A": {}, "B": {}}; poses = {"A": tuple(START["A"][:2]) + (math.radians(START["A"][2]),), "B": tuple(START["B"][:2]) + (math.radians(START["B"][2]),)}
    rot_only = {"A": 0.0, "B": 0.0}; max_rot = {"A": 0.0, "B": 0.0}
    def ev(kind, **kw):
        e = dict(t=round(t, 2), event=kind, **kw); events.append(e); print(f"[ref] {e}", flush=True)
    def crab_xyz():
        _m, lv = sim.env_mjdata(0); return {k: tuple(float(v) for v in lv.xpos[base[k]][:3]) for k in ("A", "B")}
    def fx_draw(skip=None):
        cz = crab_xyz()
        return lambda mj, scn: FXD.draw(mj, scn, cz, turret, shots, skip=skip)
    if HOOK is not None: HOOK.draw_cb = lambda mj, scn: FXD.draw(mj, scn, crab_xyz(), turret, shots)
    try:
        while True:
            t = step * dt
            if end_at is not None and t >= end_at: break
            if pending is not None and pending[0] in ("celebrate", "return"): paused_s += t - prev_t
            prev_t = t; mt = t - paused_s
            if mt >= TMAX and final is None:
                winner = "A" if score["A"] > score["B"] else "B" if score["B"] > score["A"] else None
                final = (f"{F.TEAM[winner]} WINS {score['A']}-{score['B']}" if winner else f"DRAW {score['A']}-{score['B']}") + " (time)"
                ref_line = f"time: {final}"; ev("full_time", final=final); end_at = t + 3.0; pending = ("over", 1e9)
            if step == 2:
                for k in ("A", "B"): place_robot(k, *START[k])
                park_ball()
            model, live = sim.env_mjdata(0)
            if step % vis_every == 0 and t > 0.3:
                metas = {}
                for k in ("A", "B"):
                    c_id, t_id = cams[k]
                    rend.update_scene(live, camera=c_id); FXD.draw(mujoco, rend.scene, crab_xyz(), turret, shots, skip=k); img = rend.render().copy()
                    rend.enable_segmentation_rendering(); rend.update_scene(live, camera=c_id); seg = rend.render()[..., 0].copy(); rend.disable_segmentation_rendering()
                    selfm = ndimage.binary_dilation(own_geom[k][np.clip(seg, 0, own_geom[k].size - 1)] & (seg >= 0), iterations=3)
                    rngs = tofs[k].data.range[0].detach().cpu().numpy().reshape(th, tw)
                    cpos = live.cam_xpos[c_id].copy(); cR = live.cam_xmat[c_id].reshape(3, 3).copy()
                    tpos = live.cam_xpos[t_id].copy(); tR = live.cam_xmat[t_id].reshape(3, 3).copy()
                    p = TB.perceive(k, img, selfm, cpos, cR, cam, rngs, tpos, tR, tcam)
                    q = live.xquat[base[k]]; pose = (float(live.xpos[base[k]][0]), float(live.xpos[base[k]][1]), yaw_of(q))
                    br = brains[k]
                    if p["opp"] and OPEN_LOG[-1]["contact"] is None and pending is None:
                        OPEN_LOG[-1]["contact"] = dict(t=round(t, 1), by=F.TEAM[k], at=[round(pose[0], 2), round(pose[1], 2)])
                    br.turret = turret[k]; br.own_in_flight = any(s["owner"] == k for s in shots)   # proprioception: turret angle, "reloaded" lamp
                    br.stun_until = stun_until[k]
                    br.observe(t, p, pose); poses[k] = pose
                    for s_ in shots:          # stat only (SIM TRUTH): did this crab's own vision pick up the other crab's shot?
                        if s_["owner"] != k and any(math.hypot(d_["xy"][0] - s_["x"], d_["xy"][1] - s_["y"]) < 0.3 for d_ in p["shots"]): s_["seen"] = True
                    fire = None
                    if pending is not None and pending[0] == "return": cmds[k] = br.return_cmd(pose)
                    elif pending is None:
                        try: cmds[k], fire = br.decide(pose)
                        except Exception as e:
                            cmds[k] = {}; ev("brain_error", team=F.TEAM[k], error=repr(e)[:200])
                    elif pending[0] == "celebrate" and k == pending[2]: cmds[k] = cele.cmd(t)
                    else: cmds[k] = {}
                    if t < stun_until[k]: cmds[k] = {}; fire = None
                    PADS[k].set(cmds[k])
                    # turret (sim add-on): the brain's aim, clamped to the camera view around the body heading
                    e = TB.wrap(br.turret - pose[2]); e = float(np.clip(e, -TB.FOV / 2, TB.FOV / 2)); turret[k] = TB.wrap(pose[2] + e)
                    if fire is not None and pending is None and not any(s["owner"] == k for s in shots):
                        a = turret[k]
                        shots.append(dict(owner=k, x=pose[0] + A.MUZZLE * math.cos(a), y=pose[1] + A.MUZZLE * math.sin(a),
                                          vx=A.SHOT_SPEED * math.cos(a), vy=A.SHOT_SPEED * math.sin(a), age=0.0, bounces=0, t0=round(t, 2)))
                        fired[k] += 1; ev("fire", team=F.TEAM[k], yaw=round(math.degrees(a), 1), lead_s=None if fire.get("lead") is None else round(fire["lead"], 2))
                    # spinning watchdog stat (log only): continuous turn-only time
                    c = cmds[k]
                    if abs(c.get("rx", 0)) > 0.1 and abs(c.get("lx", 0)) + abs(c.get("ly", 0)) < 0.05: rot_only[k] += VIS_DT
                    else: rot_only[k] = 0.0
                    max_rot[k] = max(max_rot[k], rot_only[k])
                    _cel = pending is not None and pending[0] == "celebrate" and k == pending[2]
                    st_ = br.state if pending is None or pending[0] == "return" else ("celebrate" if _cel else "paused" if pending[0] == "celebrate" else "referee")
                    if t < stun_until[k]: st_ = "stunned"
                    metas[k] = dict(state=st_, sub=br.sub if pending is None or pending[0] == "return" else (cele.label if _cel else "standing still" if pending[0] == "celebrate" else "over"),
                                    pose=[round(v, 3) for v in pose], cmd={a_: (round(float(b_), 2) if a_ != "press" else b_) for a_, b_ in cmds[k].items()},
                                    turret=round(math.degrees(TB.wrap(turret[k] - pose[2])), 1), turret_rad=round(turret[k], 3),
                                    opp=[round(float(v), 3) for v in br.opp] if br.opp is not None and t - br.opp_t < 3.0 else None,
                                    opp_age=round(min(t - br.opp_t, 99), 1),
                                    dets=[dict(cls="opponent", bbox=d["bbox"], rng=round(d["rng"], 2)) for d in p["opp"]] +
                                         [dict(cls="shot", bbox=d["bbox"], rng=round(d["rng"], 2)) for d in p["shots"]],
                                    nshots_seen=len(br.shots), nobst=int(p["obst"].shape[0]), belief=br.bel.coarse(4).tolist(), opener=br.op_name, opener_on=br.opening is not None)
                    if SAVEVIS:
                        (OUTD / "vis").mkdir(exist_ok=True)
                        imageio.imwrite(OUTD / "vis" / f"{k}_{len(vis_meta):05d}.jpg", img[::2, ::2], quality=85)
                        (tofA if k == "A" else tofB).append(rngs.astype(np.float16))
                    if HOOK is not None: HOOK.vision(k, img, rngs)
                meta = dict(t=round(t, 2), clock=round(mt, 2), A=metas["A"], B=metas["B"], score=dict(score), ref=ref_line, final=final, hits_to_win=HITS_TO_WIN,
                            shots=[dict(x=round(s["x"], 3), y=round(s["y"], 3), vx=round(s["vx"], 2), vy=round(s["vy"], 2), owner=s["owner"]) for s in shots])
                if SAVEVIS: vis_meta.append(meta)
                if HOOK is not None: HOOK.meta(meta)
            DUE["due"] = (step + 1) % vis_every == 0
            with torch.inference_mode():
                pB(None)
                obs, _, _, _ = env.step(pA(obs))
            if HOOK is not None and HOOK.step(t) == "stop":
                stopped = True; print(f"[run] stopped from the local runner at t={t:.1f}s", flush=True); break
            # ---------------- referee (sim truth) ----------------
            model, live = sim.env_mjdata(0)
            if shots:
                cz = crab_xyz(); before = {id(s): s for s in shots}
                for kind, s, who in A.step_shots(shots, dt, cur_props() if step % 5 == 0 or True else {}, {k: cz[k][:2] for k in cz}, bounce=BOUNCE):
                    shot_end[kind] += 1
                    if kind == "hit" and pending is None:
                        sh = s["owner"]; score[sh] += 1; stun_until[who] = t + A.STUN_S
                        if s.get("seen"): hit_seen[who] += 1
                        hits.append(dict(t=round(t, 1), team=F.TEAM[sh], target=F.TEAM[who], flight_s=round(s["age"], 2), bounces=s["bounces"]))
                        ref_line = f"HIT {F.TEAM[sh]} on {F.TEAM[who]} at {t:.0f}s"; ev("hit", team=F.TEAM[sh], target=F.TEAM[who], score=dict(score), bounces=s["bounces"])
                        win = score[sh] >= HITS_TO_WIN
                        if win: winner = sh; final = f"{F.TEAM[sh]} WINS {score['A']}-{score['B']}"; ref_line = final; ev("match_won", final=final)
                        cele = Celebration(t + A.STUN_S * 0, modes[sh], win); pending = ("celebrate", None, sh); shots.clear(); break
                    elif kind != "hit":
                        tgt = "B" if s["owner"] == "A" else "A"
                        if s.get("closest", 9) < 0.45: dodges_ok[tgt] += 1    # a shot that came close and missed = dodged (or lucky)
                        if s.get("seen") and s.get("closest", 9) < 0.6: survived_seen[tgt] += 1
                        ev("shot_end", team=F.TEAM[s["owner"]], how=kind, at=[round(s["x"], 2), round(s["y"], 2)], closest=round(s.get("closest", 9), 2))
                cz = crab_xyz()
                for s in shots:      # closest approach to the target (stat)
                    tgt = "B" if s["owner"] == "A" else "A"
                    s["closest"] = min(s.get("closest", 9.0), math.hypot(cz[tgt][0] - s["x"], cz[tgt][1] - s["y"]))
            if step % 10 == 0:
                for k in ("A", "B"):
                    z = float(live.xpos[base[k]][2]); upz = float(live.xmat[base[k]].reshape(3, 3)[2, 2])
                    isf = (z < 0.04 or upz < 0.5) and t > settle_until
                    if isf and not fallen[k]: falls[k] += 1; ev("fall", team=F.TEAM[k], z=round(z, 3))
                    fallen[k] = isf
                _hit = {"A": False, "B": False}
                for ci in range(live.ncon):
                    g1, g2 = live.contact[ci].geom1, live.contact[ci].geom2
                    for k in ("A", "B"):
                        if (own_geom[k][g1] and obst_geom[g2]) or (own_geom[k][g2] and obst_geom[g1]): _hit[k] = True
                for k in ("A", "B"):
                    if _hit[k] and not bump_prev[k]: bumps[k] += 1
                    bump_prev[k] = _hit[k]
                if pending is not None and pending[0] == "celebrate":
                    if cele.done or t - cele.t0 > 30.0:
                        if winner is not None: end_at = t + 0.5; pending = ("over", 1e9)
                        else:
                            P = cur_props()
                            ROUND = TB.roll_round(ORNG, OPEN_CFG)       # a fresh roll (opener + jitter) every round
                            _js = {k: TB.jittered_start(k, ROUND[k]["jit"]) for k in ("A", "B")}
                            spots = {k: (*F.free_spot(_js[k][0], _js[k][1], P, 0.30), _js[k][2]) for k in ("A", "B")}
                            ref_line = "referee: both crabs walk back to their start spots"; ev("walk_back", spots={F.TEAM[k]: spots[k] for k in spots})
                            pending = ("return", t, spots)
                            for k in ("A", "B"): brains[k].begin_return(spots[k], t)
                elif pending is not None and pending[0] == "return":
                    spots = pending[2]; ok = {}
                    for k in ("A", "B"):
                        sx, sy, sh_ = spots[k]; x, y, yaw = poses[k]
                        ok[k] = math.hypot(x - sx, y - sy) < RET_POS and abs(TB.wrap(math.radians(sh_) - yaw)) < RET_YAW
                    late = [k for k in ok if not ok[k]] if t - pending[1] > RETURN_TIMEOUT else None
                    if all(ok.values()) or late:
                        for k in (late or []): place_robot(k, *spots[k])
                        returns.append(dict(t=round(t - pending[1], 1), timeout=[F.TEAM[k] for k in (late or [])]))
                        for k, b in brains.items(): b.reset(kickoff=True, t=t); b.set_opening(ROUND[k], t)
                        OPEN_LOG.append(dict(round=len(OPEN_LOG) + 1, **{F.TEAM[k]: ROUND[k]["opener"] for k in ("A", "B")}, start={F.TEAM[k]: list(spots[k]) for k in ("A", "B")}, contact=None))
                        print("[openings] round %d: RED %s, BLUE %s" % (len(OPEN_LOG), ROUND["A"]["opener"], ROUND["B"]["opener"]), flush=True)
                        ref_line = (f"referee reset (timeout): {', '.join(F.TEAM[k] for k in late)} placed" if late else f"both crabs walked back ({t - pending[1]:.1f} s)") + f"; fight! ({score['A']}-{score['B']})"
                        ev("restart", took=round(t - pending[1], 1), late=[F.TEAM[k] for k in (late or [])]); pending = None; settle_until = t + 1.0
                if step % 20 == 0:
                    truth.write(json.dumps({"t": round(t, 2), **{k: [round(float(v), 3) for v in live.xpos[base[k]][:3]] for k in ("A", "B")},
                                            "shots": [[round(s["x"], 2), round(s["y"], 2), s["owner"]] for s in shots]}) + "\n")
            if step % q_every == 0:
                qbuf.append(np.array(live.qpos, dtype=np.float64)); qt.append(t)
                fxbuf.append(dict(tur={k: round(turret[k], 3) for k in turret}, shots=[[round(s["x"], 3), round(s["y"], 3), s["owner"], round(s["vx"], 3), round(s["vy"], 3)] for s in shots]))
            step += 1
            if step % 2000 == 0:
                print(f"[run] t={t:.0f}s wall={time.time() - t_wall:.0f}s score {score['A']}-{score['B']} A={brains['A'].state} B={brains['B'].state} "
                      f"shots={fired} falls={falls}", flush=True)
    finally:
        truth.close()
        res = dict(kind="tanks", label="SIM-ONLY VISION CONCEPT", seed=SEED, arena=F.ARENA, openings=dict(mode=OPEN_CFG["mode"], seed=OPEN_SEED, rounds=OPEN_LOG), tmax=TMAX, setup=dict(arena=F.ARENA, A=START["A"], B=START["B"], props=SETUP.get("props", {})),
                   bounce=BOUNCE, hits_to_win=HITS_TO_WIN, t_end=round(step * dt, 2), stopped=stopped,
                   score={"RED": score["A"], "BLUE": score["B"]}, hits=hits, shots_fired={F.TEAM[k]: v for k, v in fired.items()},
                   shot_ends=shot_end, close_misses_survived={F.TEAM[k]: v for k, v in dodges_ok.items()},
                   seen_shots_survived={F.TEAM[k]: v for k, v in survived_seen.items()}, hit_by_seen_shot={F.TEAM[k]: v for k, v in hit_seen.items()},
                   leg_bumps={F.TEAM[k]: v for k, v in bumps.items()}, returns=returns, match_clock_s=round(mt, 1),
                   max_spin_only_s={F.TEAM[k]: round(v, 1) for k, v in max_rot.items()},
                   winner=F.TEAM.get(winner) if final else None, final=final, falls={"RED": falls["A"], "BLUE": falls["B"]},
                   brain_counts={F.TEAM[k]: dict(b.counts) for k, b in brains.items()}, wall_s=round(time.time() - t_wall, 1))
        json.dump(res, open(OUTD / "result.json", "w"), indent=1)
        with open(OUTD / "events.jsonl", "w") as f:
            for e in events: f.write(json.dumps(e) + "\n")
        if qbuf:
            np.savez_compressed(OUTD / "qpos.npz", qpos=np.stack(qbuf), t=np.array(qt), fps=25.0)
            json.dump(fxbuf, open(OUTD / "tanks_fx.json", "w"))
        if SAVEVIS and vis_meta:
            n = min(len(tofA), len(tofB), len(vis_meta))
            np.savez_compressed(OUTD / "vision_tof.npz", tofA=np.stack(tofA[:n]), tofB=np.stack(tofB[:n]))
            json.dump(vis_meta[:n], open(OUTD / "vision_meta.json", "w"))
        print("[result] " + json.dumps(res), flush=True)


play._loop = _loop
sys.argv[0] = str(REPO / "scripts" / "play.py")
if __name__ == "__main__":
    play.main()
