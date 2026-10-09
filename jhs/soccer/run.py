"""1v1 crab soccer match in the local sim. SIM-ONLY VISION CONCEPT.

    python jhs/soccer/run.py --app jumper_soccer_vision.app --scene RUN/scene.map --backend native --device cpu --headless --steps 999999999
Env: SC_SETUP (json {ball, A, B, props}), VS_SEED, VS_TMAX (match length, sim s), VS_OUT (run folder), VS_SAVEVIS,
     SC_LIVE=1 (live view / pacing / Stop via jhs.soccer.live), SC_GOALS (goals to win, default 3), JUMPER_REPO.
Each crab = the official walking app driven by its own Brain (jhs/soccer/brain.py) through a virtual gamepad, from
ITS OWN camera + dToF + pose. The REFEREE below is the only code that reads the ball/crab positions from sim truth:
goals, score, kickoff resets (teleports, labelled 'referee reset'), drop-balls, falls and the match stats."""
from __future__ import annotations
import importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from jhs.soccer import field as F, brain as SB, world as WD, vision  # noqa: E402

REPO = Path(os.environ.get("JUMPER_REPO", str(HERE.parent.parent / "toolkit")))
SEED = int(os.environ.get("VS_SEED", "1")); TMAX = float(os.environ.get("VS_TMAX", "300"))
OUTD = Path(os.environ.get("VS_OUT", "soccer_run")); OUTD.mkdir(parents=True, exist_ok=True)
SAVEVIS = os.environ.get("VS_SAVEVIS", "1") == "1"
SETUP = json.load(open(os.environ["SC_SETUP"])) if os.environ.get("SC_SETUP") else {}
BALL0 = [float(v) for v in (SETUP.get("ball") or [0.0, 0.0])]
START = {k: [float(v) for v in (SETUP.get(k) or F.KICKOFF[k])] for k in ("A", "B")}
VW, VH, VIS_DT = 640, 480, 0.1
GOAL_PAUSE, DROP_AFTER = 1.5, 20.0
RETURN_TIMEOUT, RET_POS, RET_YAW = 25.0, 0.15, math.radians(15)   # crabs walk back to kickoff; late crab teleported after the timeout
GOALS_TO_WIN = int(os.environ.get("SC_GOALS", "3"))      # first to this many goals wins (the time limit stays as a cap)

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
        for part in (c.get("press", "") or "").split("+"):     # the app's own buttons (as the hide & seek brain presses them)
            if part == "dpad_up": hy = -1
            elif part == "dpad_down": hy = 1
            elif part: btn.add(part)
        self.s = controller.GamepadState(lx=float(c.get("lx", 0)), ly=float(c.get("ly", 0)), rx=float(c.get("rx", 0)), ry=0.0, lt=0.0, rt=0.0,
                                         hat_x=hx, hat_y=hy, buttons=frozenset(btn))


class Celebration:
    """the scoring crab celebrates with the official app's own moves (the same presses as the hide & seek finish):
    a goal = crab dance ~3 s; the winning goal = bow, then crab dance ~7 s. Real controller motions only."""
    def __init__(self, t, mode_of, win):
        self.mode_of = mode_of; self.t0 = t; self.done = False
        steps = [("wait", 0.3)]
        if win: steps += [("press", "dpad_down", 0.2), ("until", "gesture_bow", 3.0), ("until", "locomotion", 12.0), ("wait", 0.4)]
        steps += [("press", "menu+dpad_up", 0.2), ("until", "dance_crab", 3.0), ("wait", 7.0 if win else 3.0),
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
PADS = {"A": VirtualPad("virtual pad RED (soccer brain, sim-only)"), "B": VirtualPad("virtual pad BLUE (soccer brain, sim-only)")}
controller.open = lambda *a, **k: PADS["A"]


def yaw_of(q): return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
    import mujoco, imageio
    from scipy import ndimage
    un = env.unwrapped; sim = un.sim
    robs = {"A": un.scene["robot"], "B": un.scene["robot2"]}
    tofs = {"A": un.scene["tof"], "B": un.scene["tof2"]}
    # robot B into its spawn pose, and its own controller (the same app) on its own pad
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
    base = {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PFX[k] + "base_link") for k in PFX}
    ball_b = body("prop:football"); bq = model.jnt_qposadr[model.body_jntadr[ball_b]]; bd = model.jnt_dofadr[model.body_jntadr[ball_b]]
    pb = {k: body(v[0]) for k, v in F.PROPS.items() if body(v[0]) is not None}
    # tof ray casts only when a vision tick needs them (as in hide & seek)
    DUE = {"due": True}
    for s in tofs.values():
        _k = s.raycast_kernel
        def _when(*a, _k=_k, _s=s, **kw):
            if DUE["due"]: _s._force[:] = True; return _k(*a, **kw)
        s.raycast_kernel = _when

    def set_ball(xy):
        model, live = sim.env_mjdata(0)
        live.qpos[bq:bq + 3] = [xy[0], xy[1], F.BALL_R + 0.003]; live.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]; live.qvel[bd:bd + 6] = 0
        mujoco.mj_forward(model, live)
        if hasattr(sim, "_gather"): sim._gather()

    def place_robot(k, x, y, deg):
        """referee teleport: write the crab's free joint straight into the live MjData (like set_ball; the entity
        write_root_state_to_sim path did not move the crab on the native backend)."""
        model, live = sim.env_mjdata(0); jn = model.body_jntadr[base[k]]
        qa, da = model.jnt_qposadr[jn], model.jnt_dofadr[jn]; h = math.radians(deg) / 2
        live.qpos[qa:qa + 2] = [x, y]; live.qpos[qa + 2] = max(float(live.qpos[qa + 2]), 0.11)
        live.qpos[qa + 3:qa + 7] = [math.cos(h), 0.0, 0.0, math.sin(h)]; live.qvel[da:da + 6] = 0
        mujoco.mj_forward(model, live)
        if hasattr(sim, "_gather"): sim._gather()

    def cur_props():      # referee (SIM TRUTH): obstacles on the field now, {id: (x, y, deg)}
        model, live = sim.env_mjdata(0); out = {}
        for k, b in pb.items():
            x, y = float(live.xpos[b][0]), float(live.xpos[b][1])
            if abs(x) < F.FX + 0.2 and abs(y) < F.FY + 0.2: out[k] = (x, y, math.degrees(yaw_of(live.xquat[b])))
        return out

    # spawn sanity (obstacles may go anywhere): ball / crabs that overlap an obstacle start at the nearest free spot; a pushable
    # obstacle overlapping another one is lifted so it drops on top instead of exploding
    P0 = {k: tuple(float(c) for c in (list(v) + [0.0])[:3]) for k, v in (SETUP.get("props") or {}).items() if k in F.PROPS}
    nb = F.free_spot(BALL0[0], BALL0[1], P0, F.BALL_R + 0.03)
    if tuple(nb) != tuple(BALL0): print(f"[setup] ball overlaps an obstacle -> moved to {nb}"); BALL0[:] = list(nb)
    for k in ("A", "B"):
        fx, fy = F.free_spot(START[k][0], START[k][1], P0, 0.26)
        if (fx, fy) != (START[k][0], START[k][1]):
            print(f"[setup] {F.TEAM[k]} crab overlaps an obstacle -> starts at ({fx}, {fy})"); START[k] = [fx, fy, START[k][2]]
    _m, _l = sim.env_mjdata(0); ks = list(P0)
    for i, k in enumerate(ks):
        if any(F._box_dist((P0[k][0], P0[k][1]), k2, *P0[k2]) < max(F.PROPS[k][2:]) for k2 in ks[:i]):
            j = _m.body_jntadr[pb[k]]
            if j >= 0 and _m.jnt_type[j] == 0: _l.qpos[_m.jnt_qposadr[j] + 2] += 0.45; print(f"[setup] {k} overlaps another obstacle -> lifted, drops on top")
    mujoco.mj_forward(_m, _l)
    set_ball(BALL0)
    # log only (SIM TRUTH): crab-body contacts with props / walls / goals ("leg bumps"), counted as contact onsets per crab
    _pbset = set(pb.values()); _gn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    obst_geom = np.array([(_root(model.geom_bodyid[g]) in _pbset) or _gn[g].startswith(("wall_", "corner_", "goal")) for g in range(model.ngeom)])
    bumps = {"A": 0, "B": 0}; bump_prev = {"A": False, "B": False}
    rng = np.random.default_rng(SEED)
    brains = {k: SB.Brain(k, np.random.default_rng(SEED * 31 + i), pushable=os.environ.get("SC_PUSHABLE", "1") == "1") for i, k in enumerate(("A", "B"))}
    for b in brains.values(): b.reset(kickoff=True, t=0.0); b.ball = np.array(BALL0)
    HOOK = None
    if os.environ.get("SC_LIVE", "0") == "1" or os.environ.get("JHS_LIVE_DIR"):
        from jhs.soccer.live import SoccerHook
        HOOK = SoccerHook(sim=sim, outd=OUTD, dt=un.step_dt)
    print(f"[setup] soccer seed={SEED} tmax={TMAX} ball={BALL0} A={START['A']} B={START['B']} props={SETUP.get('props', {})} skins={nsk} "
          f"cam {VW}x{VH} tof {tw}x{th}", flush=True)
    dt = float(un.step_dt); vis_every = max(1, round(VIS_DT / dt)); q_every = max(1, round(0.04 / dt))
    obs = env.get_observations(); step = 0
    score = {"A": 0, "B": 0}; goals = []; events = []; falls = {"A": 0, "B": 0}; fallen = {"A": False, "B": False}
    cele = None; final = None; winner = None; end_at = None
    modes = {"A": lambda: pA.fsm.mode, "B": lambda: pB.fsm.mode}
    settle_until = 1.0; paused_s = 0.0; prev_t = 0.0; returns = []
    spots = {}
    def spot_of(k):
        if k in spots: return spots[k]
        x0, y0, h0 = F.KICKOFF[k]
        return (-F.ATTACK[k] * F.KICKOFF_NEAR if k == kicker else x0, y0, h0)
    def plan_spots():     # referee: ball on the nearest free spot to the centre, each crab's nearest free spot to its kickoff spot
        spots.clear(); P = cur_props(); bc = F.free_spot(0.0, 0.0, P, F.BALL_R + 0.05)
        for k in ("A", "B"): x0, y0, h0 = spot_of(k); spots[k] = (*F.free_spot(x0, y0, P, 0.27), h0)
        if math.hypot(spots["A"][0] - spots["B"][0], spots["A"][1] - spots["B"][1]) < 0.45:
            spots["B"] = (*F.free_spot(spots["B"][0] + 0.5 * F.ATTACK["A"], spots["B"][1], P, 0.27), spots["B"][2])
        return bc
    pending = None; kicker = "B"         # A kicks off first (kicker = who conceded last); alternate
    last_active = 0.0; stuck_s = 0.0; drops = 0; prev_ball = np.array(BALL0); touches = {"A": 0, "B": 0}; near_prev = {"A": False, "B": False}
    last_touch = None; ref_line = "kickoff: RED"
    qbuf, qt, vis_meta, tofA, tofB = [], [], [], [], []
    truth = open(OUTD / "truth.jsonl", "w"); t_wall = time.time(); stopped = False; t = 0.0
    cmds = {"A": {}, "B": {}}; per = {"A": None, "B": None}
    def ev(kind, **kw):
        e = dict(t=round(t, 2), event=kind, **kw); events.append(e); print(f"[ref] {e}", flush=True)
    try:
        while True:
            t = step * dt
            if end_at is not None and t >= end_at: break
            if pending is not None and pending[0] in ("celebrate", "return"): paused_s += t - prev_t    # match clock pauses
            prev_t = t; mt = t - paused_s
            if mt >= TMAX and final is None:      # time limit (a cap): higher score wins, else a draw
                winner = "A" if score["A"] > score["B"] else "B" if score["B"] > score["A"] else None
                final = (f"{F.TEAM[winner]} WINS {score['A']}-{score['B']}" if winner else f"DRAW {score['A']}-{score['B']}") + " (time)"
                ref_line = f"full time: {final}"; ev("full_time", final=final); end_at = t + 3.0; pending = ("over", 1e9)
            if step == 2:     # the second crab's init_state is not honoured by the env reset (it spawned at the centre): place both now
                for k in ("A", "B"): place_robot(k, *START[k])
                set_ball(BALL0)
            model, live = sim.env_mjdata(0)
            if step % vis_every == 0 and t > 0.3:
                metas = {}; poses = {}
                for k in ("A", "B"):
                    c_id, t_id = cams[k]
                    rend.update_scene(live, camera=c_id); img = rend.render().copy()
                    rend.enable_segmentation_rendering(); rend.update_scene(live, camera=c_id); seg = rend.render()[..., 0].copy(); rend.disable_segmentation_rendering()
                    selfm = ndimage.binary_dilation(own_geom[k][np.clip(seg, 0, own_geom[k].size - 1)] & (seg >= 0), iterations=3)
                    rngs = tofs[k].data.range[0].detach().cpu().numpy().reshape(th, tw)
                    cpos = live.cam_xpos[c_id].copy(); cR = live.cam_xmat[c_id].reshape(3, 3).copy()
                    tpos = live.cam_xpos[t_id].copy(); tR = live.cam_xmat[t_id].reshape(3, 3).copy()
                    p = SB.perceive(k, img, selfm, cpos, cR, cam, rngs, tpos, tR, tcam)
                    if t < 2.0 and step % (vis_every * 5) == 0:    # kickoff check in the log: does each crab see the ball / the other crab?
                        print(f"[kickoff-vision] t={t:.1f} {F.TEAM[k]} ball={[tuple(np.round(d['xy'], 2)) for d in p['ball']]} opp={len(p['opp'])}", flush=True)
                    q = live.xquat[base[k]]; pose = (float(live.xpos[base[k]][0]), float(live.xpos[base[k]][1]), yaw_of(q))   # own pose
                    br = brains[k]; br.observe(t, p, pose)
                    poses[k] = pose
                    if pending is not None and pending[0] == "return": cmds[k] = br.return_cmd(pose)
                    elif pending is None:
                        try: cmds[k] = br.decide(pose)
                        except Exception as e:      # a brain bug must not end the match: stand still this tick, log it
                            cmds[k] = {}; ev("brain_error", team=F.TEAM[k], error=repr(e)[:200])
                    elif pending[0] == "celebrate" and k == pending[2]: cmds[k] = cele.cmd(t)
                    else: cmds[k] = {}
                    PADS[k].set(cmds[k])
                    _cel = pending is not None and pending[0] == "celebrate" and k == pending[2]
                    metas[k] = dict(state=br.state if pending is None or pending[0] == "return" else ("celebrate" if _cel else "paused" if pending[0] == "celebrate" else "referee"),
                                    sub=br.sub if pending is None or pending[0] == "return" else (cele.label if _cel else "standing still while the scorer celebrates" if pending[0] == "celebrate" else "referee reset" if pending[0] == "kickoff" else "full time"),
                                    pose=[round(v, 3) for v in pose], cmd={a: (round(float(b_), 2) if a != "press" else b_) for a, b_ in cmds[k].items()},
                                    ball=[round(float(v), 3) for v in br.ball] if br.ball_age() < 3.0 else None, ball_age=round(min(br.ball_age(), 99), 1),
                                    opp=[round(float(v), 3) for v in br.opp] if br.opp is not None and t - br.opp_t < 3.0 else None,
                                    dets=[dict(cls="ball", bbox=d["bbox"], rng=round(d["rng"], 2)) for d in p["ball"]] +
                                         [dict(cls="opponent", bbox=d["bbox"], rng=round(d["rng"], 2)) for d in p["opp"]] +
                                         [dict(cls="goal_" + ("red" if d["team"] == "A" else "blue"), bbox=d["bbox"], rng=round(d["rng"], 2)) for d in p["goals"]],
                                    nobst=int(p["obst"].shape[0]))
                    if SAVEVIS:
                        (OUTD / "vis").mkdir(exist_ok=True)
                        imageio.imwrite(OUTD / "vis" / f"{k}_{len(vis_meta):05d}.jpg", img[::2, ::2], quality=85)
                        (tofA if k == "A" else tofB).append(rngs.astype(np.float16))
                    if HOOK is not None: HOOK.vision(k, img, rngs)
                meta = dict(t=round(t, 2), clock=round(mt, 2), A=metas["A"], B=metas["B"], score=dict(score), ref=ref_line, final=final, goals_to_win=GOALS_TO_WIN,
                            ball_truth=[round(float(v), 3) for v in live.xpos[ball_b][:2]])
                if SAVEVIS: vis_meta.append(meta)
                if HOOK is not None: HOOK.meta(meta)
            DUE["due"] = (step + 1) % vis_every == 0
            with torch.inference_mode():
                pB(None)
                obs, _, _, _ = env.step(pA(obs))
            if HOOK is not None and HOOK.step(t) == "stop":
                stopped = True; print(f"[run] stopped from the local runner at t={t:.1f}s", flush=True); break
            # ---------------- referee (sim truth) ----------------
            if step % 10 == 0:
                model, live = sim.env_mjdata(0)
                bxy = np.array(live.xpos[ball_b][:2], float); bv = (bxy - prev_ball) / (10 * dt); prev_ball = bxy
                for k in ("A", "B"):
                    z = float(live.xpos[base[k]][2]); upz = float(live.xmat[base[k]].reshape(3, 3)[2, 2])
                    isf = (z < 0.04 or upz < 0.5) and t > settle_until
                    if isf and not fallen[k]: falls[k] += 1; ev("fall", team=F.TEAM[k], z=round(z, 3))
                    fallen[k] = isf
                    near = float(np.linalg.norm(live.xpos[base[k]][:2] - bxy)) < 0.24
                    if near and not near_prev[k]: touches[k] += 1; last_touch = k
                    near_prev[k] = near
                    if near: last_active = t
                _hit = {"A": False, "B": False}
                for ci in range(live.ncon):
                    g1, g2 = live.contact[ci].geom1, live.contact[ci].geom2
                    for k in ("A", "B"):
                        if (own_geom[k][g1] and obst_geom[g2]) or (own_geom[k][g2] and obst_geom[g1]): _hit[k] = True
                for k in ("A", "B"):
                    if _hit[k] and not bump_prev[k]: bumps[k] += 1
                    bump_prev[k] = _hit[k]
                if np.linalg.norm(bv) > 0.04: last_active = t
                elif t - last_active > 3.0: stuck_s += 10 * dt
                if pending is None:
                    side = F.in_goal(*bxy)
                    if side:
                        scorer = "A" if side > 0 else "B"; score[scorer] += 1
                        own = last_touch is not None and last_touch != scorer
                        goals.append(dict(t=round(t, 1), team=F.TEAM[scorer], own_goal=bool(own), last_touch=F.TEAM.get(last_touch)))
                        ref_line = f"GOAL {F.TEAM[scorer]}{' (own goal)' if own else ''} at {t:.0f}s"
                        ev("goal", team=F.TEAM[scorer], own_goal=bool(own), score=dict(score)); kicker = "A" if scorer == "B" else "B"
                        win = score[scorer] >= GOALS_TO_WIN
                        if win:
                            winner = scorer; final = f"{F.TEAM[scorer]} WINS {score['A']}-{score['B']}"; ref_line = final; ev("match_won", final=final)
                        cele = Celebration(t, modes[scorer], win); pending = ("celebrate", None, scorer)
                    elif t - last_active > DROP_AFTER:
                        drops += 1; c = bxy * 0.7; c[1] = float(np.clip(c[1], -F.FY + 0.3, F.FY - 0.3))
                        c = c + rng.uniform(-0.1, 0.1, 2); c = np.array(F.free_spot(float(c[0]), float(c[1]), cur_props(), F.BALL_R + 0.05))
                        set_ball(c); last_active = t; ref_line = f"referee drop-ball at {t:.0f}s (ball idle {DROP_AFTER:.0f} s)"
                        ev("drop_ball", at=[round(float(v), 2) for v in c])
                elif pending[0] == "celebrate":
                    if cele.done or t - cele.t0 > 30.0:
                        if winner is not None: end_at = t + 0.5; pending = ("over", 1e9)
                        else:      # ball to the centre (referee); both crabs walk back to their kickoff spots by themselves
                            bc = plan_spots(); set_ball(bc); last_active = t; last_touch = None
                            ref_line = "referee: ball to centre" + ("" if bc == (0.0, 0.0) else f" (nearest free spot {bc[0]:.2f}, {bc[1]:.2f})")
                            ev("ball_to_centre", at=list(bc), spots={F.TEAM[k]: spots[k] for k in spots}); pending = ("return", t, bc)
                            for k in ("A", "B"): brains[k].begin_return(spot_of(k), t, bc)
                elif pending[0] == "return":
                    ok = {}
                    for k in ("A", "B"):
                        sx, sy, sh = spot_of(k); x, y, yaw = poses[k]
                        ok[k] = math.hypot(x - sx, y - sy) < RET_POS and abs(SB.wrap(math.radians(sh) - yaw)) < RET_YAW
                    late = [k for k in ok if not ok[k]] if t - pending[1] > RETURN_TIMEOUT else None
                    if all(ok.values()) or late:
                        for k in (late or []): place_robot(k, *spot_of(k))
                        if np.linalg.norm(bxy - np.array(pending[2])) > 0.05: set_ball(pending[2]); ev("ball_recentred")
                        returns.append(dict(t=round(t - pending[1], 1), timeout=[F.TEAM[k] for k in (late or [])]))
                        last_active = t; last_touch = None
                        for b in brains.values(): b.reset(kickoff=True, t=t); b.ball = np.array(pending[2], float)
                        spots.clear()
                        if late:
                            ref_line = f"referee reset (timeout): {', '.join(F.TEAM[k] for k in late)} placed on kickoff spot; kickoff {F.TEAM[kicker]}"
                            ev("referee_reset_timeout", late=[F.TEAM[k] for k in late], kickoff=F.TEAM[kicker]); settle_until = t + 1.0
                        else:
                            ref_line = f"both crabs walked back ({t - pending[1]:.1f} s); kickoff {F.TEAM[kicker]} ({score['A']}-{score['B']})"
                            ev("kickoff_after_return", took=round(t - pending[1], 1), kickoff=F.TEAM[kicker])
                        pending = None
                elif pending[0] == "kickoff" and t >= pending[1]:
                    # referee reset: ball to the centre spot, crabs teleported to their kickoff spots
                    other = "A" if kicker == "B" else "B"
                    for k in ("A", "B"):
                        x0, y0, h0 = F.KICKOFF[k]
                        if k == kicker: x0 = -F.ATTACK[k] * F.KICKOFF_NEAR
                        place_robot(k, x0, y0, h0)
                    set_ball((0.0, 0.0)); last_active = t; last_touch = None
                    for b in brains.values(): b.reset(kickoff=True, t=t)
                    ref_line = f"referee reset: kickoff {F.TEAM[kicker]} ({score['A']}-{score['B']})"
                    ev("referee_reset", kickoff=F.TEAM[kicker]); pending = None; settle_until = t + 1.0
                    kicker = other if False else kicker
                if step % 20 == 0:
                    truth.write(json.dumps({"t": round(t, 2), "ball": [round(float(v), 3) for v in live.xpos[ball_b][:3]],
                                            **{k: [round(float(v), 3) for v in live.xpos[base[k]][:3]] for k in ("A", "B")},
                                            **{k: [round(float(v), 3) for v in live.xpos[b][:2]] for k, b in pb.items()}}) + "\n")
            if step % q_every == 0: qbuf.append(np.array(live.qpos, dtype=np.float64)); qt.append(t)
            step += 1
            if step % 2000 == 0:
                print(f"[run] t={t:.0f}s wall={time.time() - t_wall:.0f}s score {score['A']}-{score['B']} "
                      f"A={brains['A'].state} B={brains['B'].state} ball={np.round(live.xpos[ball_b][:2], 2)} falls={falls}", flush=True)
    finally:
        truth.close()
        res = dict(kind="soccer", label="SIM-ONLY VISION CONCEPT", seed=SEED, tmax=TMAX, setup=dict(ball=BALL0, A=START["A"], B=START["B"], props=SETUP.get("props", {})),
                   t_end=round(step * dt, 2), stopped=stopped, score={"RED": score["A"], "BLUE": score["B"]}, goals=goals, goals_to_win=GOALS_TO_WIN, leg_bumps={F.TEAM[k]: v for k, v in bumps.items()}, returns=returns, match_clock_s=round(mt, 1),
                   winner=F.TEAM.get(winner) if final else None, final=final,
                   own_goals=sum(g["own_goal"] for g in goals), falls={"RED": falls["A"], "BLUE": falls["B"]}, drop_balls=drops,
                   ball_idle_s=round(stuck_s, 1), touches={"RED": touches["A"], "BLUE": touches["B"]},
                   brain_counts={F.TEAM[k]: dict(b.counts) for k, b in brains.items()}, wall_s=round(time.time() - t_wall, 1))
        json.dump(res, open(OUTD / "result.json", "w"), indent=1)
        with open(OUTD / "events.jsonl", "w") as f:
            for e in events: f.write(json.dumps(e) + "\n")
        if qbuf: np.savez_compressed(OUTD / "qpos.npz", qpos=np.stack(qbuf), t=np.array(qt), fps=25.0)
        if SAVEVIS and vis_meta:
            n = min(len(tofA), len(tofB), len(vis_meta))
            np.savez_compressed(OUTD / "vision_tof.npz", tofA=np.stack(tofA[:n]), tofB=np.stack(tofB[:n]))
            json.dump(vis_meta[:n], open(OUTD / "vision_meta.json", "w"))
        print("[result] " + json.dumps(res), flush=True)


play._loop = _loop
sys.argv[0] = str(REPO / "scripts" / "play.py")
if __name__ == "__main__":
    play.main()
