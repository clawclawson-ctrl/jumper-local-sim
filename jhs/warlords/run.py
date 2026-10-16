"""3-crab Crab Warlords in the local sim. SIM-ONLY VISION CONCEPT.
    python jhs/warlords/run.py --app jumper_warlords_vision.app --scene RUN/scene.map --backend native --device cpu --headless --steps 999999999
Env: WL_SETUP (json {openings, camera}), VS_SEED, VS_TMAX, VS_OUT, VS_SAVEVIS, WL_LIVE=1, WL_ROUNDS (1-10), WL_AGGR[_A|_B|_C], JUMPER_REPO.
Each crab = the official walking app driven by its own Warlords Brain through a virtual gamepad, from ITS OWN camera + dToF + pose
(+ the public scoreboard: bricks left per castle). The REFEREE (brick breaks, king hits, serves, rounds, resets) uses sim truth."""
from __future__ import annotations
import os
import importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from jhs.soccer import vision, brain as SBR  # noqa: E402
from jhs.warlords import game as G, brain as WB, world as WW, fx as FXW  # noqa: E402

REPO = Path(os.environ.get("JUMPER_REPO", str(HERE.parent.parent / "toolkit")))
SEED = int(os.environ.get("VS_SEED", "1")); TMAX = float(os.environ.get("VS_TMAX", "300"))
OUTD = Path(os.environ.get("VS_OUT", "warlords_run")); OUTD.mkdir(parents=True, exist_ok=True)
SETUP = json.load(open(os.environ["WL_SETUP"])) if os.environ.get("WL_SETUP") else {}
ROUNDS_TO_WIN = max(1, min(10, int(os.environ.get("WL_ROUNDS", "3"))))
AGGR = G.aggression_from_env(os.environ)
OPEN_CFG = dict(SETUP.get("openings") or {}); OPEN_CFG.setdefault("mode", "random")
OPEN_SEED = int(OPEN_CFG["seed"]) if OPEN_CFG.get("seed") not in (None, "") else int.from_bytes(os.urandom(4), "little")
ORNG = np.random.default_rng(OPEN_SEED); ROUND = G.roll_round(ORNG, OPEN_CFG)
START = {k: list(G.jittered_start(k, ROUND[k]["jit"])) for k in G.KEYS}
VW, VH, VIS_DT = 640, 480, 0.1
PARK = {"A": (-3.9, 3.2, 0.0), "B": (3.9, 3.2, 180.0), "C": (-3.9, 4.4, 0.0)}   # eliminated crabs wait here, outside the walls
K3 = G.KEYS
TEST = os.environ.get("WL_TEST") == "castle"     # referee test scene: crabs stand still, the ball is rolled at RED's castle three times

sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("play", REPO / "scripts" / "play.py")
play = importlib.util.module_from_spec(_spec); sys.modules["play"] = play; _spec.loader.exec_module(play)
import controller  # noqa: E402
import tasks  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("VS_THREADS", "2"))); torch.manual_seed(SEED)
WW.install(tasks, START["A"], START["B"], START["C"], SEED)
from jhs.soccer import world as WD  # noqa: E402
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


PADS = {k: VirtualPad(f"virtual pad {G.TEAM[k]} (warlords brain, sim-only)") for k in K3}
controller.open = lambda *a, **k: PADS["A"]
def yaw_of(q): return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def perceive(k, img, selfmask, cpos, cR, cam):
    """ball (white blob) + crab blobs (team colour families) from this crab's own camera."""
    dets = [d for d in vision.detect(img, cpos, cR, cam, selfmask=selfmask) if d["cls"] == "football"]
    crabs = []
    for j in K3:
        if j == k: continue
        crabs += [dict(d, team=j) for d in SBR.colour_blobs(img, cpos, cR, cam, selfmask, FXW.FAMS[j]) if d["kind"] == "crab"
                  and min(np.linalg.norm(np.asarray(d["xy"]) - G.king(q)) for q in K3) > G.castle_zone("A") + 0.1]
    ball = [dict(d, xy=np.asarray(d["xy"], float)) for d in dets if G.tri_clear(*d["xy"]) > 0.0
            and not any(np.linalg.norm(np.asarray(d["xy"]) - c["xy"]) < 0.22 for c in crabs)]
    return dict(ball=ball, crabs=crabs)


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
    global ROUND
    import mujoco, imageio
    from scipy import ndimage
    un = env.unwrapped; sim = un.sim
    tofs = {"A": un.scene["tof"], "B": un.scene["tof2"], "C": un.scene["tof3"]}
    app = Path(sys.argv[sys.argv.index("--app") + 1]); players = {"A": policy}
    for k, rn in (("B", "robot2"), ("C", "robot3")):
        rb = un.scene[rn]; rs = rb.data.default_root_state.clone(); rs[:, :3] += un.scene.env_origins
        rb.write_root_state_to_sim(rs); rb.write_joint_state_to_sim(rb.data.default_joint_pos.clone(), torch.zeros_like(rb.data.default_joint_vel))
        players[k] = ap.AppPlayer(ap.open_app(app), WD.EnvView(env, rb), pad=PADS[k])
    mgr = un.action_manager; names_ = list(mgr.active_terms)
    def allp():
        for p in players.values(): p._apply()
    mgr.get_term(names_[0]).apply_actions = allp
    model, live = sim.env_mjdata(0)
    WW.skin(model); FXW.theme(model)
    rmodel = getattr(sim, "render_model", None) or model
    if rmodel is not model: WW.skin(rmodel); FXW.theme(rmodel)
    model.vis.global_.offwidth = max(VW, model.vis.global_.offwidth); model.vis.global_.offheight = max(VH, model.vis.global_.offheight)
    rend = mujoco.Renderer(model, VH, VW, max_geom=12000)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
    def _root(b):
        while model.body_parentid[b] != 0: b = model.body_parentid[b]
        return b
    groot = [names[_root(model.geom_bodyid[g])] for g in range(model.ngeom)]
    PFX = {"A": "robot/", "B": "robot2/", "C": "robot3/"}
    own_geom = {k: np.array([n.startswith(p) for n in groot] + [False]) for k, p in PFX.items()}
    cams = {k: (model.camera(PFX[k] + "onboard").id, model.camera(PFX[k] + "tof").id) for k in PFX}
    cam = vision.Camera(VW, VH, float(model.cam_fovy[cams["A"][0]]))
    tw, th = (int(v) for v in model.cam_resolution[cams["A"][1]])
    base = {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PFX[k] + "base_link") for k in PFX}
    gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    BRICK = {k: [gid(f"wl_brick_{k}_{i}") for i in range(len(G.bricks(k)))] for k in K3}; KING = {k: gid(f"wl_king_{k}") for k in K3}
    G2B = {g: (k, i) for k in K3 for i, g in enumerate(BRICK[k])}; G2K = {g: k for k, g in KING.items()}
    ball_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "prop:football"); ball_g = {g for g in range(model.ngeom) if model.geom_bodyid[g] == ball_b}
    bj = model.body_jntadr[ball_b]; bqa, bda = model.jnt_qposadr[bj], model.jnt_dofadr[bj]
    CT = {g: (int(model.geom_contype[g]), int(model.geom_conaffinity[g])) for g in G2B}
    DUE = {"due": True}
    for s in tofs.values():
        _k = s.raycast_kernel
        def _when(*a, _k=_k, _s=s, **kw):
            if DUE["due"]: _s._force[:] = True; return _k(*a, **kw)
        s.raycast_kernel = _when
    def sync():
        if hasattr(sim, "_gather"): sim._gather()
    def place_robot(k, x, y, deg):
        model, live = sim.env_mjdata(0); jn = model.body_jntadr[base[k]]
        qa, da = model.jnt_qposadr[jn], model.jnt_dofadr[jn]; h = math.radians(deg) / 2
        live.qpos[qa:qa + 2] = [x, y]; live.qpos[qa + 2] = max(float(live.qpos[qa + 2]), 0.11)
        live.qpos[qa + 3:qa + 7] = [math.cos(h), 0.0, 0.0, math.sin(h)]; live.qvel[da:da + 6] = 0
        mujoco.mj_forward(model, live); sync()
    def serve(a):
        model, live = sim.env_mjdata(0)
        live.qpos[bqa:bqa + 3] = [0.0, 0.0, G.BALL_R + 0.003]; live.qpos[bqa + 3:bqa + 7] = [1, 0, 0, 0]
        live.qvel[bda:bda + 6] = 0; live.qvel[bda:bda + 2] = [G.SERVE_V * math.cos(a), G.SERVE_V * math.sin(a)]
        mujoco.mj_forward(model, live); sync()
    def set_brick(k, i, on):
        g = BRICK[k][i]
        for M in {id(model): model, id(rmodel): rmodel}.values():
            gg = g if M is model else mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_GEOM, f"wl_brick_{k}_{i}")
            M.geom_contype[gg], M.geom_conaffinity[gg] = (CT[g] if on else (0, 0)); M.geom_rgba[gg][3] = 1.0 if on else 0.0
    def park_props():
        model, live = sim.env_mjdata(0)
        for i_, b in enumerate(b for b in range(model.nbody) if names[b].startswith("prop:") and b != ball_b):
            j = model.body_jntadr[b]
            if j < 0 or model.jnt_type[j] != 0: continue
            qa, da = model.jnt_qposadr[j], model.jnt_dofadr[j]
            live.qpos[qa:qa + 3] = [-2.0 + 1.0 * i_, -4.6, 0.3]; live.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]; live.qvel[da:da + 6] = 0
        mujoco.mj_forward(model, live); sync()
    def truth_pose(k):
        _m, lv = sim.env_mjdata(0); return (float(lv.xpos[base[k]][0]), float(lv.xpos[base[k]][1]), yaw_of(lv.xquat[base[k]]))

    brains = {k: WB.Brain(k, np.random.default_rng(SEED * 31 + i), aggression=AGGR[k] / 100) for i, k in enumerate(K3)}
    for k in K3: brains[k].set_opening(ROUND[k], 0.0)
    castles = G.Castles()
    HOOK = None
    if os.environ.get("WL_LIVE", "0") == "1" or os.environ.get("JHS_LIVE_DIR"):
        from jhs.warlords.live import WarlordsHook
        HOOK = WarlordsHook(sim=sim, outd=OUTD, dt=un.step_dt)
    OPEN_LOG = [dict(round=1, **{G.TEAM[k]: ROUND[k]["opener"] for k in K3})]
    print(f"[setup] warlords seed={SEED} tmax={TMAX} rounds_to_win={ROUNDS_TO_WIN} aggression {AGGR} openers {[ROUND[k]['opener'] for k in K3]} (seed {OPEN_SEED})", flush=True)
    dt = float(un.step_dt); vis_every = max(1, round(VIS_DT / dt))
    obs = env.get_observations(); step = 0
    score = {k: 0 for k in K3}; events = []; falls = {k: 0 for k in K3}; fallen = {k: False for k in K3}
    final = None; winner = None; end_at = None; settle_until = 1.5; rounds = []; round_no = 1; round_t0 = 1.5
    out = {k: None for k in K3}; pending = None; ref_line = "GO!"; banner = None; served = False
    stall = 0.0; serves = 0; still = {k: 0.0 for k in K3}; max_still = {k: 0.0 for k in K3}; rot_only = {k: 0.0 for k in K3}; max_rot = {k: 0.0 for k in K3}
    t_wall = time.time(); stopped = False; t = 0.0; cmds = {k: {} for k in K3}; P = {k: tuple(START[k][:2]) + (math.radians(START[k][2]),) for k in K3}
    broken_n = 0; cracked = set()
    def ev(kind, **kw):
        e = dict(t=round(t, 2), event=kind, **kw); events.append(e); print(f"[ref] {e}", flush=True)
    def end_round(how, rw):
        nonlocal pending, winner, final, ref_line, banner, end_at
        rounds.append(dict(round=round_no, t=round(t - round_t0, 1), how=how, winner=G.TEAM.get(rw), bricks={G.TEAM[k]: castles.left(k) for k in K3},
                           out={G.TEAM[k]: v for k, v in out.items() if v}))
        if rw is not None: score[rw] += 1
        ref_line = (f"round {round_no}: {G.TEAM[rw]} takes it ({how})" if rw else f"round {round_no}: DRAW ({how})") + "  " + "-".join(str(score[k]) for k in K3)
        banner = f"{G.TEAM[rw]} TAKES ROUND {round_no}" if rw else f"ROUND {round_no} DRAW"
        ev("round_end", round=round_no, how=how, winner=G.TEAM.get(rw), score=dict(score))
        if rw is not None and score[rw] >= ROUNDS_TO_WIN: winner = rw; final = f"{G.TEAM[rw]} WINS"; ref_line = final; ev("match_won", final=final)
        pending = ("pause", t)
    try:
        while True:
            t = step * dt
            if end_at is not None and t >= end_at: break
            if t >= TMAX and final is None:
                winner = max(K3, key=lambda k: score[k]) if sorted(score.values())[-1] > sorted(score.values())[-2] else None
                final = (f"{G.TEAM[winner]} WINS" if winner else "DRAW") + " (time)"; ref_line = f"time: {final}"; ev("full_time", final=final)
                end_at = t + 2.0; pending = ("over", t)
            if step == 2:
                for k in K3: place_robot(k, *(PARK[k] if TEST else START[k]))      # test scene: crabs wait outside, the line to RED's king is clear
                park_props(); serve(-math.pi / 2); 
                model, live = sim.env_mjdata(0); live.qvel[bda:bda + 6] = 0; mujoco.mj_forward(model, live); sync()
            model, live = sim.env_mjdata(0)
            if step % vis_every == 0 and t > 0.3:
                metas = {}; bpos = [float(v) for v in live.qpos[bqa:bqa + 2]]
                for k in K3:
                    pose = truth_pose(k); P[k] = pose; br = brains[k]
                    br.scoreboard = {j: castles.left(j) for j in K3}; br.gaps = {j: castles.gaps(j) for j in K3}; br.out = {j: bool(out[j]) for j in K3}
                    if out[k] is None:
                        c_id, t_id = cams[k]
                        rend.update_scene(live, camera=c_id); img = rend.render().copy()
                        rend.enable_segmentation_rendering(); rend.update_scene(live, camera=c_id); seg = rend.render()[..., 0].copy(); rend.disable_segmentation_rendering()
                        selfm = ndimage.binary_dilation(own_geom[k][np.clip(seg, 0, own_geom[k].size - 1)] & (seg >= 0), iterations=3)
                        rngs = tofs[k].data.range[0].detach().cpu().numpy().reshape(th, tw)
                        p = perceive(k, img, selfm, live.cam_xpos[c_id].copy(), live.cam_xmat[c_id].reshape(3, 3).copy(), cam)
                        br.observe(t, p, pose)
                        if HOOK is not None: HOOK.vision(k, img, rngs)
                        if os.environ.get("WL_DUMP") and step % (vis_every * 10) == 0:
                            (OUTD / "vis").mkdir(exist_ok=True); imageio.imwrite(OUTD / "vis" / f"{k}_{t:06.1f}.jpg", img, quality=88)
                    else: p = dict(ball=[], crabs=[])
                    playing = pending is None and out[k] is None and t >= settle_until
                    if playing and not TEST:
                        try: cmds[k] = br.decide(pose)
                        except Exception as e: cmds[k] = {}; ev("brain_error", team=G.TEAM[k], error=repr(e)[:200])
                    else: cmds[k] = {}
                    PADS[k].set(cmds[k]); c = cmds[k]
                    mv = abs(c.get("lx", 0)) + abs(c.get("ly", 0)) >= 0.05
                    rot_only[k] = rot_only[k] + VIS_DT if playing and abs(c.get("rx", 0)) > 0.1 and not mv else 0.0; max_rot[k] = max(max_rot[k], rot_only[k])
                    still[k] = still[k] + VIS_DT if playing and not mv and br.state != "hold" else 0.0; max_still[k] = max(max_still[k], still[k])
                    st_ = "out" if out[k] else (br.state if pending is None else "paused")
                    metas[k] = dict(state=st_, sub=(f"OUT: king hit" if out[k] else br.sub), pose=[round(v, 3) for v in pose], cmd={a: round(float(b), 2) for a, b in c.items() if a != "press"},
                                    dets=[dict(cls="ball", bbox=d["bbox"], rng=round(d["rng"], 2)) for d in p["ball"]] + [dict(cls="crab", bbox=d["bbox"], rng=round(d["rng"], 2)) for d in p["crabs"]],
                                    bricks=castles.left(k), bricks_total=castles.total(k), opener=br.op_name, opener_on=br.opening is not None)
                meta = dict(t=round(t, 2), clock=round(t, 2), score=dict(score), ref=ref_line, final=final, banner=banner, rounds_to_win=ROUNDS_TO_WIN, aggression=dict(AGGR),
                            round=round_no, round_t=round(max(0.0, t - round_t0), 1) if pending is None else 0.0, ball=bpos, bricks_alive={k: list(castles.alive[k]) for k in K3}, **metas)
                if HOOK is not None: HOOK.meta(meta)
            DUE["due"] = (step + 1) % vis_every == 0
            with torch.inference_mode():
                for k in ("B", "C"): players[k](None)
                obs, _, _, _ = env.step(players["A"](obs))
            if HOOK is not None and HOOK.step(t) == "stop": stopped = True; break
            # ---------------- referee (sim truth) ----------------
            model, live = sim.env_mjdata(0)
            if TEST and pending is None and t >= settle_until:      # scripted referee test: the ball is rolled straight at RED's king
                for t_shot in (1.6, 6.5):
                    if abs(t - t_shot) < dt / 2:
                        K = G.king("A"); gp = castles.gaps("A"); u_ = (np.array(gp[0]) - K) / np.linalg.norm(np.array(gp[0]) - K) if gp else -G.U["A"]
                        q = K + 1.6 * u_; d_ = (K - q) / np.linalg.norm(K - q)          # 2nd shot: through the gap at the king
                        model, live = sim.env_mjdata(0)
                        live.qpos[bqa:bqa + 3] = [q[0], q[1], G.BALL_R + 0.003]; live.qpos[bqa + 3:bqa + 7] = [1, 0, 0, 0]; live.qvel[bda:bda + 6] = 0
                        live.qvel[bda:bda + 2] = 0.8 * d_; mujoco.mj_forward(model, live); sync(); served = True; ev("test_shot", at=[round(float(v), 2) for v in q])
            if pending is None and t >= settle_until:
                if not served: serve(ROUND["serve"]); served = True; serves += 1; ev("serve", angle=round(math.degrees(ROUND["serve"])))
                # touches from sim truth geometry (the ball's true centre vs the brick / king boxes; a contact can live inside one
                # env step's physics substeps only, so data.contact at the step end would miss it)
                bxy = np.array(live.qpos[bqa:bqa + 2], float)
                hits, kh_ = castles.ball_hit(bxy, G.BALL_R + 0.03)   # within 3 cm = touching (env steps sample the rebound coarsely)
                if TEST and step % 25 == 0: print(f"[test] t={t:.2f} ball {bxy.round(3).tolist()} v {np.round(live.qvel[bda:bda + 2], 2).tolist()} "
                                                  f"near-brick {min(G.brick_dist(bxy, b_) for b_ in castles.B['A']):.3f}", flush=True)
                for (k, i) in hits: cracked.add((k, i))                 # touched: it breaks once the ball has bounced off it (real rebound)
                for (k, i) in list(cracked):
                    if G.brick_dist(bxy, castles.B[k][i]) > G.BALL_R + 0.03:
                        cracked.discard((k, i)); castles.alive[k][i] = False; set_brick(k, i, False); broken_n += 1; ev("brick", castle=G.TEAM[k], i=i, left=castles.left(k))
                for kk in ([kh_] if kh_ else []):
                    if out[kk] is None:
                        kk = kk; out[kk] = round(t, 1); ev("king_hit", team=G.TEAM[kk]); place_robot(kk, *PARK[kk])
                        alive = [k for k in K3 if out[k] is None]
                        if len(alive) == 1: end_round("king hit", alive[0])
                        else: serve(float(ORNG.uniform(0, 2 * math.pi))); serves += 1
                        break
                if pending is None and step % 10 == 0:
                    vb = float(np.hypot(*live.qvel[bda:bda + 2])); stall = stall + 10 * dt if vb < 0.04 else 0.0
                    if stall > G.STALL_S and not TEST: serve(float(ORNG.uniform(0, 2 * math.pi))); serves += 1; stall = 0.0; ev("reserve")
                    if t - round_t0 > G.ROUND_MAX:
                        end_round("time: most bricks left", G.tiebreak(castles, out))
            if step % 10 == 0:
                for k in K3:
                    z = float(live.xpos[base[k]][2]); upz = float(live.xmat[base[k]].reshape(3, 3)[2, 2]); isf = (z < 0.04 or upz < 0.5) and t > 1.0
                    if isf and not fallen[k]: falls[k] += 1; ev("fall", team=G.TEAM[k])
                    fallen[k] = isf
                if pending is not None and pending[0] == "pause" and t - pending[1] > 3.0:
                    if winner is not None: end_at = t + 0.5; pending = ("over", t)
                    else:                              # referee reset: bricks back, crabs re-placed, new openers, serve
                        ROUND = G.roll_round(ORNG, OPEN_CFG); castles.reset(); cracked.clear()
                        for k in K3:
                            for i in range(castles.total(k)): set_brick(k, i, True)
                            place_robot(k, *G.jittered_start(k, ROUND[k]["jit"])); brains[k].reset(t); brains[k].set_opening(ROUND[k], t)
                        serve(0.0); model, live = sim.env_mjdata(0); live.qvel[bda:bda + 6] = 0; mujoco.mj_forward(model, live); sync()
                        round_no += 1; out = {k: None for k in K3}; served = False; banner = None; stall = 0.0
                        OPEN_LOG.append(dict(round=round_no, **{G.TEAM[k]: ROUND[k]["opener"] for k in K3}))
                        ref_line = f"round {round_no}: GO!"; ev("restart", round=round_no); pending = None; settle_until = t + 1.0; round_t0 = t + 1.0
            step += 1
            if step % 2000 == 0:
                print(f"[run] t={t:.0f}s wall={time.time() - t_wall:.0f}s score {score} round {round_no} bricks {[castles.left(k) for k in K3]} "
                      f"states {[brains[k].state for k in K3]} falls={falls}", flush=True)
    finally:
        res = dict(kind="warlords", label="SIM-ONLY VISION CONCEPT", seed=SEED, openings=dict(mode=OPEN_CFG["mode"], seed=OPEN_SEED, rounds=OPEN_LOG),
                   tmax=TMAX, rounds_to_win=ROUNDS_TO_WIN, aggression={G.TEAM[k]: v for k, v in AGGR.items()}, t_end=round(step * dt, 2), stopped=stopped,
                   score={G.TEAM[k]: v for k, v in score.items()}, rounds=rounds, winner=G.TEAM.get(winner) if final else None, final=final,
                   falls={G.TEAM[k]: v for k, v in falls.items()}, bricks_broken=broken_n, serves=serves, round_max=G.ROUND_MAX,
                   max_spin_only_s={G.TEAM[k]: round(v, 1) for k, v in max_rot.items()}, max_standing_s={G.TEAM[k]: round(v, 1) for k, v in max_still.items()},
                   brain_counts={G.TEAM[k]: dict(b.counts) for k, b in brains.items()}, wall_s=round(time.time() - t_wall, 1))
        json.dump(res, open(OUTD / "result.json", "w"), indent=1)
        with open(OUTD / "events.jsonl", "w") as f:
            for e in events: f.write(json.dumps(e) + "\n")
        print("[result] " + json.dumps(res), flush=True)


play._loop = _loop
sys.argv[0] = str(REPO / "scripts" / "play.py")
if __name__ == "__main__":
    play.main()
