"""1v1 Crab Surround in the local sim. SIM-ONLY VISION CONCEPT.
    python jhs/surround/run.py --app jumper_surround_vision.app --scene RUN/scene.map --backend native --device cpu --headless --steps 999999999
Env: SR_SETUP (json {openings, camera}), VS_SEED, VS_TMAX, VS_OUT, VS_SAVEVIS, SR_LIVE=1, SR_ROUNDS (rounds to win, 1-10), JUMPER_REPO.
Each crab = the official walking app driven by its own Surround Brain through a virtual gamepad, from ITS OWN camera + dToF + pose.
The glowing trails are a sim add-on drawn into every camera (so the crabs SEE them). The REFEREE (trail laying, crashes, score,
walk-back resets) is the only code that uses sim truth."""
from __future__ import annotations
import os
os.environ["TK_ARENA"] = "open"
import importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from jhs.soccer import world as WD, vision  # noqa: E402
from jhs.surround import game as G, brain as SB, fx as FXD  # noqa: E402

REPO = Path(os.environ.get("JUMPER_REPO", str(HERE.parent.parent / "toolkit")))
SEED = int(os.environ.get("VS_SEED", "1")); TMAX = float(os.environ.get("VS_TMAX", "300"))
OUTD = Path(os.environ.get("VS_OUT", "surround_run")); OUTD.mkdir(parents=True, exist_ok=True)
SAVEVIS = os.environ.get("VS_SAVEVIS", "1") == "1"
SETUP = json.load(open(os.environ["SR_SETUP"])) if os.environ.get("SR_SETUP") else {}
ROUNDS_TO_WIN = max(1, min(10, int(os.environ.get("SR_ROUNDS", "3"))))
AGGR = G.aggression_from_env(os.environ)        # per crab, %
OPEN_CFG = dict(SETUP.get("openings") or {}); OPEN_CFG.setdefault("mode", "random")
OPEN_SEED = int(OPEN_CFG["seed"]) if OPEN_CFG.get("seed") not in (None, "") else int.from_bytes(os.urandom(4), "little")
ORNG = np.random.default_rng(OPEN_SEED)
ROUND = G.roll_round(ORNG, OPEN_CFG)
START = {k: list(G.jittered_start(k, ROUND[k]["jit"])) for k in ("A", "B")}
VW, VH, VIS_DT = 640, 480, 0.1
RETURN_TIMEOUT, RET_POS, RET_YAW = 25.0, 0.15, math.radians(15)

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
    """the round winner celebrates with the official app's own moves: crab dance; the match winner bows first."""
    def __init__(self, t, mode_of, win):
        self.mode_of = mode_of; self.t0 = t; self.done = False
        steps = [("wait", 0.3)]
        if win: steps += [("press", "dpad_down", 0.2), ("until", "gesture_bow", 3.0), ("until", "locomotion", 12.0), ("wait", 0.4)]
        steps += [("press", "menu+dpad_up", 0.2), ("until", "dance_crab", 3.0), ("wait", 5.0 if win else 2.5),
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


PADS = {"A": VirtualPad("virtual pad RED (surround brain, sim-only)"), "B": VirtualPad("virtual pad BLUE (surround brain, sim-only)")}
controller.open = lambda *a, **k: PADS["A"]


def yaw_of(q): return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
    global ROUND
    import mujoco, imageio
    from scipy import ndimage
    un = env.unwrapped; sim = un.sim
    tofs = {"A": un.scene["tof"], "B": un.scene["tof2"]}
    rb = un.scene["robot2"]; rs = rb.data.default_root_state.clone(); rs[:, :3] += un.scene.env_origins
    rb.write_root_state_to_sim(rs); rb.write_joint_state_to_sim(rb.data.default_joint_pos.clone(), torch.zeros_like(rb.data.default_joint_vel))
    pB = ap.AppPlayer(ap.open_app(Path(sys.argv[sys.argv.index("--app") + 1])), WD.EnvView(env, rb), pad=PADS["B"])
    pA = policy
    mgr = un.action_manager; names_ = list(mgr.active_terms)
    def both(): pA._apply(); pB._apply()
    mgr.get_term(names_[0]).apply_actions = both
    model, live = sim.env_mjdata(0)
    WD.skin(model); FXD.theme(model)
    rmodel = getattr(sim, "render_model", None) or model
    if rmodel is not model: WD.skin(rmodel); FXD.theme(rmodel)
    model.vis.global_.offwidth = max(VW, model.vis.global_.offwidth); model.vis.global_.offheight = max(VH, model.vis.global_.offheight)
    rend = mujoco.Renderer(model, VH, VW, max_geom=12000)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
    def _root(b):
        while model.body_parentid[b] != 0: b = model.body_parentid[b]
        return b
    groot = [names[_root(model.geom_bodyid[g])] for g in range(model.ngeom)]
    PFX = {"A": "robot/", "B": "robot2/"}
    own_geom = {k: np.array([n.startswith(p) for n in groot] + [False]) for k, p in PFX.items()}
    cams = {k: (model.camera(PFX[k] + "onboard").id, model.camera(PFX[k] + "tof").id) for k in PFX}
    cid, tid = cams["A"]
    cam = vision.Camera(VW, VH, float(model.cam_fovy[cid]))
    tw, th = (int(v) for v in model.cam_resolution[tid])
    base = {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PFX[k] + "base_link") for k in PFX}
    DUE = {"due": True}
    for s in tofs.values():
        _k = s.raycast_kernel
        def _when(*a, _k=_k, _s=s, **kw):
            if DUE["due"]: _s._force[:] = True; return _k(*a, **kw)
        s.raycast_kernel = _when

    def place_robot(k, x, y, deg):
        model, live = sim.env_mjdata(0); jn = model.body_jntadr[base[k]]
        qa, da = model.jnt_qposadr[jn], model.jnt_dofadr[jn]; h = math.radians(deg) / 2
        live.qpos[qa:qa + 2] = [x, y]; live.qpos[qa + 2] = max(float(live.qpos[qa + 2]), 0.11)
        live.qpos[qa + 3:qa + 7] = [math.cos(h), 0.0, 0.0, math.sin(h)]; live.qvel[da:da + 6] = 0
        mujoco.mj_forward(model, live)
        if hasattr(sim, "_gather"): sim._gather()

    def park_props():      # the map's football + official props wait outside the arena (no obstacles in Surround)
        model, live = sim.env_mjdata(0)
        for i_, b in enumerate(b for b in range(model.nbody) if names[b].startswith("prop:")):
            j = model.body_jntadr[b]
            if j < 0 or model.jnt_type[j] != 0: continue
            qa, da = model.jnt_qposadr[j], model.jnt_dofadr[j]
            live.qpos[qa:qa + 3] = [4.8 + 0.8 * (i_ % 2), -2.5 + 0.9 * (i_ // 2), 0.3]; live.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]; live.qvel[da:da + 6] = 0
        mujoco.mj_forward(model, live)
        if hasattr(sim, "_gather"): sim._gather()

    brains = {k: SB.Brain(k, np.random.default_rng(SEED * 31 + i), aggression=AGGR[k] / 100) for i, k in enumerate(("A", "B"))}
    for k in ("A", "B"): brains[k].set_opening(ROUND[k], 0.0)
    OPEN_LOG = [dict(round=1, **{G.TEAM[k]: ROUND[k]["opener"] for k in ("A", "B")}, start={G.TEAM[k]: START[k] for k in ("A", "B")})]
    print("[openings] seed=%d mode=%s round 1: RED %s, BLUE %s" % (OPEN_SEED, OPEN_CFG["mode"], ROUND["A"]["opener"], ROUND["B"]["opener"]), flush=True)
    trails = {"A": G.Trail(), "B": G.Trail()}
    HOOK = None
    if os.environ.get("SR_LIVE", "0") == "1" or os.environ.get("JHS_LIVE_DIR"):
        from jhs.surround.live import SurroundHook
        HOOK = SurroundHook(sim=sim, outd=OUTD, dt=un.step_dt)
        HOOK.draw_cb = lambda mj, scn: FXD.draw(mj, scn, {k: tr.pts for k, tr in trails.items()})
        HOOK.trails = trails
    print(f"[setup] surround seed={SEED} tmax={TMAX} A={START['A']} B={START['B']} rounds_to_win={ROUNDS_TO_WIN} aggression RED {AGGR["A"]:.0f}% BLUE {AGGR["B"]:.0f}% cam {VW}x{VH} tof {tw}x{th}", flush=True)
    dt = float(un.step_dt); vis_every = max(1, round(VIS_DT / dt)); q_every = max(1, round(0.04 / dt))
    obs = env.get_observations(); step = 0
    score = {"A": 0, "B": 0}; events = []; falls = {"A": 0, "B": 0}; fallen = {"A": False, "B": False}
    cele = None; final = None; winner = None; end_at = None
    modes = {"A": lambda: pA.fsm.mode, "B": lambda: pB.fsm.mode}
    settle_until = 1.5; paused_s = 0.0; prev_t = 0.0; returns = []; rounds = []
    pending = None; ref_line = "GO!"; banner = None; round_no = 1; round_t0 = 1.5; out = {"A": None, "B": None}
    qbuf, qt, vis_meta, tofA, tofB, fxbuf, trail_rounds = [], [], [], [], [], [], []
    truth = open(OUTD / "truth.jsonl", "w"); t_wall = time.time(); stopped = False; t = 0.0; mt = 0.0
    cmds = {"A": {}, "B": {}}; poses = {k: (START[k][0], START[k][1], math.radians(START[k][2])) for k in ("A", "B")}
    wall_hits = {"A": 0, "B": 0}; wall_prev = {"A": False, "B": False}
    rot_only = {"A": 0.0, "B": 0.0}; max_rot = {"A": 0.0, "B": 0.0}; still = {"A": 0.0, "B": 0.0}; max_still = {"A": 0.0, "B": 0.0}
    def ev(kind, **kw):
        e = dict(t=round(t, 2), event=kind, **kw); events.append(e); print(f"[ref] {e}", flush=True)
    def truth_pose(k):
        _m, lv = sim.env_mjdata(0); return (float(lv.xpos[base[k]][0]), float(lv.xpos[base[k]][1]), yaw_of(lv.xquat[base[k]]))
    def end_round(t, how, rw):
        nonlocal pending, cele, winner, final, ref_line, banner, end_at
        rounds.append(dict(round=round_no, t=round(t - round_t0, 1), how=how, winner=G.TEAM.get(rw), out={G.TEAM[k]: v for k, v in out.items() if v},
                           trail_m={G.TEAM[k]: round(tr.length(), 2) for k, tr in trails.items()}, openers={G.TEAM[k]: ROUND[k]["opener"] for k in ("A", "B")}))
        if rw is not None: score[rw] += 1
        ref_line = (f"round {round_no}: {G.TEAM[rw]} survives ({how})" if rw else f"round {round_no}: DRAW ({how})") + f"  {score['A']}-{score['B']}"
        banner = f"{G.TEAM[rw]} TAKES ROUND {round_no}" if rw else f"ROUND {round_no} DRAW"
        ev("round_end", round=round_no, how=how, winner=G.TEAM.get(rw), score=dict(score))
        win = rw is not None and score[rw] >= ROUNDS_TO_WIN
        if win: winner = rw; final = f"{G.TEAM[rw]} WINS {score['A']}-{score['B']}"; ref_line = final; ev("match_won", final=final)
        if rw is not None: cele = Celebration(t, modes[rw], win); pending = ("celebrate", t, rw)
        else: cele = None; pending = ("celebrate", t, None)
    try:
        while True:
            t = step * dt
            if end_at is not None and t >= end_at: break
            if pending is not None and pending[0] in ("celebrate", "return"): paused_s += t - prev_t
            prev_t = t; mt = t - paused_s
            if mt >= TMAX and final is None:
                winner = "A" if score["A"] > score["B"] else "B" if score["B"] > score["A"] else None
                final = (f"{G.TEAM[winner]} WINS {score['A']}-{score['B']}" if winner else f"DRAW {score['A']}-{score['B']}") + " (time)"
                ref_line = f"time: {final}"; ev("full_time", final=final); end_at = t + 3.0; pending = ("over", 1e9)
            if step == 2:
                for k in ("A", "B"): place_robot(k, *START[k])
                park_props()
            model, live = sim.env_mjdata(0)
            if step % vis_every == 0 and t > 0.3:
                metas = {}
                for k in ("A", "B"):
                    c_id, t_id = cams[k]
                    rend.update_scene(live, camera=c_id); FXD.draw(mujoco, rend.scene, {kk: tr.pts for kk, tr in trails.items()}); img = rend.render().copy()
                    rend.enable_segmentation_rendering(); rend.update_scene(live, camera=c_id); seg = rend.render()[..., 0].copy(); rend.disable_segmentation_rendering()
                    selfm = ndimage.binary_dilation(own_geom[k][np.clip(seg, 0, own_geom[k].size - 1)] & (seg >= 0), iterations=3)
                    rngs = tofs[k].data.range[0].detach().cpu().numpy().reshape(th, tw)
                    cpos = live.cam_xpos[c_id].copy(); cR = live.cam_xmat[c_id].reshape(3, 3).copy()
                    p = SB.perceive(k, img, selfm, cpos, cR, cam)
                    pose = truth_pose(k); br = brains[k]
                    br.observe(t, p, pose); poses[k] = pose
                    if pending is not None and pending[0] == "return": cmds[k] = br.return_cmd(pose)
                    elif pending is None and out[k] is None:
                        try: cmds[k] = br.decide(pose)
                        except Exception as e:
                            cmds[k] = {}; ev("brain_error", team=G.TEAM[k], error=repr(e)[:200])
                    elif pending is not None and pending[0] == "celebrate" and k == pending[2]: cmds[k] = cele.cmd(t)
                    else: cmds[k] = {}
                    if t < settle_until: cmds[k] = {}
                    PADS[k].set(cmds[k])
                    c = cmds[k]; playing = pending is None and out[k] is None and t >= settle_until
                    if playing and abs(c.get("rx", 0)) > 0.1 and abs(c.get("lx", 0)) + abs(c.get("ly", 0)) < 0.05: rot_only[k] += VIS_DT
                    else: rot_only[k] = 0.0
                    max_rot[k] = max(max_rot[k], rot_only[k])
                    if playing and abs(c.get("lx", 0)) + abs(c.get("ly", 0)) < 0.05: still[k] += VIS_DT
                    else: still[k] = 0.0
                    max_still[k] = max(max_still[k], still[k])
                    _cel = pending is not None and pending[0] == "celebrate" and k == pending[2]
                    if pending is None: st_ = "out" if out[k] else br.state
                    elif pending[0] == "return": st_ = "return"
                    else: st_ = "celebrate" if _cel else "over" if pending[0] == "over" else "paused"
                    sub = br.sub if pending is None or pending[0] == "return" else (cele.label if _cel else "standing still")
                    if pending is None and out[k]: sub = f"OUT: {out[k]}"
                    metas[k] = dict(state=st_, sub=sub, pose=[round(v, 3) for v in pose],
                                    cmd={a_: (round(float(b_), 2) if a_ != "press" else b_) for a_, b_ in cmds[k].items()},
                                    opp=[round(float(v), 3) for v in br.opp] if br.opp is not None and t - br.opp_t < 3.0 else None,
                                    opp_age=round(min(t - br.opp_t, 99), 1),
                                    dets=[dict(cls="opponent", bbox=d["bbox"], rng=round(d["rng"], 2)) for d in p["opp"]],
                                    ntrail=int(len(p["trail"])), opener=br.op_name, opener_on=br.opening is not None,
                                    belief=br.bel.coarse(4).tolist())
                    if SAVEVIS:
                        (OUTD / "vis").mkdir(exist_ok=True)
                        imageio.imwrite(OUTD / "vis" / f"{k}_{len(vis_meta):05d}.jpg", img[::2, ::2], quality=85)
                        (tofA if k == "A" else tofB).append(rngs.astype(np.float16))
                    if HOOK is not None: HOOK.vision(k, img, rngs)
                meta = dict(t=round(t, 2), clock=round(mt, 2), A=metas["A"], B=metas["B"], score=dict(score), ref=ref_line, final=final, banner=banner,
                            rounds_to_win=ROUNDS_TO_WIN, aggression=dict(AGGR), round=round_no, round_t=round(max(0.0, t - round_t0), 1) if pending is None else 0.0)
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
            if step % 10 == 0:
                P = {k: truth_pose(k) for k in ("A", "B")}
                for k in ("A", "B"):                        # stat: wall contacts (walls are harmless)
                    wc_ = G.wall_contact(P[k]) and pending is None
                    if wc_ and not wall_prev[k]: wall_hits[k] += 1
                    wall_prev[k] = wc_
                if pending is None and t >= settle_until:
                    for k in ("A", "B"):
                        if out[k] is None: trails[k].lay(P[k])
                    hit = {}
                    for k in ("A", "B"):
                        if out[k] is None:
                            o = "B" if k == "A" else "A"
                            r = G.check_crash(k, P[k], trails, P[o] if out[o] is None else None)
                            if r: hit[k] = r
                    for k, r in hit.items(): out[k] = r; ev("crash", team=G.TEAM[k], how=r, at=[round(P[k][0], 2), round(P[k][1], 2)])
                    if hit:
                        alive = [k for k in ("A", "B") if out[k] is None]
                        end_round(t, "; ".join(f"{G.TEAM[k]} hit {v}" for k, v in hit.items()), alive[0] if len(alive) == 1 else None)
                    elif t - round_t0 > G.ROUND_MAX:
                        tw, terr = G.tiebreak(P, trails)
                        how = "round time limit: " + (f"territory {terr['A']}-{terr['B']} cells" if terr else "draw")
                        end_round(t, how, tw)
                for k in ("A", "B"):
                    z = float(live.xpos[base[k]][2]); upz = float(live.xmat[base[k]].reshape(3, 3)[2, 2])
                    isf = (z < 0.04 or upz < 0.5) and t > 1.0
                    if isf and not fallen[k]: falls[k] += 1; ev("fall", team=G.TEAM[k], z=round(z, 3))
                    fallen[k] = isf
                if pending is not None and pending[0] == "celebrate":
                    done = (cele.done or t - cele.t0 > 30.0) if cele is not None else t - pending[1] > 2.5
                    if done:
                        if winner is not None: end_at = t + 0.5; pending = ("over", 1e9)
                        else:
                            ROUND = G.roll_round(ORNG, OPEN_CFG)
                            spots = {k: G.jittered_start(k, ROUND[k]["jit"]) for k in ("A", "B")}
                            trail_rounds.append({k: [list(map(lambda v: round(v, 3), p)) for p in tr.pts] for k, tr in trails.items()})
                            for tr in trails.values(): tr.clear()
                            ref_line = "referee: trails cleared, both crabs walk back to their start spots"; banner = None; ev("walk_back")
                            pending = ("return", t, spots)
                            for k in ("A", "B"): brains[k].begin_return(spots[k], t)
                elif pending is not None and pending[0] == "return":
                    spots = pending[2]; ok = {}
                    for k in ("A", "B"):
                        sx, sy, sh_ = spots[k]; x, y, yaw = P[k]
                        ok[k] = math.hypot(x - sx, y - sy) < RET_POS and abs(SB.wrap(math.radians(sh_) - yaw)) < RET_YAW
                    late = [k for k in ok if not ok[k]] if t - pending[1] > RETURN_TIMEOUT else None
                    if all(ok.values()) or late:
                        for k in (late or []): place_robot(k, *spots[k])
                        returns.append(dict(t=round(t - pending[1], 1), timeout=[G.TEAM[k] for k in (late or [])]))
                        round_no += 1; out = {"A": None, "B": None}
                        for k, b in brains.items(): b.reset(t); b.set_opening(ROUND[k], t)
                        OPEN_LOG.append(dict(round=round_no, **{G.TEAM[k]: ROUND[k]["opener"] for k in ("A", "B")}, start={G.TEAM[k]: list(spots[k]) for k in ("A", "B")}))
                        print("[openings] round %d: RED %s, BLUE %s" % (round_no, ROUND["A"]["opener"], ROUND["B"]["opener"]), flush=True)
                        ref_line = f"round {round_no}: GO! ({score['A']}-{score['B']})"; ev("restart", round=round_no, late=[G.TEAM[k] for k in (late or [])])
                        pending = None; settle_until = t + 1.0; round_t0 = t + 1.0
                if step % 20 == 0:
                    truth.write(json.dumps({"t": round(t, 2), **{k: [round(v, 3) for v in P[k]] for k in P}, "trail_n": {k: len(tr.pts) for k, tr in trails.items()}}) + "\n")
            if step % q_every == 0:
                qbuf.append(np.array(live.qpos, dtype=np.float64)); qt.append(t)
                fxbuf.append(dict(r=len(trail_rounds), n={k: len(tr.pts) for k, tr in trails.items()}))
            step += 1
            if step % 2000 == 0:
                print(f"[run] t={t:.0f}s wall={time.time() - t_wall:.0f}s score {score['A']}-{score['B']} round {round_no} A={brains['A'].state} B={brains['B'].state} falls={falls}", flush=True)
    finally:
        truth.close()
        trail_rounds.append({k: [[round(v, 3) for v in p] for p in tr.pts] for k, tr in trails.items()})
        res = dict(kind="surround", label="SIM-ONLY VISION CONCEPT", seed=SEED, openings=dict(mode=OPEN_CFG["mode"], seed=OPEN_SEED, rounds=OPEN_LOG),
                   tmax=TMAX, rounds_to_win=ROUNDS_TO_WIN, aggression={G.TEAM[k]: v for k, v in AGGR.items()}, t_end=round(step * dt, 2), stopped=stopped, setup=dict(A=START["A"], B=START["B"]),
                   score={"RED": score["A"], "BLUE": score["B"]}, rounds=rounds, returns=returns, match_clock_s=round(mt, 1),
                   max_spin_only_s={G.TEAM[k]: round(v, 1) for k, v in max_rot.items()}, max_standing_s={G.TEAM[k]: round(v, 1) for k, v in max_still.items()},
                   winner=G.TEAM.get(winner) if final else None, final=final, falls={"RED": falls["A"], "BLUE": falls["B"]},
                   wall_contacts={G.TEAM[k]: v for k, v in wall_hits.items()}, round_max=G.ROUND_MAX,
                   brain_counts={G.TEAM[k]: dict(b.counts) for k, b in brains.items()}, wall_s=round(time.time() - t_wall, 1))
        json.dump(res, open(OUTD / "result.json", "w"), indent=1)
        with open(OUTD / "events.jsonl", "w") as f:
            for e in events: f.write(json.dumps(e) + "\n")
        json.dump(trail_rounds, open(OUTD / "trails.json", "w"))
        if qbuf:
            np.savez_compressed(OUTD / "qpos.npz", qpos=np.stack(qbuf), t=np.array(qt), fps=25.0)
            json.dump(fxbuf, open(OUTD / "surround_fx.json", "w"))
        if SAVEVIS and vis_meta:
            n = min(len(tofA), len(tofB), len(vis_meta))
            np.savez_compressed(OUTD / "vision_tof.npz", tofA=np.stack(tofA[:n]), tofB=np.stack(tofB[:n]))
            json.dump(vis_meta[:n], open(OUTD / "vision_meta.json", "w"))
        print("[result] " + json.dumps(res), flush=True)


play._loop = _loop
sys.argv[0] = str(REPO / "scripts" / "play.py")
if __name__ == "__main__":
    play.main()
