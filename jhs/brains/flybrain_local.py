"""flybrain in the local sim. SIM-ONLY VISION CONCEPT: fly-inspired rules, not a connectome.

Runs flybrain.app (the official Jumper walking policy, mode `flybrain`) in the room, and drives its gamepad with the
hand-written fly-inspired rules in sim_only_flybrain.py (`Fly`, used unchanged). This file is only the local runner
glue, the same hooks as the hide & seek brain:
  JUMPER_REPO   official toolkit (default: the local sim's toolkit/)
  VS_START      "x,y,yaw_deg" crab start (default rug centre, heading from the seed)
  VS_PLACE      JSON file {toy: [x, y]} where the person put the toys (others parked outside the room)
  VS_LIVE_HOOK  live view / pacing / Stop (display only; never feeds the rules)
  VS_SEED, VS_TMAX (sim s), VS_OUT (run folder), VS_SAVEVIS (1 = record camera/dToF/meta for the MP4), VS_THREADS
The --scene map is the per-run copy written by the local runner (pushable / moved props).
Sim truth (toy positions, props) is used only to set the scene up and for the log; the rules see the camera + dToF.
"""
from __future__ import annotations
import importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sim_only_flybrain as FB  # noqa: E402  (the rules; unchanged copy of flybrain/sim_only_flybrain.py)

REPO = Path(os.environ.get("JUMPER_REPO", str(HERE.parent.parent / "toolkit")))
SEED = int(os.environ.get("VS_SEED", "1")); TMAX = float(os.environ.get("VS_TMAX", "120"))
OUTD = Path(os.environ.get("VS_OUT", "flybrain_run")); OUTD.mkdir(parents=True, exist_ok=True)
SAVEVIS = os.environ.get("VS_SAVEVIS", "1") == "1"
VW, VH, VIS_DT = FB.VW, FB.VH, FB.VIS_DT
TOYBODY = {"ball": "prop:ball", "duck": "prop:toy_duck", "duck2": "prop:toy_duck_2", "minifb": "prop:mini_football", "football": "prop:football"}
PROPB = {"stairE": "stair_8", "stairW": "stair_9", "planterN": "planterN_10", "planterS": "planterW_11", "crateNE": "prop:crate_blue",
         "crateSE": "prop:crate_medium", "crateSW": "prop:crate_small", "crateNW": "prop:crate_medium_nw"}
rng = np.random.default_rng(SEED)
START_X = START_Y = 0.0; START_YAW = float(rng.uniform(-math.pi, math.pi))
if os.environ.get("VS_START"):
    START_X, START_Y, _yd = (float(v) for v in os.environ["VS_START"].split(",")); START_YAW = math.radians(_yd)

sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("play", REPO / "scripts" / "play.py")
play = importlib.util.module_from_spec(_spec); sys.modules["play"] = play; _spec.loader.exec_module(play)
import controller  # noqa: E402
import tasks  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("VS_THREADS", "2"))); torch.manual_seed(SEED)
_orig = tasks.load_env_cfg
def _pinned(*a, **k):
    cfg = _orig(*a, **k); ev = cfg.events.get("reset_base") if getattr(cfg, "events", None) else None
    if ev is not None and "pose_range" in ev.params:
        ev.params["pose_range"] = {"x": (START_X, START_X), "y": (START_Y, START_Y), "z": (0.01, 0.01), "yaw": (START_YAW, START_YAW)}
    cfg.seed = SEED
    return cfg
tasks.load_env_cfg = _pinned


class VirtualPad:
    connected = True; name = "virtual pad (flybrain, sim-only)"; path = "virtual"
    def __init__(self): self.s = controller.GamepadState()
    def state(self): return self.s
    def close(self): pass
PAD = VirtualPad(); controller.open = lambda *a, **k: PAD


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
    import mujoco, imageio
    un = getattr(env, "unwrapped", env); sim = un.sim
    model, live = sim.env_mjdata(0)
    model.vis.global_.offwidth = max(VW, model.vis.global_.offwidth); model.vis.global_.offheight = max(VH, model.vis.global_.offheight)
    rend = mujoco.Renderer(model, VH, VW)
    cid = model.camera("robot/onboard").id; tid = model.camera("robot/tof").id
    fovy = float(model.cam_fovy[cid]); f_px = (VH / 2) / math.tan(math.radians(fovy) / 2)
    cam_hfov = 2 * math.atan((VW / 2) / f_px)
    tw, th = (int(v) for v in model.cam_resolution[tid])
    tof_w = float(model.cam_sensorsize[tid][0]); tof_f = float(model.cam_intrinsic[tid][0]) if tof_w > 0 else 0.0
    if tof_w > 0 and tof_f > 0: tof_hfov = 2 * math.atan((tof_w / 2) / tof_f)
    else: tof_hfov = 2 * math.atan(math.tan(math.radians(float(model.cam_fovy[tid])) / 2) * tw / max(th, 1))   # fovy-defined camera
    tof = un.scene["tof"]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
    def body(suffix):
        m = [i for i, n in enumerate(names) if n.endswith(suffix)]; return m[0] if m else None
    base_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
    def _root(b):
        while model.body_parentid[b] != 0: b = model.body_parentid[b]
        return b
    robot_geom = np.array([names[_root(model.geom_bodyid[g])].startswith("robot") for g in range(model.ngeom)])
    # --- scene setup (sim truth, setup only): toys where the person put them, the rest parked outside the room ---
    place = json.load(open(os.environ["VS_PLACE"])) if os.environ.get("VS_PLACE") else {}
    tb = {}
    for i, (k, v) in enumerate(TOYBODY.items()):
        b = body(v)
        if b is None or model.body_jntadr[b] < 0: continue
        qa = model.jnt_qposadr[model.body_jntadr[b]]; da = model.jnt_dofadr[model.body_jntadr[b]]
        if k in place: live.qpos[qa], live.qpos[qa + 1] = float(place[k][0]), float(place[k][1]); tb[k] = b
        else: live.qpos[qa], live.qpos[qa + 1] = 3.5, 3.5 + 0.3 * i
        live.qvel[da:da + 6] = 0
    mujoco.mj_forward(model, live)
    if hasattr(sim, "_gather"): sim._gather()
    pb = {k: body(v) for k, v in PROPB.items() if body(v) is not None}
    print(f"[setup] floor returns dropped below {FB.FLOOR_MARGIN * 100:.0f} cm (pose + dToF ray geometry); ranges horizontal", flush=True)
    print(f"[setup] flybrain seed={SEED} start=({START_X:.2f},{START_Y:.2f}) yaw={math.degrees(START_YAW):.0f} toys={place} "
          f"cam_hfov={math.degrees(cam_hfov):.1f} tof_hfov={math.degrees(tof_hfov):.1f} tof={tw}x{th}", flush=True)
    HOOK = None
    if os.environ.get("VS_LIVE_HOOK"):
        _hs = importlib.util.spec_from_file_location("vs_live_hook", os.environ["VS_LIVE_HOOK"]); _hm = importlib.util.module_from_spec(_hs); _hs.loader.exec_module(_hm)
        HOOK = _hm.Hook(sim=sim, toys={}, rug=(0.0, 0.0, 0.87, 0.64), room=FB.ROOM, outd=OUTD, dt=un.step_dt, kind="fly")
    fly = FB.Fly(rng)
    dt = float(un.step_dt); vis_every = max(1, round(VIS_DT / dt)); q_every = max(1, round(0.04 / dt))
    obs = env.get_observations(); step = 0; falls = 0; fallen = False; f_small = f_px / 4.0
    logf = open(OUTD / "behaviours.jsonl", "w"); truth = open(OUTD / "truth.jsonl", "w")
    qbuf, qt, vis_meta, vis_tof, vis_floor = [], [], [], [], []
    t_wall = time.time(); path_m = 0.0; last_xy = None; stopped = False; t = 0.0
    cmd = {"ly": 0.0, "lx": 0.0, "rx": 0.0, "lb": False}
    try:
        while step * dt < TMAX:
            t = step * dt
            model, live = sim.env_mjdata(0)
            yaw = FB.yaw_of(live.qpos[3:7]); xy = np.array(live.qpos[:2], float)
            if step % vis_every == 0 and t > 0.4:
                rend.update_scene(live, camera="robot/onboard"); img = rend.render().copy()
                rend.enable_segmentation_rendering(); rend.update_scene(live, camera="robot/onboard"); seg = rend.render()[..., 0].copy(); rend.disable_segmentation_rendering()
                selfm = robot_geom[np.clip(seg, 0, robot_geom.size - 1)] & (seg >= 0)
                rngs = tof.data.range[0].detach().cpu().numpy().reshape(th, tw)
                # the crab's own pose (dToF camera = base pose + mount) + ray geometry -> floor returns dropped (fix1)
                tpose = (live.cam_xpos[tid].copy(), live.cam_xmat[tid].reshape(3, 3).copy(), float(model.cam_fovy[tid]))
                cmd = fly.step(t, VIS_DT, img, selfm, rngs, yaw, xy, f_small, cam_hfov, tof_hfov, tof_pose=tpose)
                floor = fly.floor if fly.floor is not None else np.zeros_like(rngs, bool)
                ov = fly.overlay
                from jhs.fly_overlay import display_state
                st = display_state(fly.beh, fly.sub, cmd)
                meta = {"t": round(t, 2), "state": st, "beh": fly.beh, "sub": fly.sub, "cmd": dict(cmd),
                        "flow": ov.get("flow"), "residual": ov.get("residual"), "hemifield": [int(v) for v in ov.get("hemifield", [0, 0])],
                        "near": ov.get("near"), "loom": ov.get("loom"), "blob": ov.get("blob"), "blob_range": ov.get("blob_range"), "floor_px": ov.get("floor_px"),
                        "pose": [round(float(xy[0]), 3), round(float(xy[1]), 3), round(yaw, 3)], "counts": dict(fly.counts)}
                logf.write(json.dumps(meta) + "\n")
                if SAVEVIS:
                    (OUTD / "vis").mkdir(exist_ok=True)
                    imageio.imwrite(OUTD / "vis" / f"cam_{len(vis_meta):05d}.jpg", img, quality=88)
                    vis_tof.append(rngs.astype(np.float16)); vis_floor.append(floor); vis_meta.append(meta)
                if HOOK is not None: HOOK.vision(t, img, rngs, meta, floor)     # 5th arg: floor mask (display only)
            btn = frozenset(["LB"]) if cmd.get("lb") else frozenset()
            PAD.s = controller.GamepadState(lx=float(cmd.get("lx", 0)), ly=float(cmd.get("ly", 0)), rx=float(cmd.get("rx", 0)), ry=0.0, buttons=btn)
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            if HOOK is not None and HOOK.step(t) == "stop":
                stopped = True; print(f"[run] stopped from the local runner at t={t:.1f}s", flush=True); break
            if step % 10 == 0:
                model, live = sim.env_mjdata(0)
                z = float(live.xpos[base_b][2]); upz = float(live.xmat[base_b].reshape(3, 3)[2, 2])
                isf = z < 0.04 or upz < 0.5
                if isf and not fallen: falls += 1; print(f"[fall] t={t:.2f} z={z:.3f} up={upz:.2f}", flush=True)
                fallen = isf
                if last_xy is not None: path_m += float(np.hypot(*(xy - last_xy)))
                last_xy = xy
            if step % 20 == 0:
                truth.write(json.dumps({"t": round(t, 2), "base": [round(float(v), 3) for v in live.qpos[:3]], "upz": round(float(live.xmat[base_b].reshape(3, 3)[2, 2]), 3),
                                        **{k: [round(float(v), 3) for v in live.xpos[b][:3]] for k, b in {**tb, **pb}.items()}}) + "\n")
            if step % q_every == 0: qbuf.append(np.array(live.qpos, dtype=np.float64)); qt.append(t)
            step += 1
            if step % 2000 == 0:
                print(f"[run] t={t:.0f}s wall={time.time() - t_wall:.0f}s beh={fly.beh}/{fly.sub} xy={np.round(xy, 2)} falls={falls} counts={fly.counts}", flush=True)
    finally:
        logf.close(); truth.close()
        model, live = sim.env_mjdata(0)
        T = [json.loads(l) for l in open(OUTD / "truth.jsonl")] if (OUTD / "truth.jsonl").exists() else []
        res = dict(kind="flybrain", label="SIM-ONLY VISION CONCEPT: fly-inspired rules, not a connectome", seed=SEED, tmax=TMAX,
                   start=[START_X, START_Y, round(math.degrees(START_YAW), 1)], placed=place, t_end=round(step * dt, 2), stopped=stopped,
                   falls=falls, path_m=round(path_m, 2), end_xy=[round(float(live.qpos[0]), 3), round(float(live.qpos[1]), 3)],
                   time_s={k: round(v, 2) for k, v in fly.time.items()}, counts=dict(fly.counts),
                   props_moved_m={k: round(float(np.linalg.norm(live.xpos[b][:2] - np.array(T[0][k][:2]))), 3) for k, b in pb.items() if T and k in T[0]},
                   toys_moved_m={k: round(float(np.linalg.norm(live.xpos[b][:2] - np.array(T[0][k][:2]))), 3) for k, b in tb.items() if T and k in T[0]},
                   n_toys=len(tb), wall_s=round(time.time() - t_wall, 1))
        json.dump(res, open(OUTD / "result.json", "w"), indent=1)
        if qbuf: np.savez_compressed(OUTD / "qpos.npz", qpos=np.stack(qbuf), t=np.array(qt), fps=25.0)
        if SAVEVIS and vis_tof:
            np.savez_compressed(OUTD / "vision_tof.npz", tof=np.stack(vis_tof), floor=np.stack(vis_floor)); json.dump(vis_meta, open(OUTD / "vision_meta.json", "w"))
        print("[result] " + json.dumps(res), flush=True)


play._loop = _loop
sys.argv[0] = str(REPO / "scripts" / "play.py")
if __name__ == "__main__":
    play.main()
