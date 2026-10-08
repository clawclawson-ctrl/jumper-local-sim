"""Deliberate push test: start the crab facing an obstacle and walk straight into it; log how the obstacle moves.
    python -m jhs.push_test --app APP --target planterS --start 0,-0.80,-90 [--ly -0.4] [--secs 12] [--fixed]
                            [--mass '{"planterS": 2.0}'] [--out result.json]
Sim test only (sim truth is read here to measure the push). SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import argparse, importlib.util, json, math, os, sys, tempfile, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jhs import TOOLKIT, ROOT  # noqa: E402
from jhs.runner import gl_env  # noqa: E402
from jhs import pushable  # noqa: E402
for _k, _v in gl_env().items(): os.environ.setdefault(_k, _v)

ap = argparse.ArgumentParser()
ap.add_argument("--app", default=str(ROOT / "apps" / "jumper_hide_seek_vision.app")); ap.add_argument("--target", default="planterS")
ap.add_argument("--start", default="0,-0.80,-90"); ap.add_argument("--ly", type=float, default=-0.4); ap.add_argument("--secs", type=float, default=12.0)
ap.add_argument("--settle", type=float, default=2.0); ap.add_argument("--fixed", action="store_true"); ap.add_argument("--mass", default="{}")
ap.add_argument("--rest", type=float, default=3.0, help="seconds standing still after the push (checks the obstacle settles)")
ap.add_argument("--out"); ap.add_argument("--physics-hz", type=int, default=1000)
A = ap.parse_args()
SX, SY, SYAW = (float(v) for v in A.start.split(","))
mapf = Path(tempfile.mkdtemp()) / "push_test.map"
pushable.variant(ROOT / "maps" / "jumper-hide-seek-v7.map", mapf, pushable=not A.fixed, masses=json.loads(A.mass))

REPO = Path(os.environ.get("JUMPER_REPO", str(TOOLKIT)))
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("play", REPO / "scripts" / "play.py")
play = importlib.util.module_from_spec(spec); sys.modules["play"] = play; spec.loader.exec_module(play)
import tasks  # noqa: E402
_orig = tasks.load_env_cfg
def _pinned(*a, **k):
    cfg = _orig(*a, **k); ev = cfg.events.get("reset_base")
    if ev is not None and "pose_range" in ev.params:
        ev.params["pose_range"] = {"x": (SX, SX), "y": (SY, SY), "z": (0.01, 0.01), "yaw": (math.radians(SYAW),) * 2}
    return cfg
tasks.load_env_cfg = _pinned
import controller  # noqa: E402
class Pad:
    connected = True; name = "push test"; path = "virtual"
    def __init__(self): self.s = controller.GamepadState()
    def state(self): return self.s
    def close(self): pass
PAD = Pad(); controller.open = lambda *a, **k: PAD


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
    import mujoco, torch
    un = getattr(env, "unwrapped", env); sim = un.sim; dt = un.step_dt
    model, live = sim.env_mjdata(0)
    gids = pushable.prop_geoms(model); gmap = {k: g for g, k in gids}
    g = gmap[A.target]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
    base = names.index("robot/base_link") if "robot/base_link" in names else 1
    robot_mass = float(sum(model.body_mass[b] for b in range(model.nbody) if names[b].startswith("robot/")))
    tb = model.geom_bodyid[g]
    obs = env.get_observations(); t = 0.0; step = 0; log = []; falls = 0; maxtilt = 0.0; maxz = 0.0; vmax_rest = 0.0
    p_all0 = {k: live.geom_xpos[gg][:2].copy() for k, gg in gmap.items()}
    T = A.settle + A.secs + A.rest
    t_w = time.time()
    while t < T:
        if A.settle <= t < A.settle + A.secs: PAD.s = controller.GamepadState(lx=0.0, ly=A.ly, rx=0.0, ry=0.0, lt=0.0, rt=0.0, hat_x=0, hat_y=0, buttons=frozenset())
        else: PAD.s = controller.GamepadState()
        with torch.inference_mode(): obs, _, _, _ = env.step(policy(obs))
        model, live = sim.env_mjdata(0); t += dt; step += 1
        R = live.geom_xmat[g].reshape(3, 3); tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
        maxtilt = max(maxtilt, tilt); maxz = max(maxz, float(live.geom_xpos[g][2]))
        if t > A.settle + A.secs + 1.0: vmax_rest = max(vmax_rest, float(np.abs(live.cvel[tb][3:]).max()))
        if step % 10 == 0:
            up = live.xmat[base].reshape(3, 3)[:, 2]
            if live.xpos[base][2] < 0.03 or up[2] < 0.5: falls += 1
            yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
            log.append(dict(t=round(t, 2), xy=[round(float(v), 4) for v in live.geom_xpos[g][:2]], yaw=round(yaw, 2), z=round(float(live.geom_xpos[g][2]), 4),
                            robot=[round(float(v), 3) for v in live.xpos[base][:2]], tilt=round(tilt, 2)))
    p0 = np.array(log[int(A.settle / dt / 10) - 1]["xy"]); pe = np.array(log[int((A.settle + A.secs) / dt / 10) - 1]["xy"]); pend = np.array(log[-1]["xy"])
    disp = float(np.linalg.norm(pe - p0)); yaw0 = log[int(A.settle / dt / 10) - 1]["yaw"]
    # contact phase: from first motion > 5 mm to end of push
    first = next((r["t"] for r in log if np.linalg.norm(np.array(r["xy"]) - p0) > 0.005), None)
    speed = disp / max(1e-6, (A.settle + A.secs - first)) if first else 0.0
    others = {k: round(float(np.linalg.norm(live.geom_xpos[gg][:2] - p_all0[k])), 4) for k, gg in gmap.items() if k != A.target}
    res = dict(target=A.target, pushable=not A.fixed, mass=None if A.fixed else {**pushable.MASS, **json.loads(A.mass)}[A.target], friction=pushable.FRICTION,
               robot_mass_kg=round(robot_mass, 3), ly=A.ly, push_s=A.secs, moved_m=round(disp, 4), contact_t=first,
               speed_cm_s=round(100 * speed, 2), yaw_change_deg=round(log[int((A.settle + A.secs) / dt / 10) - 1]["yaw"] - yaw0, 2),
               drift_after_push_mm=round(1000 * float(np.linalg.norm(pend - pe)), 2), max_tilt_deg=round(maxtilt, 2), max_z=round(maxz, 4),
               max_speed_at_rest=round(vmax_rest, 5), robot_falls=falls, other_obstacles_moved_m=others, wall_s=round(time.time() - t_w, 1))
    print("[push_test] " + json.dumps(res), flush=True)
    if A.out: Path(A.out).write_text(json.dumps(dict(res, log=log), indent=1))


play._loop = _loop
sys.argv = [str(REPO / "scripts" / "play.py"), "--app", A.app, "--scene", str(mapf), "--backend", "native", "--device", "cpu", "--headless",
            "--physics-hz", str(A.physics_hz)]
play.main()
