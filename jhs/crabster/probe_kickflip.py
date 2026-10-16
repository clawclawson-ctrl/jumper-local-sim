# real-sim kickflip trial: scripted thruster flight controller (jhs.crabster.flight) on the walking crab
import os, sys, importlib.util, math, json
from pathlib import Path
sys.path.insert(0, "/workspace/repo-jumper-local-sim")
from jhs.crabster import flight as F
REPO = Path("/workspace/repo-jumper-local-sim/toolkit"); sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("play", REPO / "scripts" / "play.py"); play = importlib.util.module_from_spec(spec); sys.modules["play"] = play; spec.loader.exec_module(play)
import controller, tasks, numpy as np
class Pad:
    connected = True; path = "v"; name = "v"
    def __init__(self): self.s = controller.GamepadState()
    def state(self): return self.s
    def close(self): pass
PAD = Pad(); controller.open = lambda *a, **k: PAD
orig = tasks.load_env_cfg
def pinned(*a, **k):
    cfg = orig(*a, **k)
    if isinstance(cfg.terminations, dict): cfg.terminations.clear()
    else:
        for n in list(vars(cfg.terminations)): setattr(cfg.terminations, n, None)
    cfg.episode_length_s = 1e6; return cfg
tasks.load_env_cfg = pinned
T0 = float(os.environ.get('T0', 3.0)); KP, KD = float(os.environ.get("KP", 36)), float(os.environ.get("KD", 9)); FLIP_T = float(os.environ.get("FLIP_T", 0.55)); LIFT_T = float(os.environ.get("LIFT_T", 0.30))
def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
    import mujoco, torch
    un = env.unwrapped; sim = un.sim; model, live = sim.env_mjdata(0)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link"); dt = un.step_dt
    rb = [i for i in range(model.nbody) if (mujoco.mj_id2name(model, 1, i) or "").startswith("robot/")]
    model.body_mass[b] += F.DRY_ADD + F.GAS0
    mujoco.mj_forward(model, live)
    m = sum(model.body_mass[i] for i in rb)
    R0 = live.xmat[b].reshape(3, 3); p0 = live.xpos[b]; I = np.zeros((3, 3))
    for i in rb:
        Ri = live.ximat[i].reshape(3, 3); Ib = Ri @ np.diag(model.body_inertia[i]) @ Ri.T; d = live.xipos[i] - p0
        I += Ib + model.body_mass[i] * (d @ d * np.eye(3) - np.outer(d, d))
    Ib = R0.T @ I @ R0; print("mass", round(m, 3), "I diag", np.round(np.diag(Ib), 4), flush=True)
    da = model.jnt_dofadr[model.body_jntadr[b]]; print("arm", np.round(live.subtree_com[b] - live.xipos[b], 4), "xipos-xpos", np.round(live.xipos[b] - live.xpos[b], 4), flush=True)
    MB = np.array([model.body_mass[i] for i in rb])
    fl = F.Flight(inertia=np.diag(Ib), mass=m); tank = F.Tank()
    obs = env.get_observations(); ph = "walk"; t_ph = 0; yaw0 = None
    for s in range(int(9 / dt)):
        t = s * dt; model, live = sim.env_mjdata(0)
        R = live.xmat[b].reshape(3, 3).copy(); w_b = live.qvel[da + 3:da + 6].copy(); v = live.qvel[da:da + 3].copy(); z = live.xpos[b][2]
        X = sim.data.xfrc_applied; X[0, b, :] = 0; lift = 0.0; tq = np.zeros(3)
        PAD.s = controller.GamepadState(lx=0.0, ly=-0.4 if 1.0 < t < 2.5 else 0.0, rx=0, ry=0, lt=0, rt=0, hat_x=0, hat_y=0, buttons=frozenset())
        if ph == "walk" and t > T0: ph = "lift"; t_ph = t; yaw0 = math.atan2(R[1, 0], R[0, 0])
        if ph not in ("walk", "touchdown"):
            Rup = F.rz(yaw0)
            if ph == "lift":
                lift, tq = fl.wrench(R, w_b, v, Rup, lift=F.LIFT_MAX, kp=KP, kd=KD)
                if t - t_ph > LIFT_T: ph = "flip"; t_ph = t
            elif ph == "flip":
                u = min(1.0, (t - t_ph) / FLIP_T); sm = u * u * (3 - 2 * u); rate = 6 * u * (1 - u) / FLIP_T * 2 * math.pi
                lift, tq = fl.wrench(R, w_b, v, Rup @ F.rx(2 * math.pi * sm), np.array([rate, 0, 0]), lift=0.0, kp=KP, kd=KD)
                if u >= 1.0: ph = "land"; t_ph = t
            elif ph == "land":
                vzd = float(np.clip(-2.5 * (z - 0.12), -3.0, -0.3))
                lift, tq = fl.wrench(R, w_b, v, Rup, vz_des=vzd, kp=KP, kd=KD); lift *= float(np.clip((R[2, 2] - 0.5) / 0.4, 0, 1))
                if z < 0.16 and abs(v[2]) < 0.6: ph = "touchdown"; t_ph = t
            f = tank.draw(F.Flight.nozzle_sum(lift, tq), dt); lift *= f; tq = tq * f
            Fw = R @ np.array([0, 0, lift]); arm = (MB @ live.xipos[rb]) / MB.sum() - live.xipos[b]        # lift nozzles straddle the whole crab's COM
            X[0, b, 0:3] = torch.as_tensor(Fw, dtype=X.dtype); X[0, b, 3:6] = torch.as_tensor(R @ tq + np.cross(arm, Fw), dtype=X.dtype)
        with torch.inference_mode(): obs, _, _, _ = env.step(policy(obs))
        if s % int(0.05 / dt) == 0:
            model, live = sim.env_mjdata(0); R = live.xmat[b].reshape(3, 3)
            print(dict(t=round(t, 2), ph=ph, z=round(float(live.xpos[b][2]), 3), up=round(float(R[2, 2]), 2), roll=round(math.degrees(math.atan2(R[2, 1], R[2, 2]))), vz=round(float(live.qvel[da + 2]), 2), gas=round(tank.gas, 4)), flush=True)
    R = live.xmat[b].reshape(3, 3); ok = R[2, 2] > 0.9 and live.xpos[b][2] > 0.06
    print("RESULT", "SUCCESS" if ok else "FAIL", "up", round(float(R[2, 2]), 2), "z", round(float(live.xpos[b][2]), 3), "gas used", round(F.GAS0 - tank.gas, 4), flush=True)
play._loop = _loop
sys.argv = [str(REPO / "scripts" / "play.py"), "--app", "/workspace/repo-jumper-local-sim/apps/jumper_surround_vision.app", "--scene",
            sys.argv[1], "--backend", "native", "--device", "cpu", "--headless", "--steps", "999999999"]
play.main()
