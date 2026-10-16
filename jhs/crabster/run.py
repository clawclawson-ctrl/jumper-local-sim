"""Crabster stunt show in the local sim. SCRIPTED STUNT, SIM-ONLY THRUSTER + SUCTION CONCEPT.
    python jhs/crabster/run.py --app jumper_surround_vision.app --scene STAGE.map --backend native --device cpu --headless --steps 999999999
Env: CB_STUNT (ramp | kickflip | both | sprint), VS_SEED, VS_OUT, VS_TMAX.
On the ground the crab is the official walking policy, steered through a virtual gamepad. The add-ons are real applied
forces on the crab's base in MuJoCo (xfrc_applied): a suction fan (pull toward the surface, only within its gap), two lift
nozzles, side/pitch/yaw pairs and a fore-aft boost nozzle, all fed from one finite cold-gas tank. The stunt script decides
when they fire; contact, landing and falling are plain physics, so a stunt can fail and is then reported as failed."""
from __future__ import annotations
import importlib.util, json, math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from jhs.crabster import flight as F, scene as S, skin as SK  # noqa: E402

REPO = Path(os.environ.get("JUMPER_REPO", str(HERE.parent.parent / "toolkit")))
STUNT = os.environ.get("CB_STUNT", "both").lower()
SEED = int(os.environ.get("VS_SEED", "1")); TMAX = float(os.environ.get("VS_TMAX", "60"))
OUTD = Path(os.environ.get("VS_OUT", "crabster_run")); OUTD.mkdir(parents=True, exist_ok=True)
LABEL = "CRABSTER -- SCRIPTED STUNT, SIM-ONLY THRUSTER + SUCTION CONCEPT"
CAMERA = os.environ.get("CB_CAMERA", "follow")
ROLL_T = 0.45             # s, the half roll after release
YAW_T = 1.0               # s, the half twist that turns it to face the cars
BOOST_MAX = 18.0          # N, fore-aft boost nozzle (sprint / flights)
CLIMB_MAX = 45.0
RELIEF = float(os.environ.get("CB_RELIEF", 1.0))   # fraction of the crab's weight the nozzles carry on the arch
TUCK = float(os.environ.get("CB_TUCK", 0.0))      # legs-tucked skate pose: blend of default stance (0) and the gait pose (1)
SUC_MIN = float(os.environ.get("CB_SUC", 20.0))     # N, suction normal force floor on the arch          # N, the same nozzle at climb power on the arch
rng = np.random.default_rng(SEED)
x0, y0, h0 = S.START["sprint" if STUNT == "sprint" else ("kickflip" if STUNT == "kickflip" else "ramp")]
x0 += rng.uniform(-0.05, 0.05); y0 += rng.uniform(-0.03, 0.03); h0 += rng.uniform(-2, 2)

sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("play", REPO / "scripts" / "play.py")
play = importlib.util.module_from_spec(_spec); sys.modules["play"] = play; _spec.loader.exec_module(play)
import controller, tasks, torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("VS_THREADS", "2"))); torch.manual_seed(SEED)


class Pad:
    connected = True; path = "virtual"; name = "virtual pad (Crabster stunt script, sim-only)"
    def __init__(self): self.s = controller.GamepadState()
    def state(self): return self.s
    def close(self): pass
    def walk(self, fwd): self.s = controller.GamepadState(lx=0.0, ly=-float(fwd), rx=0.0, ry=0.0, lt=0.0, rt=0.0, hat_x=0, hat_y=0, buttons=frozenset())
PAD = Pad(); controller.open = lambda *a, **k: PAD
_orig = tasks.load_env_cfg
def _pinned(*a, **k):
    cfg = _orig(*a, **k)
    ev = cfg.events.get("reset_base") if getattr(cfg, "events", None) else None
    r = math.radians(h0)
    if ev is not None and "pose_range" in ev.params:
        ev.params["pose_range"] = {"x": (x0, x0), "y": (y0, y0), "z": (0.01, 0.01), "yaw": (r, r)}
    if isinstance(cfg.terminations, dict): cfg.terminations.clear()
    else:
        for n in list(vars(cfg.terminations)): setattr(cfg.terminations, n, None)
    cfg.episode_length_s = 1e6; cfg.seed = SEED; return cfg
tasks.load_env_cfg = _pinned


def minjerk(u): u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u), 6 * u * (1 - u)
def yaw_of(R): return math.atan2(R[1, 0], R[0, 0])


class Show:
    """the stunt script: a list of acts, each a small state machine over phases. Every phase ends on a measured
    condition or a timeout; a timeout or a bad landing marks that act FAILED (the show goes on honestly)."""
    def __init__(self, fl, m):
        self.fl, self.m = fl, m; self.tank = F.Tank()
        self.acts = {"ramp": ["ramp"], "kickflip": ["kickflip"], "both": ["ramp", "kickflip"], "sprint": ["sprint"]}[STUNT]
        self.ai = 0; self.ph = "approach"; self.t_ph = 0.0; self.results = []; self.events = []; self.cars_hit = False
        self.suction = False; self.hold = None; self.sprint = {}
    @property
    def act(self): return self.acts[self.ai] if self.ai < len(self.acts) else None
    def go(self, ph, t, msg=None):
        self.ph = ph; self.t_ph = t
        self.events.append(dict(t=round(t, 3), act=self.act, phase=ph, msg=msg or ph))
    def finish(self, t, ok, why, **kw):
        self.results.append(dict(act=self.act, success=bool(ok), why=why, t=round(t, 2), **kw))
        self.events.append(dict(t=round(t, 3), act=self.act, phase="result", msg=("SUCCESS: " if ok else "FAILED: ") + why))
        self.ai += 1; self.ph = "approach"; self.t_ph = t; self.cars_hit = False

    def step(self, t, R, w, v, p, com):
        th = 0.0
        """-> dict(fwd pad, lift N body z, tq N m body, boost N world xy vector, suction N)."""
        out = dict(fwd=0.0, lift=0.0, tq=np.zeros(3), push=np.zeros(3), suction=0.0)
        a, ph, dt_ph = self.act, self.ph, t - self.t_ph
        fl = self.fl; up = R[2, 2]; hdg = R[:, 0]
        if a == "ramp": hdg = -hdg                        # the crab climbs tail-first (claws trailing, they jam head-first)
        if a is None: return out
        if a == "ramp":
            cx, cz = S.ARCH_X, S.ARCH_R; rho = math.hypot(p[0] - cx, p[2] - cz); th = math.atan2(p[0] - cx, -(p[2] - cz))
            gap = S.ARCH_R - rho if p[0] > cx - 0.05 else p[2]
            if ph == "approach":
                out["fwd"] = -0.7
                if p[0] > cx - 0.35: self.go("climb", t, "suction fan ON -- climbing the arch")
                elif dt_ph > 15: self.finish(t, False, "never reached the arch")
            elif ph == "climb":
                self.suction = True; out["tuck"] = th > math.radians(10); out["fwd"] = 0.0 if out["tuck"] else -0.8
                # boost nozzle along the crab's heading keeps it moving up the wall (P on speed along the surface)
                # weight relief along the surface: the boost nozzle cancels gravity's down-slope pull (RELIEF of it) + sliding
                # friction + a speed loop; the suction fan supplies the normal force. tangent = up-slope direction of the loop.
                tan = np.array([math.cos(th), 0.0, math.sin(th)]); vs = float(v @ tan)
                f_t = self.m * 9.81 * math.sin(th) * RELIEF + 0.30 * (SUC_MIN + self.m * 9.81 * max(0.0, math.cos(th))) + self.m * 8.0 * (0.50 - vs)
                out["push"] = tan * float(np.clip(f_t, 0, CLIMB_MAX)) + np.array([0.0, float(np.clip(self.m * (-14.0 * p[1] - 7.0 * v[1]), -20, 20)), 0.0])
                ey = R.T @ np.array([0.0, 1.0, 0.0]); out["tq"] = np.clip(np.array([0.0, 0.0, fl.I[2] * (25.0 * math.atan2(-ey[0], ey[1]) - 7.0 * w[2])]), -F.TQ_MAX, F.TQ_MAX)
                if th > math.radians(172) or (th > math.radians(150) and up < -0.93 and p[0] < cx + 0.08):
                    self.go("hang", t, "inverted at the top of the arch -- hanging on suction")
                elif dt_ph > 14: self.finish(t, False, f"stalled on the arch at {math.degrees(th):.0f} deg")
            elif ph == "hang":
                self.suction = True; out["tuck"] = True
                if dt_ph > 1.6:
                    self.go("release", t, "suction OFF -- release: half roll (+ twist to face the cars)"); self.hold = R.copy(); self.t_rel = t
                    self.tw = math.remainder(math.pi - yaw_of(self.hold @ F.rx(math.pi)), 2 * math.pi)   # twist needed to face the cars (-x)
            elif ph == "release":
                self.suction = False; s, sd = minjerk(dt_ph / ROLL_T); y2, yd = minjerk(dt_ph / YAW_T)
                out["lift"], out["tq"] = fl.wrench(R, w, v, self.hold @ F.rx(math.pi * s) @ F.rz(self.tw * y2), np.array([math.pi * sd / ROLL_T, 0, self.tw * yd / YAW_T]), lift=0.0)
                out["push"] = np.array([self.m * 3.0 * (-0.35 - v[0]), 0, 0]); out["push"][0] = float(np.clip(out["push"][0], -BOOST_MAX, BOOST_MAX))
                if dt_ph > ROLL_T: self.go("land", t, "landing burn")
            elif ph == "land":
                y2, yd = minjerk((t - self.t_rel) / YAW_T); Rt = self.hold @ F.rx(math.pi) @ F.rz(self.tw * y2); Rt = F.rz(yaw_of(Rt))   # roll done, yaw finishing: face the cars
                lift, tq = fl.wrench(R, w, v, Rt, np.array([0, 0, self.tw * yd / YAW_T]), vz_des=float(np.clip(-2.5 * (p[2] - (0.12 if y2 > 0.97 else 0.40)), -3.0, -0.3 if y2 > 0.97 else 0.5)))   # hover until the twist is done
                out["lift"] = lift * float(np.clip((up - 0.5) / 0.4, 0, 1)); out["tq"] = tq
                out["push"] = np.array([float(np.clip(self.m * 3.0 * (0 - v[0]), -BOOST_MAX, BOOST_MAX)), 0, 0])
                if p[2] < 0.16 and abs(v[2]) < 0.6 and y2 > 0.97: self.go("settle", t, "touchdown")
                elif dt_ph > 3: self.go("settle", t, "touchdown (late)")
            elif ph == "settle":
                if dt_ph > 1.5:
                    face = math.degrees(abs(math.remainder(yaw_of(R) - math.pi, 2 * math.pi)))
                    ok = up > 0.9 and p[2] > 0.06 and face < 30
                    self.finish(t, ok, f"landed on its feet facing the cars ({face:.0f} deg off)" if ok else f"bad landing (up {up:.2f}, {face:.0f} deg off the cars)", x=round(float(p[0]), 2))
        elif a == "kickflip":
            land_x = S.CARS_X - 0.75
            if ph == "approach":                                # turn-free: the ramp act lands facing -x, the solo act starts so
                out["fwd"] = 0.7
                if p[0] < S.CARS_X + 0.62: self.go("lift", t, "lift nozzles -- launch over the cars"); self.hold = F.rz(yaw_of(R))
                elif dt_ph > 20: self.finish(t, False, "never reached the cars")
            elif ph in ("lift", "flip", "land"):
                vxd = -0.95 if ph != "land" else (0.0 if p[0] < S.CARS_X - 0.35 else -0.6)
                out["push"] = np.array([float(np.clip(self.m * 4.0 * (vxd - v[0]), -BOOST_MAX, BOOST_MAX)), float(np.clip(self.m * 3.0 * (-v[1] - p[1]), -6, 6)), 0])
                if ph == "lift":
                    out["lift"], out["tq"] = fl.wrench(R, w, v, self.hold, lift=F.LIFT_MAX)
                    if dt_ph > 0.27: self.go("flip", t, "side thrusters -- 360 kickflip")
                elif ph == "flip":
                    s, sd = minjerk(dt_ph / 0.6)
                    out["lift"], out["tq"] = fl.wrench(R, w, v, self.hold @ F.rx(2 * math.pi * s), np.array([2 * math.pi * sd / 0.6, 0, 0]), lift=0.0)
                    if dt_ph > 0.6: self.go("land", t, "landing burn")
                else:
                    lift, tq = fl.wrench(R, w, v, self.hold, vz_des=float(np.clip(-2.5 * (p[2] - 0.12), -3.0, -0.3)))
                    out["lift"] = lift * float(np.clip((up - 0.5) / 0.4, 0, 1)); out["tq"] = tq
                    if p[2] < 0.16 and abs(v[2]) < 0.6 and abs(v[0]) < 0.35: self.go("settle", t, "touchdown")
                    elif dt_ph > 3: self.go("settle", t, "touchdown (late)")
            elif ph == "settle":
                if dt_ph > 1.5:
                    ok = up > 0.9 and p[2] > 0.06 and p[0] < S.CARS_X - 0.3 and not self.cars_hit
                    why = "cleared all five cars, landed on its feet" if ok else ("hit a car" if self.cars_hit else f"bad landing (up {up:.2f}, x {p[0]:.2f})")
                    self.finish(t, ok, why, x=round(float(p[0]), 2))
        elif a == "sprint":
            sp = self.sprint
            if ph == "approach":
                if dt_ph > 1.0: self.go("run", t, "GO -- sprint (walking + boost nozzle)"); sp.update(t0=t, x0=float(p[0]), t1=None, vmax=0.0)
            elif ph == "run":
                out["fwd"] = 1.0; vx = float(v @ hdg); sp["vmax"] = max(sp["vmax"], vx)
                out["push"] = hdg * BOOST_MAX * float(up > 0.8)
                if sp["t1"] is None and vx >= 1.5: sp["t1"] = t - sp["t0"]; self.events.append(dict(t=round(t, 3), act=a, phase="mark", msg=f"0-1.5 m/s in {sp['t1']:.2f} s (sim)"))
                if p[0] - sp["x0"] > 4.0 or dt_ph > 12: sp["t4"] = t - sp["t0"]; self.go("brake", t, "brake")
            elif ph == "brake":
                vx = float(v @ hdg); out["push"] = hdg * float(np.clip(-self.m * 4 * vx, -BOOST_MAX, BOOST_MAX)) * float(up > 0.8)
                if dt_ph > 2.0:
                    ok = up > 0.9 and p[2] > 0.06 and sp.get("t1") is not None
                    self.finish(t, ok, (f"0-1.5 m/s in {sp['t1']:.2f} s, 4 m in {sp['t4']:.2f} s, top {sp['vmax']:.2f} m/s (sim)" if sp.get("t1") else "never reached 1.5 m/s")
                                if up > 0.9 else "fell over", t_0_1p5=sp.get("t1"), t_4m=sp.get("t4"), vmax=round(sp["vmax"], 2))
        out["suction"] = (float(np.clip(self.m * 9.81 * max(0.0, -math.cos(th)) + SUC_MIN, SUC_MIN, F.SUCTION_N))
                          if (self.suction and a == "ramp" and gap < 0.10 + 0.12 and th > math.radians(10)) else 0.0)
        if self.ph not in ("climb", "hang"): self.suction = False
        return out


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
    import mujoco
    un = env.unwrapped; sim = un.sim; model, live = sim.env_mjdata(0); dt = un.step_dt
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link"); da = model.jnt_dofadr[model.body_jntadr[b]]
    rb = [i for i in range(model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("robot/")]
    model.body_mass[b] += F.DRY_ADD + F.GAS0
    rm = getattr(sim, "render_model", None)
    if rm is not None and rm is not model: rm.body_mass[b] = model.body_mass[b]
    mujoco.mj_forward(model, live)
    MB = np.array([model.body_mass[i] for i in rb]); m = float(MB.sum())
    R0 = live.xmat[b].reshape(3, 3); I = np.zeros((3, 3))
    for i in rb:
        Ri = live.ximat[i].reshape(3, 3); d = live.xipos[i] - live.xpos[b]
        I += Ri @ np.diag(model.body_inertia[i]) @ Ri.T + model.body_mass[i] * (d @ d * np.eye(3) - np.outer(d, d))
    # legs-tucked skate: while the script holds a pose, the joint targets are frozen to it instead of the policy's
    HOLD = {"q": None}; mgr = un.action_manager; term = mgr.get_term(list(mgr.active_terms)[0]); robot = un.scene["robot"]
    def _apply():
        if HOLD["q"] is not None: robot.set_joint_position_target(HOLD["q"])
        else: policy._apply()
    term.apply_actions = _apply
    show = Show(F.Flight(inertia=np.diag(R0.T @ I @ R0), mass=m), m)
    robot_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in rb}
    car_geoms = {g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("car_")}
    SHOTS = [x for x in os.environ.get("CB_SHOTS", "").split(",") if x]          # "ramp:hang+0.8,kickflip:flip+0.3" -> PNGs
    LIVE = os.environ.get("JHS_LIVE", "0") == "1"; LDIR = Path(os.environ.get("JHS_LIVE_DIR", str(OUTD / "live"))); LDIR.mkdir(parents=True, exist_ok=True)
    STOPF = Path(os.environ.get("JHS_STOP", str(LDIR / "STOP"))); SPEED = float(os.environ.get("JHS_SPEED", "0")); shot_done = set()
    painter = None
    if SHOTS or LIVE:
        from jhs.crabster.view import Painter, caption
        W_, H_ = (int(os.environ.get("JHS_LIVE_W", 1280)), int(os.environ.get("JHS_LIVE_H", 720))) if LIVE else (1280, 720)
        painter = Painter(rm if rm is not None else model, W_, H_, CAMERA)
    import time as _time
    wall0 = _time.time(); last_live = -1e9; last_status = 0.0
    obs = env.get_observations(); rec = dict(t=[], qpos=[], lift=[], tq=[], push=[], suction=[], gas=[], g=[], phase=[])
    vprev = None; gnow = 1.0; every = max(1, int(round(0.04 / dt)))
    for s in range(int(TMAX / dt)):
        t = s * dt; model, live = sim.env_mjdata(0)
        R = live.xmat[b].reshape(3, 3).copy(); w = live.qvel[da + 3:da + 6].copy(); v = live.qvel[da:da + 3].copy(); p = live.xpos[b].copy()
        com = (MB @ live.xipos[rb]) / m
        if show.act == "kickflip" and not show.cars_hit:
            for c in live.contact[:live.ncon]:
                if (c.geom1 in car_geoms and c.geom2 in robot_geoms) or (c.geom2 in car_geoms and c.geom1 in robot_geoms):
                    show.cars_hit = True; show.events.append(dict(t=round(t, 3), act="kickflip", phase="hit", msg="contact with a car")); break
        u = show.step(t, R, w, v, p, com)
        if os.environ.get("CB_DEBUG") and abs(t - float(os.environ["CB_DEBUG"])) < dt / 2:
            for c in live.contact[:live.ncon]:
                n1, n2 = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (c.geom1, c.geom2)); print("CONTACT", n1, n2, np.round(c.pos, 3), flush=True)
        PAD.walk(u["fwd"])
        if u.get("tuck") and HOLD["q"] is None:
            q = policy._targets.clone(); d = robot.data.default_joint_pos.clone(); HOLD["q"] = d + TUCK * (q - d) if TUCK < 1 else q
        elif not u.get("tuck"): HOLD["q"] = None
        lift, tq, push = u["lift"], u["tq"], u["push"]
        f = show.tank.draw(F.Flight.nozzle_sum(lift, tq) + float(np.linalg.norm(push)), dt); lift *= f; tq = tq * f; push = push * f
        Fw = R @ np.array([0, 0, lift]) + push - R[:, 2] * u["suction"]       # the fan pulls along -body z (toward the surface)
        arm = com - live.xipos[b]
        X = sim.data.xfrc_applied; X[0, b, :] = 0
        X[0, b, 0:3] = torch.as_tensor(Fw, dtype=X.dtype); X[0, b, 3:6] = torch.as_tensor(R @ tq + np.cross(arm, Fw), dtype=X.dtype)
        with torch.inference_mode(): obs, _, _, _ = env.step(policy(obs))
        if painter is not None and (SHOTS or LIVE):
            snap = dict(lift=lift, tq=tq, push=push, suction=u["suction"], gas=show.tank.gas, g=gnow, phase=f"{show.act}:{show.ph}")
            for sh in SHOTS:
                key, off = sh.split("+"); a_, ph_ = key.split(":")
                if sh not in shot_done and show.act == a_ and show.ph == ph_ and t - show.t_ph >= float(off):
                    shot_done.add(sh); model, live = sim.env_mjdata(0)
                    painter.frame(live.qpos.copy(), t, snap, caption(show.events, t), show.results, "sim still").save(OUTD / f"shot_{a_}_{ph_}.png"); print("SHOT", sh, flush=True)
            if LIVE and t - last_live >= 0.2:
                last_live = t; model, live = sim.env_mjdata(0); wall = _time.time() - wall0
                fr = painter.frame(live.qpos.copy(), t, snap, caption(show.events, t), show.results, f"live, {t / wall:.2f}x" if wall > 1 else "live")
                import io
                buf = io.BytesIO(); fr.save(buf, "JPEG", quality=82); (LDIR / "live.jpg.tmp").write_bytes(buf.getvalue()); os.replace(LDIR / "live.jpg.tmp", LDIR / "live.jpg")
        if _time.time() - last_status > 1.0:
            last_status = _time.time(); wall = last_status - wall0
            (LDIR / "status.json").write_text(json.dumps(dict(t=round(t, 2), wall=round(wall, 1), rate=round(t / wall, 3) if wall > 1 else None, kind="crabster",
                                                              state=f"{show.act}: {show.ph}" if show.act else "show over", gas_g=round(show.tank.gas * 1000), results=show.results)))
        if SPEED > 0:
            ahead = t / SPEED - (_time.time() - wall0)
            if ahead > 0.002: _time.sleep(min(ahead, 0.5))
        if STOPF.exists(): show.events.append(dict(t=round(t, 3), act=show.act, phase="stop", msg="stopped from the page")); break
        if s % every == 0:
            model, live = sim.env_mjdata(0); vn = live.qvel[da:da + 3].copy()
            if vprev is not None: gnow = float(np.linalg.norm((vn - vprev) / (every * dt) + np.array([0, 0, 9.81])) / 9.81)
            vprev = vn
            for k, val in (("t", round(t, 3)), ("qpos", live.qpos.copy()), ("lift", lift), ("tq", np.asarray(tq, float)), ("push", np.asarray(push, float)),
                           ("suction", u["suction"]), ("gas", show.tank.gas), ("g", gnow), ("phase", f"{show.act}:{show.ph}")): rec[k].append(val)
        if s % int(1.0 / dt) == 0:
            print(f"t={t:5.1f} {show.act}:{show.ph:9s} x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:.2f} up={R[2, 2]:+.2f} gas={show.tank.gas:.3f} suc={u['suction']:.0f}", flush=True)
        if show.act is None and not show.results[-1:] == []:
            if t - show.t_ph > 2.0: break
    np.savez_compressed(OUTD / "qpos.npz", **{k: np.array(v) for k, v in rec.items()})
    (OUTD / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in show.events))
    (OUTD / "vision_meta.json").write_text("[]"); np.savez_compressed(OUTD / "vision_tof.npz", tof=np.zeros((0, 1)))   # no vision in this app
    res = dict(kind="crabster", t_end=round(float(rec["t"][-1]) if rec["t"] else 0.0, 2), stopped=STOPF.exists(), camera=CAMERA, label=LABEL, stunt=STUNT, seed=SEED, results=show.results, gas_left=round(show.tank.gas, 4), gas0=F.GAS0,
               success=bool(show.results) and all(r["success"] for r in show.results) and len(show.results) == len(show.acts))
    (OUTD / "result.json").write_text(json.dumps(res, indent=1)); print("RESULT", json.dumps(res), flush=True)


play._loop = _loop


def main():
    play.main()


if __name__ == "__main__":
    main()
