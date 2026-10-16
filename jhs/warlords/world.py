"""Three crabs in one world (local sim): the official player's env plus robot entities "robot2" / "robot3", each with its own dToF
("tof2" / "tof3"); same machinery as jhs.soccer.world (two crabs). Plus render-only team skins (RED / BLUE / GREEN). SIM-ONLY."""
from __future__ import annotations
import copy, math, re

EXTRA = ("robot2", "robot3")


def install(tasks, A, B, C, seed):
    orig = tasks.load_env_cfg
    def pinned(*a, **k):
        cfg = orig(*a, **k)
        ev = cfg.events.get("reset_base") if getattr(cfg, "events", None) else None
        r = math.radians(A[2])
        if ev is not None and "pose_range" in ev.params:
            ev.params["pose_range"] = {"x": (A[0], A[0]), "y": (A[1], A[1]), "z": (0.01, 0.01), "yaw": (r, r)}
        rob = cfg.scene.entities["robot"]
        sens = list(cfg.scene.sensors or ())
        t1 = next((s for s in sens if getattr(s, "name", "") == "tof"), None)
        if t1 is None:
            from tasks.jumper.common.tof import tof_sensor
            t1 = tof_sensor(); sens.append(t1)
        for name, P, tn in (("robot2", B, "tof2"), ("robot3", C, "tof3")):
            r2 = copy.deepcopy(rob); h = math.radians(P[2]) / 2
            r2.init_state.pos = (P[0], P[1], rob.init_state.pos[2] + 0.01); r2.init_state.rot = (math.cos(h), 0.0, 0.0, math.sin(h))
            cfg.scene.entities[name] = r2
            t2 = copy.deepcopy(t1); t2.name = tn
            for ref in (t2.frame, t2.pattern.camera, t2.pattern.body):
                try: ref.entity = name
                except Exception: object.__setattr__(ref, "entity", name)
            sens.append(t2)
        cfg.scene.sensors = tuple(sens)
        if isinstance(cfg.terminations, dict): cfg.terminations.clear()
        else:
            for n in list(vars(cfg.terminations)): setattr(cfg.terminations, n, None)
        cfg.episode_length_s = 1e6; cfg.seed = seed
        return cfg
    tasks.load_env_cfg = pinned


TEAM_RGB = {"robot": (0.90, 0.24, 0.10, 1.0), "robot2": (0.12, 0.20, 0.55, 1.0), "robot3": (0.10, 0.55, 0.16, 1.0)}
BELT_RGB = {"robot": (1.00, 0.86, 0.12, 1.0), "robot2": (0.40, 0.43, 0.48, 1.0), "robot3": (0.85, 0.80, 0.55, 1.0)}
GREY = (0.30, 0.30, 0.32, 1.0); VISOR = (0.05, 0.05, 0.07, 1.0); STOCK_RED = (0.85, 0.08, 0.05)


def skin(model):
    import mujoco
    n = 0
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        m = re.match(r"(robot[23]?)/(.*)$", name)
        if not m or model.geom_rgba[g][3] == 0: continue
        who, part = m.group(1), m.group(2)
        if re.match(r"base_link", part): c = BELT_RGB[who]
        elif re.search(r"(display_module|camera|tof_sensor)_link", part): c = VISOR
        elif all(abs(a - b) < 0.05 for a, b in zip(model.geom_rgba[g][:3], STOCK_RED)): c = TEAM_RGB[who]
        elif re.search(r"(shell|shoulder|hip|upper_arm)", part): c = TEAM_RGB[who]
        else: c = GREY
        model.geom_matid[g] = -1; model.geom_rgba[g] = c; n += 1
    return n
