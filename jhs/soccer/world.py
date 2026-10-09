"""Two crabs in one world (local sim): the official player's env with a second robot entity ("robot2") and its own
dToF ("tof2"), the same machinery as the earlier two-crab sumo/tag runs. Also the render-only team skins.
SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import copy, math, os, re, sys


def install(tasks, A, B, seed):
    """patch tasks.load_env_cfg: robot A at A=(x, y, deg), robot2 at B, a dToF on robot2, no fall resets."""
    orig = tasks.load_env_cfg
    def pinned(*a, **k):
        cfg = orig(*a, **k)
        ev = cfg.events.get("reset_base") if getattr(cfg, "events", None) else None
        r = math.radians(A[2])
        if ev is not None and "pose_range" in ev.params:
            ev.params["pose_range"] = {"x": (A[0], A[0]), "y": (A[1], A[1]), "z": (0.01, 0.01), "yaw": (r, r)}
        rob = cfg.scene.entities["robot"]; r2 = copy.deepcopy(rob); h = math.radians(B[2]) / 2
        r2.init_state.pos = (B[0], B[1], rob.init_state.pos[2] + 0.01); r2.init_state.rot = (math.cos(h), 0.0, 0.0, math.sin(h))
        cfg.scene.entities["robot2"] = r2
        sens = list(cfg.scene.sensors or ())
        t1 = next((s for s in sens if getattr(s, "name", "") == "tof"), None)
        if t1 is None:
            from tasks.jumper.common.tof import tof_sensor
            t1 = tof_sensor(); sens.append(t1)
        t2 = copy.deepcopy(t1); t2.name = "tof2"
        for ref in (t2.frame, t2.pattern.camera, t2.pattern.body):
            try: ref.entity = "robot2"
            except Exception: object.__setattr__(ref, "entity", "robot2")
        sens.append(t2); cfg.scene.sensors = tuple(sens)
        if isinstance(cfg.terminations, dict): cfg.terminations.clear()
        else:
            for n in list(vars(cfg.terminations)): setattr(cfg.terminations, n, None)
        cfg.episode_length_s = 1e6; cfg.seed = seed
        return cfg
    tasks.load_env_cfg = pinned


def share_controller():
    """one controller extension per process (a pyo3 module initialises once): the second app reuses the first's."""
    import mjrl.app_play as ap
    ext = {}
    orig = ap._load_extension
    def once(bundle, manifest):
        if "m" not in ext: ext["m"] = orig(bundle, manifest); return ext["m"]
        return ext["m"][0], f"{ext['m'][1]} (shared with the first crab)"
    ap._load_extension = once
    return ap


class _UW:
    def __init__(self, uw, robot): self._uw = uw; self.scene = {"robot": robot}
    def __getattr__(self, k): return getattr(self._uw, k)


class EnvView:
    """the env as robot B's controller sees it: the same world, its own robot."""
    def __init__(self, env, robot): self.unwrapped = _UW(env.unwrapped, robot); self.action_space = env.action_space


TEAM_RGB = {"robot": (0.90, 0.24, 0.10, 1.0), "robot2": (0.12, 0.20, 0.55, 1.0)}
BELT_RGB = {"robot": (1.00, 0.86, 0.12, 1.0), "robot2": (0.40, 0.43, 0.48, 1.0)}
GREY = (0.30, 0.30, 0.32, 1.0); VISOR = (0.05, 0.05, 0.07, 1.0)
STOCK_RED = (0.85, 0.08, 0.05)


def skin(model):
    """team colours on the crabs' drawn geoms (rgba only; physics does not read it). RED = vermilion + gold belt,
    BLUE = navy + steel belt; the stock light greys are darkened on both so nothing on a crab is ball-white."""
    import mujoco
    n = 0
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        m = re.match(r"(robot2?)/(.*)$", name)
        if not m or model.geom_rgba[g][3] == 0: continue
        who, part = m.group(1), m.group(2)
        if re.match(r"base_link", part): c = BELT_RGB[who]
        elif re.search(r"(display_module|camera|tof_sensor)_link", part): c = VISOR
        elif all(abs(a - b) < 0.05 for a, b in zip(model.geom_rgba[g][:3], STOCK_RED)): c = TEAM_RGB[who]
        elif re.search(r"(shell|shoulder|hip|upper_arm)", part): c = TEAM_RGB[who]
        else: c = GREY
        model.geom_matid[g] = -1; model.geom_rgba[g] = c; n += 1
    return n
