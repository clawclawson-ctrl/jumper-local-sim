"""Sim add-on drawing (turrets + projectiles) into a MuJoCo scene, used by the crab cameras, the live view and the MP4.
SIM-ONLY: the turret and shots are not part of the robot model; physics never touches them."""
from __future__ import annotations
import math
import numpy as np
from . import arena as A

SHOT_RGBA = np.array([0.0, 1.0, 1.0, 1.0], np.float32)     # bright glowing cyan (a colour nothing else in the arena has)
SHOT_DRAW_R = 0.045                                           # drawn head radius (the hit test uses arena.HIT_R)
TRAIL = ((0.07, 0.030, 0.75), (0.13, 0.021, 0.5), (0.19, 0.013, 0.3))   # (distance behind, radius, alpha)
TURRET_RGBA = np.array([0.18, 0.18, 0.20, 1.0], np.float32)
BARREL_RGBA = np.array([0.55, 0.55, 0.58, 1.0], np.float32)
TEAM_RGBA = {"A": np.array([0.95, 0.30, 0.15, 1.0], np.float32), "B": np.array([0.20, 0.35, 0.95, 1.0], np.float32)}


def _add(mj, scn, typ, size, pos, mat, rgba, emission=0.0):
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(g, typ, np.asarray(size, np.float64), np.asarray(pos, np.float64), np.asarray(mat, np.float64).reshape(9), np.asarray(rgba, np.float32))
    g.emission = emission; g.segid = -1; scn.ngeom += 1


def _rz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def draw(mj, scn, crabs, turrets, shots, skip=None):
    """crabs {k: (x, y, z)} base positions; turrets {k: world yaw}; shots [dict(x, y, owner)]; skip = crab whose turret is not drawn
    (its own camera)."""
    for k, p in crabs.items():
        if k == skip or p is None: continue
        a = turrets.get(k, 0.0); R = _rz(a)
        base = np.array([p[0], p[1], A.SHOT_Z - 0.012])
        _add(mj, scn, mj.mjtGeom.mjGEOM_CYLINDER, [0.045, 0.014, 0], base, np.eye(3), TURRET_RGBA)
        _add(mj, scn, mj.mjtGeom.mjGEOM_BOX, [0.03, 0.025, 0.006], base + np.array([0, 0, 0.016]), R, TEAM_RGBA[k])
        # barrel: capsule along the turret yaw (capsule axis = local z -> rotate z onto x)
        Rb = R @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
        _add(mj, scn, mj.mjtGeom.mjGEOM_CAPSULE, [0.011, 0.06, 0], base + R @ np.array([0.07, 0, 0.004]), Rb, BARREL_RGBA)
    for s in shots:
        _add(mj, scn, mj.mjtGeom.mjGEOM_SPHERE, [SHOT_DRAW_R, 0, 0], [s["x"], s["y"], A.SHOT_Z], np.eye(3), SHOT_RGBA, emission=1.0)
        vx, vy = s.get("vx", 0.0), s.get("vy", 0.0); n = math.hypot(vx, vy)
        if n < 1e-6: continue
        for dist, r, al in TRAIL:      # a short fading trail behind the head (separate small spheres)
            c = SHOT_RGBA.copy(); c[3] = al
            _add(mj, scn, mj.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], [s["x"] - vx / n * dist, s["y"] - vy / n * dist, A.SHOT_Z], np.eye(3), c, emission=1.0)
