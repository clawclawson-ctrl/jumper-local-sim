"""CYBER theme + glowing trail walls drawn into a MuJoCo scene (crab cameras, live view, MP4). SIM-ONLY: the trails are a sim
add-on, physics never touches them."""
from __future__ import annotations
import math, re
import numpy as np
from . import game as G

GRID_RGBA = np.array([0.30, 0.16, 0.55, 1.0], np.float32)    # dim violet floor grid (hue ~265: not a team or trail colour)
FLOOR_RGBA = (0.02, 0.015, 0.05, 1.0)
WALL_RGBA = (0.42, 0.18, 1.0, 1.0)                           # electric violet rim walls
DARK_RGBA = (0.04, 0.04, 0.08, 1.0)
NEON_TEAM = {"robot": (1.0, 0.12, 0.30, 1.0), "robot2": (0.15, 0.35, 1.0, 1.0)}   # still inside the red / blue vision families


def theme(model):
    """dark floor + neon rim walls + neon-tinted crabs (rgba only; call after jhs.soccer.world.skin)."""
    import mujoco
    from jhs.soccer.world import TEAM_RGB
    n = 0
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        m = re.match(r"(robot2?)/", name)
        if m:
            if all(abs(a - b) < 0.02 for a, b in zip(model.geom_rgba[g][:3], TEAM_RGB[m.group(1)][:3])):
                model.geom_rgba[g] = NEON_TEAM[m.group(1)]; n += 1
            continue
        if model.geom_rgba[g][3] == 0: continue
        model.geom_matid[g] = -1
        if name == "ground": model.geom_rgba[g] = FLOOR_RGBA
        elif name.startswith(("wall_", "corner_")): model.geom_rgba[g] = WALL_RGBA
        else: model.geom_rgba[g] = DARK_RGBA
        n += 1
    for t in range(model.ntex):          # night sky: the skybox becomes a deep violet-black
        if int(model.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
            a = model.tex_adr[t]; nb = model.tex_width[t] * model.tex_height[t] * model.tex_nchannel[t]
            px = model.tex_data[a:a + nb].reshape(-1, model.tex_nchannel[t]); px[:, :3] = (px[:, :3].astype(float) * [0.06, 0.03, 0.12]).astype(px.dtype)
    model.vis.headlight.ambient[:] = [0.35, 0.35, 0.40]
    return n


def _add(mj, scn, typ, size, pos, mat, rgba, emission=0.0):
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(g, typ, np.asarray(size, np.float64), np.asarray(pos, np.float64), np.asarray(mat, np.float64).reshape(9), np.asarray(rgba, np.float32))
    g.emission = emission; g.segid = -1; scn.ngeom += 1


def _rz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def draw_grid(mj, scn, step=0.5):
    for x in np.arange(-G.FX + step, G.FX, step):
        _add(mj, scn, mj.mjtGeom.mjGEOM_BOX, [0.006, G.FY - 0.1, 0.0008], [x, 0, 0.001], np.eye(3), GRID_RGBA, 0.6)
    for y in np.arange(-G.FY + step, G.FY, step):
        _add(mj, scn, mj.mjtGeom.mjGEOM_BOX, [G.FX - 0.1, 0.006, 0.0008], [0, y, 0.001], np.eye(3), GRID_RGBA, 0.6)


def draw(mj, scn, trails, grid=True):
    """trails {k: [(x, y), ...]} (point lists): one glowing low box per ~10 cm, plus a bright cap line on top."""
    if grid: draw_grid(mj, scn)
    for k, pts in trails.items():
        col = np.array(G.TRAIL_RGBA[k], np.float32); top = np.minimum(1.0, col + 0.35); top[3] = 1.0
        if len(pts) < 2: continue
        P = [pts[0]]
        for p in pts[1:]:
            if math.hypot(p[0] - P[-1][0], p[1] - P[-1][1]) >= 0.10 or p is pts[-1]: P.append(p)
        for a, b in zip(P[:-1], P[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy)
            if L < 1e-4: continue
            R = _rz(math.atan2(dy, dx)); c = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
            _add(mj, scn, mj.mjtGeom.mjGEOM_BOX, [L / 2 + 0.01, G.TRAIL_W / 2, G.TRAIL_H / 2], c + [G.TRAIL_H / 2], R, col, 0.85)
            _add(mj, scn, mj.mjtGeom.mjGEOM_BOX, [L / 2 + 0.01, G.TRAIL_W / 2 + 0.006, 0.004], c + [G.TRAIL_H + 0.002], R, top, 1.0)
