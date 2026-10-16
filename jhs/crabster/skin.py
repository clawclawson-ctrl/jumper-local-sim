"""Crabster look -- render only (rgba + extra scene geoms; physics never reads them). SIM-ONLY CONCEPT.
Pearl-white Roadster skin: glossy pearl white shell, a black glass canopy band (display + sensors), slim red accents,
a white light bar across the front, glowing thruster nozzles, and a rear spoiler (wing on two struts)."""
from __future__ import annotations
import re
import numpy as np

PEARL = (0.93, 0.93, 0.95, 1.0); GLASS = (0.02, 0.025, 0.035, 1.0); RED = (0.80, 0.04, 0.05, 1.0)
DARK = (0.10, 0.10, 0.11, 1.0); SATIN = (0.80, 0.80, 0.83, 1.0)


def skin(model):
    """recolour the crab's drawn geoms; give them a pearl gloss (specular/shininess/reflectance on a shared material)."""
    import mujoco
    n = 0
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        m = re.match(r"robot/(.*)$", name)
        if not m or model.geom_rgba[g][3] == 0: continue
        part = m.group(1)
        if re.search(r"(display_module|camera|tof_sensor)_link", part): c = GLASS              # the black glass canopy band
        elif re.search(r"(shoulder)", part): c = RED                                         # slim red accents
        elif re.search(r"(foot_tip|finger_tip|pad)", part): c = DARK
        elif re.search(r"(calf|palm)", part): c = SATIN
        else: c = PEARL                                                                     # shell, base, hips, upper arms, fingers
        model.geom_matid[g] = -1; model.geom_rgba[g] = c; n += 1
    return n


def base_extent(model, data, base):
    """half-extent box of the base body's drawn geoms in its own frame (from their aabbs)."""
    lo, hi = np.full(3, 1e9), np.full(3, -1e9)
    Rb = data.xmat[base].reshape(3, 3)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != base: continue
        c = model.geom_aabb[g][:3]; h = model.geom_aabb[g][3:]
        R = data.geom_xmat[g].reshape(3, 3); off = Rb.T @ (data.geom_xpos[g] - data.xpos[base]); Rr = Rb.T @ R
        ext = np.abs(Rr) @ h; ctr = off + Rr @ c
        lo = np.minimum(lo, ctr - ext); hi = np.maximum(hi, ctr + ext)
    return lo, hi


def _add(scn, typ, size, pos, R, rgba, emission=0.0):
    import mujoco
    if scn.ngeom >= scn.maxgeom: return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, typ, np.array(size, float), np.array(pos, float), np.asarray(R, float).reshape(-1), np.array(rgba, np.float32))
    g.category = mujoco.mjtCatBit.mjCAT_DECOR; g.emission = emission; scn.ngeom += 1; return g


def nozzles(lo, hi):
    """nozzle spots in the base frame: two lift (belly), roll pair each side (up/down), boost at the tail."""
    zc = (lo[2] + hi[2]) / 2
    return dict(lift=[(0.05, 0.0, lo[2]), (-0.05, 0.0, lo[2])],
                roll=[(0.0, hi[1], zc + 0.015), (0.0, lo[1], zc + 0.015)],
                boost=[(lo[0], 0.0, zc)], front=[(hi[0], 0.0, zc)])


def decor(scn, data, base, lo, hi, lift=0.0, tq=(0, 0, 0), push=(0, 0, 0)):
    """spoiler + light bar + glowing nozzles + thruster plumes, appended to an mjvScene after update_scene."""
    import mujoco
    BOX, SPH, CAP = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE
    R = data.xmat[base].reshape(3, 3); p = data.xpos[base]; P = lambda v: p + R @ np.asarray(v, float)
    w = (hi[1] - lo[1]) * 0.42; xr = lo[0] + 0.015; zt = hi[2] + 0.045
    # rear spoiler: pearl wing with a red gurney lip, black struts and end plates
    _add(scn, BOX, (0.035, w, 0.006), P((xr, 0, zt)), R @ _ry(-0.25), PEARL)
    _add(scn, BOX, (0.004, w, 0.008), P((xr - 0.034, 0, zt + 0.006)), R, RED)
    for sy in (-1, 1):
        _add(scn, BOX, (0.006, 0.006, (zt - hi[2]) / 2), P((xr + 0.01, sy * w * 0.6, (zt + hi[2]) / 2)), R, GLASS)
        _add(scn, BOX, (0.04, 0.003, 0.02), P((xr, sy * w, zt)), R, GLASS)
    # front light bar (emissive)
    _add(scn, BOX, (0.004, (hi[1] - lo[1]) * 0.36, 0.004), P((hi[0] + 0.003, 0, (lo[2] + hi[2]) / 2 + 0.02)), R, (1.0, 0.98, 0.92, 1), emission=1.0)
    # nozzles glow; brighter (and a plume) when firing
    nz = nozzles(lo, hi); tq = np.asarray(tq, float); pb = R.T @ np.asarray(push, float)
    fire = dict(lift=[lift / 2] * 2, roll=[max(0.0, tq[0]) / 0.09, max(0.0, -tq[0]) / 0.09], boost=[max(0.0, pb[0])], front=[max(0.0, -pb[0])])
    dirs = dict(lift=[(0, 0, -1)] * 2, roll=[(0, 0, -1), (0, 0, -1)], boost=[(-1, 0, 0)], front=[(1, 0, 0)])
    for k, pts in nz.items():
        for (pos, f, d) in zip(pts, fire[k], dirs[k]):
            on = min(1.0, f / 30.0)
            _add(scn, SPH, (0.011, 0, 0), P(pos), np.eye(3), (0.35 + 0.65 * on, 0.65 + 0.3 * on, 1.0, 1), emission=0.4 + 0.6 * on)
            if f > 0.5:
                L = 0.03 + 0.14 * on; a = P(pos); b = a + R @ np.asarray(d, float) * L
                for rad, rgba, frac in ((0.017, (0.55, 0.78, 1.0, 0.45), 1.0), (0.008, (1.0, 1.0, 1.0, 0.85), 0.55)):
                    g = _add(scn, CAP, (rad, 0.01, 0), a, np.eye(3), rgba, emission=1.0)
                    if g is not None: mujoco.mjv_connector(g, CAP, rad, a, a + (b - a) * frac)


def _ry(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
