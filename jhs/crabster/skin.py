"""Crabster look -- render only (rgba + extra scene geoms; physics never reads them). SIM-ONLY CONCEPT.
Roadster skin: gloss red shell with black trim; a rear spoiler (wing on two struts) drawn on the crab's tail."""
from __future__ import annotations
import re
import numpy as np

RED = (0.78, 0.03, 0.04, 1.0); BLACK = (0.04, 0.04, 0.05, 1.0); CHROME = (0.55, 0.56, 0.6, 1.0); GLASS = (0.06, 0.08, 0.12, 1.0)


def skin(model):
    """recolour the crab: shell / upper limbs red, display + sensors glass-black, lower legs and feet black, base trim chrome."""
    import mujoco
    n = 0
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        m = re.match(r"robot/(.*)$", name)
        if not m or model.geom_rgba[g][3] == 0: continue
        part = m.group(1)
        if re.search(r"(display_module|camera|tof_sensor)_link", part): c = GLASS
        elif re.search(r"(shell|shoulder|hip|upper_arm|base_link)", part): c = RED
        elif re.search(r"(foot|tip|pad|palm|finger)", part): c = BLACK
        else: c = CHROME if "calf" in part else BLACK
        model.geom_matid[g] = -1; model.geom_rgba[g] = c; n += 1
    for g in range(model.ngeom):                                  # a little gloss on everything red
        if tuple(np.round(model.geom_rgba[g], 3)) == tuple(np.round(RED, 3)): pass
    return n


def base_extent(model, data, base):
    """half-extent of the base body's drawn geoms in its own frame (from their aabbs)."""
    lo, hi = np.full(3, 1e9), np.full(3, -1e9)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != base: continue
        c = model.geom_aabb[g][:3]; h = model.geom_aabb[g][3:]
        R = data.geom_xmat[g].reshape(3, 3); Rb = data.xmat[base].reshape(3, 3)
        off = Rb.T @ (data.geom_xpos[g] - data.xpos[base]); Rr = Rb.T @ R
        ext = np.abs(Rr) @ h; ctr = off + Rr @ c
        lo = np.minimum(lo, ctr - ext); hi = np.maximum(hi, ctr + ext)
    return lo, hi


def spoiler(scn, data, base, lo, hi):
    """append the rear wing + struts to an mjvScene (call after update_scene). Rear = the base's -x end."""
    import mujoco
    R = data.xmat[base].reshape(3, 3); p = data.xpos[base]
    w = (hi[1] - lo[1]) * 0.42; xr = lo[0] + 0.015; zt = hi[2] + 0.045
    parts = [((xr, 0.0, zt), (0.035, w, 0.006), RED, R @ _ry(-0.25)),           # wing, a few degrees of attack
             ((xr - 0.03, 0.0, zt + 0.008), (0.006, w, 0.012), BLACK, R),         # gurney lip
             ((xr + 0.01, w * 0.6, (zt + hi[2]) / 2), (0.006, 0.006, (zt - hi[2]) / 2), BLACK, R),
             ((xr + 0.01, -w * 0.6, (zt + hi[2]) / 2), (0.006, 0.006, (zt - hi[2]) / 2), BLACK, R)]
    for side in (-1, 1):                                                          # end plates
        parts.append(((xr, side * w, zt), (0.04, 0.003, 0.02), BLACK, R))
    for pos, size, rgba, Rg in parts:
        if scn.ngeom >= scn.maxgeom: break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array(size, float), p + R @ np.array(pos, float), Rg.reshape(-1), np.array(rgba, np.float32))
        g.category = mujoco.mjtCatBit.mjCAT_DECOR; scn.ngeom += 1


def flames(scn, data, base, lift, tq, push, lo, hi, suction=0.0):
    """thruster plumes as translucent cones (capsules) at the nozzles, length ~ thrust. Visual only."""
    import mujoco
    R = data.xmat[base].reshape(3, 3); p = data.xpos[base]
    jets = []
    if lift > 0.5:
        for sx in (-1, 1): jets.append((np.array([sx * 0.05, 0, lo[2]]), np.array([0, 0, -1.0]), lift / 2))
    for ax, k in ((0, 1), (1, 0)):
        if abs(tq[ax]) > 0.05:
            sgn = np.sign(tq[ax])
            for e in (-1, 1):
                pos = np.zeros(3); pos[k] = e * (hi[k] if e > 0 else -lo[k]) * (1 if e > 0 else -1); d = np.zeros(3); d[2] = -sgn * e
                jets.append((pos, d, abs(tq[ax]) / 0.09 / 2))
    pb = R.T @ np.asarray(push, float)
    if np.linalg.norm(pb) > 0.5:
        d = -pb / np.linalg.norm(pb); jets.append((np.array([lo[0] if d[0] < 0 else hi[0], 0, 0.0]) * (1 if abs(d[0]) > 0.5 else 0), d, np.linalg.norm(pb)))
    for pos, d, f in jets:
        L = 0.03 + 0.12 * min(1.0, f / 35.0)
        if scn.ngeom + 2 >= scn.maxgeom: break
        a = p + R @ pos; b = a + R @ d * L
        for rad, rgba, frac in ((0.018, (0.55, 0.75, 1.0, 0.55), 1.0), (0.009, (1.0, 1.0, 1.0, 0.85), 0.55)):
            g = scn.geoms[scn.ngeom]; mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), np.array(rgba, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, rad, a, a + (b - a) * frac); g.category = mujoco.mjtCatBit.mjCAT_DECOR; scn.ngeom += 1


def _ry(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
