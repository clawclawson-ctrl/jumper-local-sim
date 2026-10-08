"""Pushable obstacles for the local sim's hide & seek room (map v7).

The stair steps, planters and crates become free bodies the crab can shove; the boundary walls stay fixed.
`variant()` writes a per-run copy of the map with
  - pushable=True : a freejoint on every stair step / planter, and realistic-for-a-toy-room masses and friction
                    on all of them (crates too, which the official map ships as 0.13-0.19 kg free bodies);
  - pushable=False: the map exactly as evaluated (steps/planters fixed, crates as shipped), and
  - moves         : any obstacle moved/rotated to where the person put it on the setup screen.
Sim geometry only. The crab never reads any of this; it sees obstacles through its dToF like before.
SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import hashlib, json, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# id -> (body in scene.xml, collision geom in scene.xml, label, mass kg when pushable)
OBST = {
    "stairE":   ("stair_8",               "stair_8_geom",                    "east stairs",   None),
    "stairW":   ("stair_9",               "stair_9_geom",                    "west stairs",   None),
    "planterN": ("planterN_10",           "planterN_10_geom",                "north planter", None),
    "planterS": ("planterW_11",           "planterW_11_geom",                "south planter", None),
    "crateNE":  ("prop:crate_blue",       "prop:crate_blue_geom_geom",       "NE crate",      None),
    "crateSE":  ("prop:crate_medium",     "prop:crate_medium_geom_geom",     "SE crate",      None),
    "crateSW":  ("prop:crate_small",      "prop:crate_small_geom_geom",      "SW crate",      None),
    "crateNW":  ("prop:crate_medium_nw",  "prop:crate_medium_nw_geom_geom",  "NW crate",      None),
}
# Tuned on the box with jhs/push_test.py (see README-LOCAL.md "Pushable obstacles"): heavy enough not to skate on a
# brush, light enough that the crab walking into one moves it a few cm/s.
# Push trials (crab 2.54 kg walking straight into the south planter at the brain's top push speed, ly=-0.4, 10 s):
#   1.0 kg -> 9.1 cm/s, 1.5 kg -> 5.8 cm/s, 2.0 kg -> 1.9 cm/s, 4.0 kg -> 0.25 cm/s; crate 1.0 kg -> 10.7 cm/s; stairs 1.6 kg -> 5.3 cm/s.
MASS = {"stairE": 1.8, "stairW": 1.8, "planterN": 2.0, "planterS": 1.8, "crateNE": 1.6, "crateSE": 1.8, "crateSW": 1.4, "crateNW": 1.8}
# sliding, torsional, rolling. MuJoCo uses the larger of the two geoms' values per contact, so on the floor
# (0.8 0.01 0.001) the sliding friction is 0.8 -- about what the toys get (duck 0.8, ball 1.3).
FRICTION = "0.7 0.03 0.003"
CONDIM = "4"                       # sliding + torsional (no endless spinning); boxes cannot roll
SOLREF = "0.02 1"


def geom_to_id(name: str):
    """'scn_stair_8_geom' (room json / compiled model) -> 'stairE'."""
    n = name[4:] if name.startswith("scn_") else name
    for k, v in OBST.items():
        if n == v[1]: return k
    return None


def edit_scene(xml: str, pushable=True, moves=None, masses=None, friction=FRICTION) -> str:
    root = ET.fromstring(xml)
    wb = root.find("worldbody")
    bodies = {b.get("name"): b for b in wb.iter("body")}
    masses = {**MASS, **(masses or {})}
    for k, (bname, gname, _, _) in OBST.items():
        b = bodies.get(bname)
        if b is None: continue
        g = next((x for x in b.iter("geom") if x.get("name") == gname), None)
        if pushable and g is not None:
            if b.find("freejoint") is None:
                fj = ET.Element("freejoint", {"name": bname + "_free"}); b.insert(0, fj)
            g.attrib.pop("density", None); g.set("mass", f"{float(masses[k]):.3f}")
            g.set("friction", friction); g.set("condim", CONDIM); g.set("solref", SOLREF)
        if moves and k in moves:
            x, y = float(moves[k][0]), float(moves[k][1]); yaw = float(moves[k][2]) if len(moves[k]) > 2 else None
            z = b.get("pos", "0 0 0").split()[2]
            b.set("pos", f"{x:.4f} {y:.4f} {z}")
            if yaw is not None:
                b.attrib.pop("quat", None); b.set("euler", f"0 0 {yaw:.3f}")
    return ET.tostring(root, encoding="unicode")


def variant(src, dst, pushable=True, moves=None, masses=None) -> Path:
    """Write a copy of the .map with pushable obstacles and/or moved obstacles. Other files are copied byte for byte."""
    src, dst = Path(src), Path(dst)
    with zipfile.ZipFile(src) as zi:
        xml = zi.read("scene.xml").decode("utf-8")
        new = edit_scene(xml, pushable, moves, masses).encode("utf-8")
        pkg = json.loads(zi.read("scene-package.json"))
        files = pkg.get("files") or {}
        if "scene.xml" in files: files["scene.xml"] = {"bytes": len(new), "sha256": hashlib.sha256(new).hexdigest()}
        pkg["localSim"] = {"pushableObstacles": bool(pushable), "moved": moves or {}, "masses": {**MASS, **(masses or {})} if pushable else None}
        with zipfile.ZipFile(dst, "w") as zo:
            for it in zi.infolist():
                if it.filename == "scene.xml": zo.writestr(it, new)
                elif it.filename == "scene-package.json": zo.writestr(it, json.dumps(pkg, indent=1))
                else: zo.writestr(it, zi.read(it.filename))
    return dst


# ---- current footprints (display only: the overlay draws where the props are NOW, from sim truth) ----------------
def prop_geoms(model):
    """[(geom id, obstacle id)] for the obstacles' collision boxes in a compiled model."""
    import mujoco
    out = []
    for g in range(model.ngeom):
        k = geom_to_id(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
        if k: out.append((g, k))
    return out


def footprints(model, data, gids):
    """world-xy corner lists of each obstacle's box footprint (sim truth, display only)."""
    out = []
    for g, k in gids:
        c = data.geom_xpos[g]; R = data.geom_xmat[g].reshape(3, 3); h = model.geom_size[g]
        ex, ey = R[:2, 0] * h[0], R[:2, 1] * h[1]
        out.append((k, [(c[0] + sx * ex[0] + sy * ey[0], c[1] + sx * ex[1] + sy * ey[1]) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]))
    return out


if __name__ == "__main__":
    import sys
    variant(sys.argv[1], sys.argv[2], pushable=True)
    print("wrote", sys.argv[2])
