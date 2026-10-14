"""The Crab Tanks arena: its own map, about 2x the soccer field (7.0 x 4.5 m, rounded bouncy walls) with a fixed wall layout
after the classic Atari Combat ("Tanks") map, mirror-symmetric about both axes. Map knowledge for both brains, the referee,
the setup page and the map builder. SIM-ONLY VISION CONCEPT.

Pieces (fixed walls, 20 cm tall, axis-aligned boxes cx, cy, hx, hy):
  stubs in from the middle of the top and bottom walls; 4 short bars near the corners; 4 L brackets framing the centre
  (opening toward it); a '[' / ']' bracket near each end (opening outward to the side wall) with a spawn pocket behind it;
  a square block just inside each bracket. Every gap a crab must use is >= 0.60 m (2 x 0.30 m leg clearance)."""
from __future__ import annotations
import math, os
from pathlib import Path
from jhs.soccer.field import PROPS, TEAM, BALL_R, _box_dist, footprint  # noqa: F401  (the same official obstacles)

FX, FY = 3.5, 2.25
CORNER_R = 0.45
GW = GD = 0.0
CENTRE_R = 0.0
WALL_H = 0.20


def _mirror(cx, cy, hx, hy, xs=True, ys=True):
    out = set()
    for sx in ((1, -1) if xs else (1,)):
        for sy in ((1, -1) if ys else (1,)): out.add((sx * cx, sy * cy, hx, hy))
    return sorted(out)


PIECES = (
    _mirror(0.0, FY - 0.20, 0.19, 0.20, xs=False)                       # stubs from the top / bottom walls (0.40 deep)
    + _mirror(2.30, 1.65, 0.28, 0.10)                                    # short bars near the corners
    + _mirror(0.83, 1.12, 0.28, 0.10) + _mirror(1.015, 0.92, 0.095, 0.20)  # L brackets: horizontal + vertical arm
    + _mirror(2.25, 0.0, 0.09, 0.82, ys=False) + _mirror(2.33, 0.72, 0.17, 0.10)   # '[' / ']': spine + arms (open outward)
    + _mirror(1.30, 0.0, 0.19, 0.20, ys=False)                           # square blocks inside the brackets
)
START = {"A": (-3.0, 0.0, 0.0), "B": (3.0, 0.0, 180.0)}   # in the pocket behind each bracket (1.0 m wide), facing the centre
PARK = {k: (4.6, -2.0 + 0.9 * i) for i, k in enumerate(PROPS)}
BALL_PARK = (-4.6, 0.2)


def piece_dist(x, y):
    d = 1e9
    for cx, cy, hx, hy in PIECES:
        d = min(d, math.hypot(max(0.0, abs(x - cx) - hx), max(0.0, abs(y - cy) - hy)))
    return d


def outer_clear(x, y):
    dx = FX - abs(x); dy = FY - abs(y)
    cx, cy = abs(x) - (FX - CORNER_R), abs(y) - (FY - CORNER_R)
    if cx > 0 and cy > 0: return min(dx, dy, CORNER_R - math.hypot(cx, cy))
    return min(dx, dy)


def wall_clear(x, y):
    """distance to the nearest fixed wall (outer rounded wall or a layout piece). Map knowledge."""
    return min(outer_clear(x, y), piece_dist(x, y))


def wall_normal(x, y, h=0.02):
    """unit vector pointing away from the nearest wall (numerical gradient of wall_clear)."""
    gx = wall_clear(x + h, y) - wall_clear(x - h, y); gy = wall_clear(x, y + h) - wall_clear(x, y - h)
    n = math.hypot(gx, gy)
    if n < 1e-9: return (-x / max(1e-6, math.hypot(x, y)), -y / max(1e-6, math.hypot(x, y)))
    return gx / n, gy / n


def validate(setup, strict=False):
    E = "error" if strict else "warn"; out = []
    crabs = {k: setup.get(k) or list(START[k]) for k in ("A", "B")}
    props = setup.get("props") or {}
    for k, c in crabs.items():
        if outer_clear(c[0], c[1]) < 0.22: out.append(("error", f"{TEAM[k]} crab: too close to a wall or outside the arena"))
        elif piece_dist(c[0], c[1]) < 0.22: out.append(("error", f"{TEAM[k]} crab: inside / against a fixed wall"))
    if math.hypot(crabs["A"][0] - crabs["B"][0], crabs["A"][1] - crabs["B"][1]) < 0.45: out.append(("error", "the two crabs are on top of each other"))
    for k, v in props.items():
        if k not in PROPS: out.append(("error", f"unknown obstacle {k}")); continue
        x, y = float(v[0]), float(v[1]); yd = float(v[2]) if len(v) > 2 else 0.0; lab = PROPS[k][1]
        if any(outer_clear(*p) < 0.0 for p in footprint(k, x, y, yd)): out.append(("error", f"{lab}: outside the arena or in a wall")); continue
        if piece_dist(x, y) < 0.02: out.append((E, f"{lab}: on a fixed wall"))
        for ck, c in crabs.items():
            if _box_dist(c, k, x, y, yd) < 0.24: out.append((E, f"{lab}: on the {TEAM[ck]} crab (the crab will start at the nearest free spot)"))
    return out


def free_spot(x, y, props, r):
    def ok(px, py): return wall_clear(px, py) >= r and all(_box_dist((px, py), k, *v) >= r for k, v in props.items() if k in PROPS)
    if ok(x, y): return (x, y)
    for ring in range(1, 40):
        d = 0.05 * ring
        for i in range(8 * ring):
            a = 2 * math.pi * i / (8 * ring); px, py = x + d * math.cos(a), y + d * math.sin(a)
            if ok(px, py): return (round(px, 3), round(py, 3))
    return (x, y)


def random_layout(seed, n=None):
    """extra movable obstacles (mirror-symmetric pairs) in free floor, on top of the fixed layout."""
    import random
    r = random.Random(int(seed) * 7907 + 3); n = n if n is not None else 2
    pairs = [("crateNE", "crateSE"), ("crateSW", "crateNW"), ("planterN", "planterS"), ("stairE", "stairW")]
    out = {}
    for _ in range(400):
        if len(out) >= n: break
        a, b = r.choice(pairs)
        if a in out: continue
        x, y = r.uniform(0.3, FX - 0.6), r.uniform(-FY + 0.5, FY - 0.5)
        if any(_box_dist((cx, cy), a, x, y, 0.0) < 0.75 for cx, cy, _, _ in [(p[0], p[1], 0, 0) for p in PIECES]) or piece_dist(x, y) < 0.9: continue
        if math.hypot(abs(x) - 3.0, y) < 1.0: continue
        out[a] = [round(x, 2), round(y, 2), 0.0]; out[b] = [round(-x, 2), round(-y, 2), 0.0]
    return out


def gaps():
    """free-floor gap widths a crab needs (checked by the unit test): returns [(name, width m)]."""
    out = []
    P = {(round(p[0], 3), round(p[1], 3)): p for p in PIECES}
    sp = P[(-2.25, 0.0)]; arm = P[(-2.33, 0.72)]; bar = P[(-2.3, -1.65)] if (-2.3, -1.65) in P else P[(-2.3, 1.65)]
    out.append(("spawn pocket (outer wall to bracket arm ends)", (FX - (abs(arm[0]) + arm[2]))))
    out.append(("bracket end to corner bar", (abs(bar[1]) - bar[3]) - (arm[1] + arm[3])))
    blk = P[(-1.3, 0.0)]
    out.append(("bracket spine to square block", (abs(sp[0]) - sp[2]) - (abs(blk[0]) + blk[2])))
    lv = P[(-1.015, 0.92)]; lh = P[(-0.83, 1.12)]
    out.append(("between the top and bottom L brackets", 2 * (lv[1] - lv[3])))
    out.append(("between the left and right L brackets", 2 * (abs(lh[0]) - lh[2])))
    st = P[(0.0, FY - 0.2)]
    out.append(("L bracket to top stub", math.hypot((abs(lh[0]) - lh[2]) - st[2], (st[1] - st[3]) - (lh[1] + lh[3]))))
    out.append(("L bracket to top wall", FY - (lh[1] + lh[3])))
    return out


def _geom(parent, name, **a):
    import xml.etree.ElementTree as ET
    return ET.SubElement(parent, "geom", {"name": name, **{k: str(v) for k, v in a.items()}})


def build(src_map, dst_map):
    """Write maps/jumper-tanks-arena.map: the soccer field builder with the arena's size, no goals / lines, then the fixed pieces."""
    import hashlib, json, zipfile, xml.etree.ElementTree as ET
    from jhs.soccer import field as SF
    keep = {k: getattr(SF, k) for k in ("FX", "FY", "GW", "PARK", "KICKOFF", "KICK_DEG")}
    tmp = Path(dst_map).with_suffix(".tmp.map")
    try:
        SF.FX, SF.FY, SF.GW, SF.PARK, SF.KICKOFF = FX, FY, 0.0, PARK, {"A": START["A"], "B": START["B"]}
        SF.build(src_map, tmp)
    finally:
        for k, v in keep.items(): setattr(SF, k, v)
    with zipfile.ZipFile(tmp) as zi:
        root = ET.fromstring(zi.read("scene.xml").decode()); pkg = json.loads(zi.read("scene-package.json"))
        wb = root.find("worldbody"); field = next(b for b in wb.iter("body") if b.get("name") == "soccer_field"); field.set("name", "tanks_arena")
        for g in list(field):
            n = g.get("name") or ""
            if n.startswith(("goal", "line_")): field.remove(g)
        ref = next(g for g in field if (g.get("name") or "").startswith("wall_side"))
        wkw = {k: ref.get(k) for k in ("type", "friction", "condim", "solref", "priority", "material") if ref.get(k)}
        for i, (cx, cy, hx, hy) in enumerate(PIECES):
            _geom(field, f"wall_tk_{i}", pos=f"{cx:.3f} {cy:.3f} {WALL_H / 2}", size=f"{hx:.3f} {hy:.3f} {WALL_H / 2}", rgba="0.86 0.62 0.24 1", **wkw)
        for g in wb.iter("geom"):
            if g.get("name") == "ground": g.set("rgba", "0.42 0.16 0.08 1")
            if (g.get("name") or "").startswith("kick_"): g.set("rgba", "0.36 0.14 0.07 1")
        for b in wb.iter("body"):
            if b.get("name") == "prop:football": b.set("pos", f"{BALL_PARK[0]} {BALL_PARK[1]} {BALL_R + 0.003:.3f}")
        for m in root.find("asset"):
            if m.get("name") == "shelf_maze_appearance_floor": m.set("rgba", "0.42 0.16 0.08 1")
        root.set("model", "jumper-tanks")
        xml = ET.tostring(root, encoding="unicode").encode()
        pkg["files"]["scene.xml"] = {"bytes": len(xml), "sha256": hashlib.sha256(xml).hexdigest()}
        pkg["id"] = "jumper-tanks"; pkg["title"] = "Jumper Crab Tanks arena (SIM-ONLY VISION CONCEPT)"
        pkg["description"] = ("SIM-ONLY VISION CONCEPT. A 7.0 x 4.5 m arena (rounded bouncy walls) with fixed walls laid out after the classic "
                              "Atari Combat map, built from the official room v7 floor / wall material; the official crates, planters and stair "
                              "steps wait outside as optional obstacles.")
        pkg.pop("soccer", None); pkg["tanks"] = {"half": [FX, FY], "corner_radius": CORNER_R, "pieces": PIECES, "start": START}
        with zipfile.ZipFile(dst_map, "w") as zo:
            for it in zi.infolist():
                if it.filename == "scene.xml": zo.writestr(it, xml)
                elif it.filename == "scene-package.json": zo.writestr(it, json.dumps(pkg, indent=1))
                elif it.filename == "README.md": zo.writestr(it, "# Jumper Crab Tanks arena (SIM-ONLY VISION CONCEPT)\n\n" + pkg["description"] + "\n\nBuilt by `jhs/tanks/field.py`.\n")
                else: zo.writestr(it, zi.read(it.filename))
    os.remove(tmp)
    return dst_map


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    out = build(here / "maps" / "jumper-hide-seek-v7.map", here / "maps" / "jumper-tanks-arena.map")
    print("wrote", out, out.stat().st_size)
