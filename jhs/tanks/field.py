"""The Crab Tanks arenas: one map size (7.0 x 4.5 m, rounded bouncy walls) and three fixed Combat (1977) tank playfields.
Mirror-symmetric about both axes. Map knowledge for both brains, the referee, the setup page and the map builder.
SIM-ONLY VISION CONCEPT.

Sources for the three tank layouts (biplane clouds are not a tank playfield):
  Atari Combat manual C011402-01 (1977), AtariAge HTML transcription: Fig. G Open Field, Fig. H Easy Maze, Fig. I Complex Maze.
  Games 1-14 use only those three fields. Roger Williams' commented disassembly of Larry Wagner's ROM (combat.asm):
  VARMAP bits 3,2 select the maze (01 none, 10 simple, 00 complex) and PLFPNT points at PF0/PF1/PF2. The kernel mirrors
  the bitmap horizontally (CTRLPF reflect) and vertically (ScanLine EOR #$F8).

Arenas (id, default "simple"):
  open    — Fig. G. No interior walls. Tanks start on the left and right, facing each other.
  simple  — the maze already built here, kept as the default. It follows Fig. H (centre stubs, side brackets with
            spawn pockets, inward bars) and adds corner bars, centre L brackets and square blocks that are not in the
            ROM easy-maze bitmap (PF1_2 / PF2_2). Narrowest crab passage is 0.67 m. The corner bars sit 0.50 m under
            the top and bottom walls; that slot is below the 0.30 m leg clearance, so it is not a route.
  complex — Fig. I / PF1_0+PF2_0: centre stubs, upper and lower side bars, four L's, side brackets with outward caps,
            and a block in each centre lane. A pixel-faithful copy on this field would leave ~0.18 m gaps (one
            playfield pixel is 7.0/40 m by 4.5/23 m). Passages a crab can use are widened to >= 0.67 m: the side
            brackets are a little shorter than the pixel copy, the side bars keep 0.69 m to the outer wall and 0.72 m
            to the bracket, and the centre blocks sit 0.72 m in from the spines (the ROM gap there is 3 pixels, ~0.53 m).

Pieces are axis-aligned boxes (cx, cy, hx, hy), 20 cm tall. Every passage the 0.30 m clearance can enter is >= 0.67 m."""
from __future__ import annotations
import math, os
from pathlib import Path
from jhs.soccer.field import PROPS, TEAM, BALL_R, _box_dist, footprint  # noqa: F401  (the same official obstacles)

MIN_GAP = 0.67          # every passage the leg clearance can enter (see passage_gaps)

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
START = {"A": (-3.0, 0.0, 0.0), "B": (3.0, 0.0, 180.0)}   # left / right, facing the centre (the maze pockets, and the open-field start spots)
PARK = {k: (4.6, -2.0 + 0.9 * i) for i, k in enumerate(PROPS)}
BALL_PARK = (-4.6, 0.2)

# Complex maze (Fig. I). Centres follow the ROM bitmap; thicknesses and separations are opened so every crab passage is >= MIN_GAP.
# Widened vs the ROM: side-bracket half-height 0.68 m (pixel copy ~0.69 m of a 7-row spine, but the caps sit lower to free 0.72 m
# under the side bars); side bars 0.69 m from the outer wall; centre blocks 0.72 m clear of the spines (ROM: 3 pixels, ~0.53 m).
_COMPLEX = (
    _mirror(0.0, FY - 0.20, 0.18, 0.20, xs=False)                    # stubs from the top / bottom walls
    + _mirror(2.05, 1.48, 0.26, 0.08)                                 # upper / lower side bars
    + _mirror(0.88, 0.98, 0.24, 0.08)                                 # L horizontal bars (stem on the outer end)
    + _mirror(1.04, 0.74, 0.08, 0.16)                                 # L stems, toward the centre
    + _mirror(2.25, 0.0, 0.09, 0.68, ys=False)                        # side-bracket spines (spawn pocket outside each)
    + _mirror(2.34, 0.58, 0.15, 0.09)                                 # outward caps on the spines
    + _mirror(1.32, 0.0, 0.12, 0.12, ys=False)                        # blocks in the centre lanes
)


def _pts_simple(opener, side):
    """team-A x (positive = toward BLUE's side). The brain mirrors x and fills in the facing."""
    s = side if opener == "PEEK_NEAR" else -side
    if opener in ("PEEK_NEAR", "PEEK_FAR"): return [(2.80, 1.12 * s, None), (1.85, 1.20 * s, "fwd")]
    if opener == "HOLD": return [(2.85, 0.42 * side, "hold")]
    return [(2.80, 1.12 * side, None), (1.85, 1.20 * side, None), (0.55, 0.0, "fwd")]


def _pts_open(opener, side):
    s = side if opener == "PEEK_NEAR" else -side
    if opener in ("PEEK_NEAR", "PEEK_FAR"): return [(2.20, 0.90 * s, None), (0.40, 1.15 * s, "fwd")]
    if opener == "HOLD": return [(2.40, 0.55 * side, "hold")]
    return [(2.20, 0.70 * side, None), (0.0, 0.0, "fwd")]


def _pts_complex(opener, side):
    s = side if opener == "PEEK_NEAR" else -side
    if opener in ("PEEK_NEAR", "PEEK_FAR"): return [(2.60, 1.10 * s, None), (0.20, 0.50 * s, "fwd")]
    if opener == "HOLD": return [(2.85, 0.15 * side, "hold")]
    return [(2.60, 1.10 * side, None), (0.20, 0.50 * side, None), (0.0, 0.0, "fwd")]


ARENAS = {
    "open": {"label": "Open field", "pieces": (), "start": START, "openers": _pts_open,
             "note": "Combat manual Fig. G. No interior walls."},
    "simple": {"label": "Simple maze", "pieces": PIECES, "start": START, "openers": _pts_simple,
               "note": "The current maze (default). Fig. H plus corner bars, centre L brackets and square blocks."},
    "complex": {"label": "Complex maze", "pieces": _COMPLEX, "start": START, "openers": _pts_complex,
                "note": "Combat manual Fig. I / ROM PF1_0+PF2_0. Passages widened to >= 0.67 m."},
}
ORDER = ("open", "simple", "complex")
DEFAULT = "simple"
ARENA = os.environ.get("TK_ARENA", DEFAULT)
if ARENA not in ARENAS: ARENA = DEFAULT
PIECES = ARENAS[ARENA]["pieces"]
START = ARENAS[ARENA]["start"]


def piece_dist(x, y, pieces=None):
    d = 1e9
    for cx, cy, hx, hy in (PIECES if pieces is None else pieces):
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


def validate(setup, strict=False, pieces=None, start=None):
    E = "error" if strict else "warn"; out = []
    pcs = PIECES if pieces is None else pieces
    st = START if start is None else start
    crabs = {k: setup.get(k) or list(st[k]) for k in ("A", "B")}
    props = setup.get("props") or {}
    for k, c in crabs.items():
        if outer_clear(c[0], c[1]) < 0.22: out.append(("error", f"{TEAM[k]} crab: too close to a wall or outside the arena"))
        elif piece_dist(c[0], c[1], pcs) < 0.22: out.append(("error", f"{TEAM[k]} crab: inside / against a fixed wall"))
    if math.hypot(crabs["A"][0] - crabs["B"][0], crabs["A"][1] - crabs["B"][1]) < 0.45: out.append(("error", "the two crabs are on top of each other"))
    for k, v in props.items():
        if k not in PROPS: out.append(("error", f"unknown obstacle {k}")); continue
        x, y = float(v[0]), float(v[1]); yd = float(v[2]) if len(v) > 2 else 0.0; lab = PROPS[k][1]
        if any(outer_clear(*p) < 0.0 for p in footprint(k, x, y, yd)): out.append(("error", f"{lab}: outside the arena or in a wall")); continue
        if piece_dist(x, y, pcs) < 0.02: out.append((E, f"{lab}: on a fixed wall"))
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


def opener_points(opener, side):
    """waypoints for the active arena, or [] when that opener has no plan (the brain then searches)."""
    fn = ARENAS[ARENA]["openers"]
    return list(fn(opener, side) or [])


def set_arena(arena_id):
    """Switch the active layout and rebuild the explore / belief grids. The match process sets this via TK_ARENA at import."""
    global ARENA, PIECES, START
    if arena_id not in ARENAS: raise KeyError(arena_id)
    ARENA = arena_id
    PIECES = ARENAS[arena_id]["pieces"]
    START = ARENAS[arena_id]["start"]
    from jhs.tanks import arena as A
    A.START = START
    from jhs.tanks import explore as E
    E.refresh()
    from jhs.tanks.brain import OppBelief
    OppBelief.refresh_free()
    return ARENA


def passage_gaps(min_overlap=0.05):
    """Clear widths of slots between walls whose projections overlap. Pieces that touch (an L's two arms) are one wall.
    Returns [(description, width m)] for separations of at least 2 cm."""
    walls = [(-FX - 1.0, 0.0, 1.0, FY + 2.0), (FX + 1.0, 0.0, 1.0, FY + 2.0),
             (0.0, -FY - 1.0, FX + 2.0, 1.0), (0.0, FY + 1.0, FX + 2.0, 1.0)]
    boxes = list(PIECES) + walls
    out = []
    for i, a in enumerate(boxes):
        ax0, ax1 = a[0] - a[2], a[0] + a[2]; ay0, ay1 = a[1] - a[3], a[1] + a[3]
        for b in boxes[i + 1:]:
            bx0, bx1 = b[0] - b[2], b[0] + b[2]; by0, by1 = b[1] - b[3], b[1] + b[3]
            yov = min(ay1, by1) - max(ay0, by0); xov = min(ax1, bx1) - max(ax0, bx0)
            if yov > min_overlap and xov <= 0.0:
                gap = -xov
                if gap >= 0.02: out.append((f"horizontal slot {gap:.2f} m", gap))
            if xov > min_overlap and yov <= 0.0:
                gap = -yov
                if gap >= 0.02: out.append((f"vertical slot {gap:.2f} m", gap))
    return out


def gaps():
    """free-floor gap widths a crab needs (checked by the unit test): returns [(name, width m)]."""
    if ARENA == "open":
        return [("open width", 2 * FX), ("open height", 2 * FY)]
    if ARENA == "complex":
        return [(n, w) for n, w in passage_gaps() if w < 2.0]
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


def build(src_map, dst_map, pieces=None, start=None, label=None):
    """Write a tanks map: the soccer field builder with the arena's size, no goals / lines, then the fixed pieces."""
    import hashlib, json, zipfile, xml.etree.ElementTree as ET
    from jhs.soccer import field as SF
    pcs = tuple(PIECES if pieces is None else pieces)
    st = START if start is None else start
    label = label or ARENAS.get(ARENA, {}).get("label", "Combat")
    keep = {k: getattr(SF, k) for k in ("FX", "FY", "GW", "PARK", "KICKOFF", "KICK_DEG")}
    tmp = Path(dst_map).with_suffix(".tmp.map")
    try:
        SF.FX, SF.FY, SF.GW, SF.PARK, SF.KICKOFF = FX, FY, 0.0, PARK, {"A": st["A"], "B": st["B"]}
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
        for i, (cx, cy, hx, hy) in enumerate(pcs):
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
        pkg["description"] = ("SIM-ONLY VISION CONCEPT. A 7.0 x 4.5 m arena (rounded bouncy walls) with fixed walls laid out after an Atari "
                              f"Combat tank playfield ({label}), built from the official room v7 floor / wall material; the official crates, "
                              "planters and stair steps wait outside as optional obstacles.")
        pkg.pop("soccer", None); pkg["tanks"] = {"half": [FX, FY], "corner_radius": CORNER_R, "pieces": pcs, "start": st, "arena": label}
        with zipfile.ZipFile(dst_map, "w") as zo:
            for it in zi.infolist():
                if it.filename == "scene.xml": zo.writestr(it, xml)
                elif it.filename == "scene-package.json": zo.writestr(it, json.dumps(pkg, indent=1))
                elif it.filename == "README.md": zo.writestr(it, "# Jumper Crab Tanks arena (SIM-ONLY VISION CONCEPT)\n\n" + pkg["description"] + "\n\nBuilt by `jhs/tanks/field.py`.\n")
                else: zo.writestr(it, zi.read(it.filename))
    os.remove(tmp)
    return dst_map


def symmetric(pieces=None):
    """both-axis mirror: every box has its three reflections."""
    S = {(round(p[0], 4), round(p[1], 4), round(p[2], 4), round(p[3], 4)) for p in (PIECES if pieces is None else pieces)}
    for x, y, hx, hy in list(S):
        for sx in (1, -1):
            for sy in (1, -1):
                if (round(sx * x, 4), round(sy * y, 4), round(hx, 4), round(hy, 4)) not in S: return False
    return True


def save_png(path, arena_id=None):
    """Top-down picture of one arena (floor, fixed walls, the two start spots)."""
    from PIL import Image, ImageDraw
    aid = arena_id or ARENA
    spec = ARENAS[aid]
    W, H = 960, 680
    im = Image.new("RGB", (W, H), (24, 18, 14))
    dr = ImageDraw.Draw(im)
    pad, foot = 70, 56
    def P(x, y):
        return (pad + (x + FX) / (2 * FX) * (W - 2 * pad), pad + (FY - y) / (2 * FY) * (H - pad - foot))
    x0, y0 = P(-FX, FY); x1, y1 = P(FX, -FY)
    dr.rounded_rectangle([x0, y0, x1, y1], radius=22, fill=(107, 42, 20), outline=(180, 170, 150), width=4)
    for cx, cy, hx, hy in spec["pieces"]:
        a, b = P(cx - hx, cy + hy); c, e = P(cx + hx, cy - hy)
        dr.rectangle([min(a, c), min(b, e), max(a, c), max(b, e)], fill=(220, 160, 60))
    for k, col in (("A", (235, 60, 40)), ("B", (60, 120, 240))):
        sx, sy, hd = spec["start"][k]
        px, py = P(sx, sy); dr.ellipse([px - 9, py - 9, px + 9, py + 9], fill=col)
        dr.line([px, py, px + 22 * math.cos(math.radians(hd)), py - 22 * math.sin(math.radians(hd))], fill=(255, 255, 255), width=3)
    dr.text((pad, 18), f"{spec['label']}   7.0 x 4.5 m   Atari Combat 1977", fill=(245, 236, 220))
    dr.text((pad, H - 36), "fixed walls · red starts on the left, blue on the right · passages at least 0.67 m", fill=(210, 198, 180))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return path


def unit():
    """per arena: passages >= 0.67 m, a clear route out of each spawn, both-axis symmetry, openers or the search fallback."""
    import numpy as np
    from jhs.tanks.explore import Memory, INFL
    from jhs.tanks.brain import OPENERS, opener_waypoints, Brain, OppBelief
    from jhs.tanks import arena as A
    out = []; prev = ARENA
    try:
        for aid in ORDER:
            set_arena(aid)
            named = gaps()
            bad = [(n, round(w, 3)) for n, w in named if w < MIN_GAP]
            assert not bad, (aid, bad)
            tight = [(n, round(w, 3)) for n, w in passage_gaps() if 2 * INFL - 1e-6 <= w < MIN_GAP - 1e-9]
            assert not tight, (aid, "passage the legs can enter is under 0.67 m", tight)
            assert symmetric(), aid
            assert START["A"][0] == -START["B"][0] and START["A"][1] == START["B"][1], START
            for k, (sx, sy, _) in START.items():
                assert wall_clear(sx, sy) >= INFL, (aid, k, wall_clear(sx, sy))
            m = Memory()
            pa = m.plan(np.array(START["A"][:2]), np.array([0.0, 0.0]), False, sign=1)
            pb = m.plan(np.array(START["B"][:2]), np.array([0.0, 0.0]), False, sign=-1)
            assert pa and pb and abs(len(pa) - len(pb)) <= 1, (aid, len(pa) if pa else 0, len(pb) if pb else 0)
            assert min(wall_clear(*p) for p in pa) >= INFL - 0.06, aid
            assert all(abs(p[0] + q[0]) < 0.16 and abs(p[1] + q[1]) < 0.16 for p, q in zip(pa, pb)), aid
            if aid == "open": assert max(abs(p[1]) for p in pa) < 0.25, aid
            else: assert max(abs(p[1]) for p in pa) > 0.5, (aid, max(abs(p[1]) for p in pa))
            legs = 0; fell_back = []
            for op in OPENERS:
                for side in (-1, 1):
                    wa = opener_waypoints("A", op, side); wb = opener_waypoints("B", op, side)
                    assert [(-x, y) for x, y, _ in wa] == [(x, y) for x, y, _ in wb], (aid, op, side)
                    if not wa: fell_back.append(op); continue
                    cur = np.array(START["A"][:2])
                    for gx, gy, _f in wa:
                        assert wall_clear(gx, gy) >= INFL, (aid, op, gx, gy, wall_clear(gx, gy))
                        pth = m.plan(cur, np.array([gx, gy]), False, sign=1)
                        assert pth and min(wall_clear(*q) for q in pth) >= INFL - 0.06, (aid, op, side, gx, gy)
                        cur = np.array([gx, gy]); legs += 1
            bel = OppBelief(); bel.reset(START["B"][:2], sigma=0.25)
            assert abs(float(bel.p.sum()) - 1.0) < 1e-6, aid
            if PIECES:
                i, j = __import__("jhs.tanks.explore", fromlist=["cell"]).cell((PIECES[0][0], PIECES[0][1]))
                assert not bool(bel.FREE[i, j]), (aid, "belief treats a wall cell as floor")
                x, y, vx, vy, bounced = A.reflect(PIECES[0][0] - PIECES[0][2] - 0.02, PIECES[0][1], 0.6, 0.0)
                assert bounced and vx < 0, (aid, bounced, vx)
            else:
                x, y, vx, vy, bounced = A.reflect(FX + 0.05, 0.0, 0.6, 0.0)
                assert bounced and vx < 0, aid
            br = Brain("A"); br.reset(True, 0.0); br.wait_until = 0.0; br.t = 1.0
            br.set_opening(dict(opener="PEEK_NEAR", side=1, scan=1, bias=1, jit=(0, 0, 0)), 1.0)
            cmd, _ = br.decide((START["A"][0], START["A"][1], 0.0))
            assert br.state in ("opener", "search"), (aid, br.state)
            out.append(f"{aid}: gaps>={MIN_GAP:.2f} ({min(w for _, w in named):.2f} m), spawn route {len(pa)}/{len(pb)} cells, "
                       f"openers {legs} legs" + (f", fallback {sorted(set(fell_back))}" if fell_back else ""))
        # a waypoint inside a wall is not an opener: the brain falls back to plain search
        set_arena("simple")
        saved = ARENAS["simple"]["openers"]
        ARENAS["simple"]["openers"] = lambda opener, side: [(1.30, 0.0, None)]
        try:
            assert opener_waypoints("A", "PUSH_MID", 1) == []
        finally:
            ARENAS["simple"]["openers"] = saved
        out.append("opener inside a wall falls back to search")
    finally:
        set_arena(prev)
    return out


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    out = build(here / "maps" / "jumper-hide-seek-v7.map", here / "maps" / "jumper-tanks-arena.map")
    print("wrote", out, out.stat().st_size)
