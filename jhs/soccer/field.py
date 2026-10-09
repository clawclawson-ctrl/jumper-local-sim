"""The soccer field: geometry constants (map knowledge both brains may use), the .map builder, setup validation and
random obstacle layouts. SIM-ONLY VISION CONCEPT.

Field x in [-FX, FX] (4.8 m long), y in [-FY, FY] (3.2 m wide), boundary walls, rounded corners (r 0.45 m), a goal at
each end (mouth 0.8 m, 0.35 m deep, 0.3 m tall posts/walls). RED (robot, "A") defends the red goal at -x and attacks
+x; BLUE (robot2, "B") defends the blue goal at +x. Built from the official hide & seek room v7 package: the same
official floor (tinted turf green), boundary-wall material, football, planters, stairs and crates. Corners are rounded (r 0.45 m) so the ball cannot stick in them."""
from __future__ import annotations
import hashlib, json, math, os, re, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

FX, FY = 2.4, 1.6                 # half length / half width of the playing area (inner wall faces)
GW, GD, GH = 0.4, 0.35, 0.30      # goal mouth half-width, depth, height
WT, WH = 0.05, 0.10               # wall half-thickness, half-height (20 cm walls keep the 13 cm ball in)
CORNER_R = 0.45                   # rounded corners: each 90-degree corner is a quarter circle of this radius (8 short wall segments)
CORNER_SEG = 8
WALL_SOLREF = os.environ.get("SC_WALL_SOLREF", "-4000 -25")   # bouncy boundary walls + corners (not the goal nets)
WALL_FRICTION = "0.3 0.005 0.0001"
KICK_DEG, KICK_W = float(os.environ.get("SC_KICK_DEG", "6")), 0.12      # kick strip at the wall foot (0 = none)
KICK_RGBA = "0.30 0.52 0.30 1"
BALL_R = 0.065
CENTRE_R = 0.45
TEAM = {"A": "RED", "B": "BLUE"}
ATTACK = {"A": +1, "B": -1}       # x-direction of the goal each crab attacks
GOAL_RGB = {"A": (0.92, 0.14, 0.10), "B": (0.14, 0.34, 0.95)}    # goal colour = colour of the team that defends it
KICKOFF = {"A": (-0.9, 0.0, 0.0), "B": (0.9, 0.0, 180.0)}        # x, y, heading deg
KICKOFF_NEAR = 0.42               # the kicking-off crab starts this far behind the ball
PROPS = {   # id -> (body, label, half-size x, y) -- the official props from room v7 (jhs.pushable.OBST names)
    "planterN": ("planterN_10", "planter (large)", 0.42, 0.35),
    "planterS": ("planterW_11", "planter (small)", 0.32, 0.42),
    "stairE":   ("stair_8", "stair step A", 0.1475, 0.565),
    "stairW":   ("stair_9", "stair step B", 0.1475, 0.565),
    "crateNE":  ("prop:crate_blue", "blue crate", 0.15, 0.15),
    "crateSE":  ("prop:crate_medium", "crate (medium)", 0.14, 0.14),
    "crateSW":  ("prop:crate_small", "crate (small)", 0.12, 0.12),
    "crateNW":  ("prop:crate_medium_nw", "crate (medium) 2", 0.14, 0.14),
}
PARK = {k: (3.6 + 0.0 * i, -2.0 + 0.9 * i) for i, k in enumerate(PROPS)}   # unused props wait outside the field


def goal_centre(team_key):
    """centre of the goal mouth (on the goal line) that crab `team_key` attacks."""
    return (ATTACK[team_key] * FX, 0.0)


def own_goal(team_key):
    return (-ATTACK[team_key] * FX, 0.0)


def in_goal(x, y):
    """+1 / -1 when the ball centre is fully over the +x / -x goal line inside a goal, else 0 (referee, sim truth)."""
    if abs(y) < GW and x > FX + BALL_R: return +1
    if abs(y) < GW and x < -FX - BALL_R: return -1
    return 0


def wall_clear(x, y):
    """distance from (x, y) to the nearest field boundary (walls + corner blocks; the goal mouths count as open)."""
    dx = FX - abs(x); dy = FY - abs(y)
    if abs(y) < GW and dx < 0.3: dx = 0.3 + max(0.0, dx)          # in front of / inside a goal mouth: no wall ahead
    cx, cy = abs(x) - (FX - CORNER_R), abs(y) - (FY - CORNER_R)
    if cx > 0 and cy > 0: return min(dx, dy, CORNER_R - math.hypot(cx, cy))      # in a rounded corner
    return min(dx, dy)


def footprint(k, x, y, yaw_deg=0.0):
    _, _, hx, hy = PROPS[k]; c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [(x + sx * hx * c - sy * hy * s, y + sx * hx * s + sy * hy * c) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]


def _box_dist(p, k, x, y, yaw_deg):
    _, _, hx, hy = PROPS[k]; c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    dx, dy = p[0] - x, p[1] - y; lx, ly = c * dx + s * dy, -s * dx + c * dy
    ex, ey = max(0.0, abs(lx) - hx), max(0.0, abs(ly) - hy)
    return math.hypot(ex, ey)


def validate(setup):
    """setup = {ball: [x, y], A: [x, y, deg], B: [x, y, deg], props: {id: [x, y, deg]}} -> list of (level, text).
    level 'error' blocks Start. Goal mouths and both kickoff spots must stay clear of obstacles."""
    out = []
    ball = setup.get("ball") or [0.0, 0.0]
    crabs = {k: setup.get(k) or list(KICKOFF[k]) for k in ("A", "B")}
    props = setup.get("props") or {}
    def inside(p, m): return abs(p[0]) <= FX - m and abs(p[1]) <= FY - m and wall_clear(p[0], p[1]) >= m
    if not inside(ball, BALL_R + 0.02): out.append(("error", "ball: too close to a wall or outside the field"))
    for k, c in crabs.items():
        if not inside(c, 0.22): out.append(("error", f"{TEAM[k]} crab: too close to a wall or outside the field"))
        if math.hypot(c[0] - ball[0], c[1] - ball[1]) < 0.25: out.append(("error", f"{TEAM[k]} crab: standing on the ball"))
    if math.hypot(crabs["A"][0] - crabs["B"][0], crabs["A"][1] - crabs["B"][1]) < 0.45: out.append(("error", "the two crabs are on top of each other"))
    keep = [("the red goal mouth", (-FX + 0.25, 0.0), 0.65), ("the blue goal mouth", (FX - 0.25, 0.0), 0.65),
            ("the centre spot", (0.0, 0.0), CENTRE_R + 0.05)] + [(f"the {TEAM[k]} kickoff spot", KICKOFF[k][:2], 0.42) for k in KICKOFF]
    for k, v in props.items():
        if k not in PROPS: out.append(("error", f"unknown obstacle {k}")); continue
        x, y = float(v[0]), float(v[1]); yd = float(v[2]) if len(v) > 2 else 0.0
        lab = PROPS[k][1]
        if any(not inside(p, 0.0) for p in footprint(k, x, y, yd)): out.append(("error", f"{lab}: outside the field or in a wall")); continue
        for name, p, r in keep:
            if _box_dist(p, k, x, y, yd) < r: out.append(("error", f"{lab}: keep {name} clear"))
        if _box_dist(ball, k, x, y, yd) < BALL_R + 0.03: out.append(("error", f"{lab}: on the ball"))
        for ck, c in crabs.items():
            if _box_dist(c, k, x, y, yd) < 0.24: out.append(("error", f"{lab}: on the {TEAM[ck]} crab"))
        for k2, v2 in props.items():
            if k2 <= k or k2 not in PROPS: continue
            if any(_box_dist(p, k2, float(v2[0]), float(v2[1]), float(v2[2]) if len(v2) > 2 else 0.0) < 0.02 for p in footprint(k, x, y, yd) + [(x, y)]):
                out.append(("error", f"{lab} overlaps {PROPS[k2][1]}"))
        if wall_clear(x, y) < 0.45 and min(PROPS[k][2:]) < 0.2: out.append(("warn", f"{lab}: close to a wall -- the ball may get stuck behind it"))
    return out


def random_layout(seed, n=None):
    """a random valid obstacle layout (mirror-symmetric so neither team is favoured)."""
    import random
    r = random.Random(int(seed) * 7907 + 3)
    n = n if n is not None else r.choice([2, 4])
    pairs = [("crateNE", "crateSE"), ("crateSW", "crateNW"), ("planterN", "planterS"), ("stairE", "stairW")]
    out = {}
    for _ in range(400):
        if len(out) >= n: break
        a, b = r.choice(pairs)
        if a in out: continue
        x, y = r.uniform(0.6, FX - 0.6), r.uniform(-FY + 0.45, FY - 0.45); yd = r.choice([0.0, 0.0, 45.0, 90.0])
        trial = {**out, a: [round(x, 2), round(y, 2), yd], b: [round(-x, 2), round(-y, 2), yd]}
        if not any(l == "error" for l, _ in validate({"ball": [0, 0], "A": list(KICKOFF["A"]), "B": list(KICKOFF["B"]), "props": trial})):
            out = trial
    return out


# ---------------------------------------------------------------------------------------------------------------------
def _geom(parent, name, **a):
    e = ET.SubElement(parent, "geom", {"name": name, **{k: str(v) for k, v in a.items()}}); return e


def build(src_map, dst_map, ball_rolling=0.0025):
    """Write the soccer field .map from the official room v7 package (assets copied byte for byte, unused ones dropped)."""
    src_map, dst_map = Path(src_map), Path(dst_map)
    with zipfile.ZipFile(src_map) as zi:
        root = ET.fromstring(zi.read("scene.xml").decode())
        pkg = json.loads(zi.read("scene-package.json"))
        wb = root.find("worldbody")
        wall_mat = None
        for b in list(wb):
            n = b.get("name") or ""
            if n.startswith(("wallX", "wallY")):
                g = b.find("geom"); wall_mat = wall_mat or (g.get("material"), g.get("rgba")); wb.remove(b)
            elif n.startswith("decor_") or n in ("prop:ball", "prop:toy_duck", "prop:toy_duck_2", "prop:mini_football"):
                wb.remove(b)
        for b in wb.iter("body"):
            n = b.get("name")
            for k, (bn, _, _, _) in PROPS.items():
                if n == bn:
                    z = b.get("pos").split()[2]; b.set("pos", f"{PARK[k][0]:.3f} {PARK[k][1]:.3f} {z}")
            if n == "prop:football":
                b.set("pos", f"0 0 {BALL_R + 0.003:.3f}")
                g = next(x for x in b.iter("geom") if x.get("name") == "prop:football_geom_geom")
                g.set("friction", f"0.65 0.005 {ball_rolling}")   # rolls, but slows down like a real ball on turf
        for g in wb.iter("geom"):
            if g.get("name") == "ground": g.set("rgba", "0.25 0.47 0.24 1")
        for m in root.find("asset"):
            if m.get("name") == "shelf_maze_appearance_floor": m.set("rgba", "0.25 0.47 0.24 1")
        field = ET.SubElement(wb, "body", {"name": "soccer_field", "pos": "0 0 0"})
        wm = {"material": wall_mat[0]} if wall_mat and wall_mat[0] else {}
        # boundary walls are "bouncy" (WALL_SOLREF: direct stiffness/damping, light damping -> the ball rebounds into play instead
        # of settling against them) and slippery; priority 1 makes the wall's contact parameters win over the ball's / crab's.
        wkw = dict(type="box", friction=WALL_FRICTION, condim="3", solref=WALL_SOLREF, priority="1", rgba="0.34 0.4 0.4 1", **wm)
        L = FX + WT * 2
        for sy in (1, -1):    # long side walls
            _geom(field, f"wall_side_{'n' if sy > 0 else 's'}", pos=f"0 {sy * (FY + WT)} {WH}", size=f"{L} {WT} {WH}", **wkw)
        seg = (FY - GW) / 2
        for sx in (1, -1):    # end walls either side of the goal mouth
            for sy in (1, -1):
                _geom(field, f"wall_end_{'e' if sx > 0 else 'w'}{'n' if sy > 0 else 's'}", pos=f"{sx * (FX + WT)} {sy * (GW + seg)} {WH}",
                      size=f"{WT} {seg + WT} {WH}", **wkw)
        for sx in (1, -1):    # rounded corners: a quarter circle of short tangent wall segments (inner face at CORNER_R)
            for sy in (1, -1):
                ox, oy = sx * (FX - CORNER_R), sy * (FY - CORNER_R); da = (math.pi / 2) / CORNER_SEG
                half = (CORNER_R + WT) * math.tan(da / 2) * 1.15
                for i in range(CORNER_SEG):
                    a = (i + 0.5) * da; nx, ny = sx * math.cos(a), sy * math.sin(a)
                    _geom(field, f"corner_{sx}{sy}_{i}", pos=f"{ox + nx * (CORNER_R + WT):.4f} {oy + ny * (CORNER_R + WT):.4f} {WH}",
                          size=f"{WT} {half:.4f} {WH}", euler=f"0 0 {math.degrees(math.atan2(ny, nx)):.2f}", **wkw)
        # kick strip: a 12 cm wide, 1.2 cm high ramp along the foot of every boundary wall / corner (6 deg), so a ball that creeps to a
        # wall rolls back into play instead of resting against it (the ball's rolling friction needs > ~2.2 deg)
        th = math.radians(KICK_DEG); sw, st = KICK_W, 0.01
        def strip(name, cx, cy, nx, ny, hl):          # (cx, cy) on the wall's inner face, (nx, ny) = unit normal into the field
            h0 = KICK_W * math.tan(th); s0 = sw / 2; zt = h0 - s0 * math.tan(th)
            px = cx + nx * s0 - nx * math.sin(th) * st; py = cy + ny * s0 - ny * math.sin(th) * st; pz = zt - math.cos(th) * st
            X = (nx * math.cos(th), ny * math.cos(th), -math.sin(th)); Y = (-ny, nx, 0.0)
            _geom(field, name, type="box", pos=f"{px:.4f} {py:.4f} {pz:.4f}", size=f"{s0 / math.cos(th):.4f} {hl:.4f} {st}",
                  xyaxes=" ".join(f"{v:.5f}" for v in (*X, *Y)), friction="0.8 0.005 0.0001", condim="3", rgba=KICK_RGBA)
        sl = (FX - CORNER_R) / 2
        if KICK_DEG <= 0: strip = lambda *a: None
        for sy in (1, -1):
            for sx in (1, -1): strip(f"kick_side_{sy}{sx}", sx * sl, sy * FY, 0, -sy, sl)
        for sx in (1, -1):
            for sy in (1, -1): strip(f"kick_end_{sx}{sy}", sx * FX, sy * (GW + (FY - CORNER_R - GW) / 2), -sx, 0, (FY - CORNER_R - GW) / 2)
        for sx in (1, -1):
            for sy in (1, -1):
                ox, oy = sx * (FX - CORNER_R), sy * (FY - CORNER_R); da = (math.pi / 2) / CORNER_SEG
                for i in range(CORNER_SEG):
                    a = (i + 0.5) * da; nx, ny = sx * math.cos(a), sy * math.sin(a)
                    strip(f"kick_corner_{sx}{sy}_{i}", ox + nx * CORNER_R, oy + ny * CORNER_R, -nx, -ny, CORNER_R * math.tan(da / 2) * 1.05)
        for key, sx in (("A", -1), ("B", 1)):   # goals: coloured posts, crossbar, side and back walls (net), floor
            r, g_, b_ = GOAL_RGB[key]; rgba = f"{r} {g_} {b_} 1"; net = f"{r * 0.8:.3f} {g_ * 0.8:.3f} {b_ * 0.8:.3f} 1"
            gk = dict(friction="0.8 0.005 0.0001", condim="3", solref="0.02 1")
            for sy in (1, -1):
                _geom(field, f"goal{key}_post_{sy}", type="cylinder", pos=f"{sx * FX} {sy * GW} {GH / 2}", size=f"0.03 {GH / 2}", rgba=rgba, **gk)
                _geom(field, f"goal{key}_side_{sy}", type="box", pos=f"{sx * (FX + GD / 2)} {sy * (GW + 0.02)} {GH / 2}", size=f"{GD / 2} 0.02 {GH / 2}", rgba=net, **gk)
            _geom(field, f"goal{key}_back", type="box", pos=f"{sx * (FX + GD + 0.02)} 0 {GH / 2}", size=f"0.02 {GW + 0.04} {GH / 2}", rgba=net, **gk)
            _geom(field, f"goal{key}_bar", type="box", pos=f"{sx * FX} 0 {GH - 0.02}", size=f"0.025 {GW} 0.02", rgba=rgba, **gk)
            _geom(field, f"goal{key}_floor", type="box", pos=f"{sx * (FX + GD / 2)} 0 0.001", size=f"{GD / 2} {GW} 0.001", rgba=net,
                  contype="0", conaffinity="0", group="2")
        lk = dict(type="box", rgba="0.93 0.90 0.72 1", contype="0", conaffinity="0", group="2")   # cream lines (not ball-white)
        _geom(field, "line_centre", pos="0 0 0.0015", size=f"0.012 {FY} 0.0015", **lk)
        for i in range(32):
            a = 2 * math.pi * i / 32
            _geom(field, f"line_circle_{i}", pos=f"{CENTRE_R * math.cos(a):.4f} {CENTRE_R * math.sin(a):.4f} 0.0015",
                  size=f"0.012 {CENTRE_R * math.pi / 32 + 0.004:.4f} 0.0015", euler=f"0 0 {math.degrees(a):.2f}", **lk)
        _geom(field, "line_spot", pos="0 0 0.0015", size="0.03 0.03 0.0015", **lk)
        for sx in (1, -1):   # goal areas 0.45 deep x 1.3 wide
            _geom(field, f"line_box_{sx}", pos=f"{sx * (FX - 0.45)} 0 0.0015", size="0.012 0.65 0.0015", **lk)
            for sy in (1, -1):
                _geom(field, f"line_boxs_{sx}{sy}", pos=f"{sx * (FX - 0.225)} {sy * 0.65} 0.0015", size="0.225 0.012 0.0015", **lk)
        # drop assets nothing references any more
        used = set(re.findall(r'(?:mesh|material|texture)="([^"]+)"', ET.tostring(root, encoding="unicode")))
        asset = root.find("asset")
        for _ in range(2):
            for a in list(asset):
                if a.tag in ("mesh", "material") and a.get("name") not in used: asset.remove(a)
            used = set(re.findall(r'(?:mesh|material|texture)="([^"]+)"', ET.tostring(root, encoding="unicode")))
        files_used = {"assets/" + a.get("file") for a in asset if a.get("file")}
        root.set("model", "jumper-soccer")
        xml = ET.tostring(root, encoding="unicode").encode()
        files = pkg.get("files") or {}
        newfiles = {}
        for k, v in files.items():
            if k.startswith("assets/") and k not in files_used: continue
            newfiles[k] = v
        newfiles["scene.xml"] = {"bytes": len(xml), "sha256": hashlib.sha256(xml).hexdigest()}
        pkg["files"] = newfiles
        pkg["id"] = "jumper-soccer"; pkg["title"] = "Jumper Crab Soccer field (SIM-ONLY VISION CONCEPT)"
        pkg["description"] = ("SIM-ONLY VISION CONCEPT. A 4.8 x 3.2 m walled soccer field for two crabs, built from the official room v7 "
                              "pieces: the Shelf-maze floor (tinted turf green) and boundary-wall material, the Home football, and the Plaza "
                              "planters, Bedroom stair steps and Warehouse crates as optional obstacles (parked outside the field until placed). "
                              "Goals (mouth 0.8 m) are simple coloured walls: red at -x, blue at +x. The four corners are rounded (quarter circles of "
                              "radius 0.45 m made of short wall segments) so the ball glides round them. Lines are non-colliding floor decals.")
        pkg["props"] = [p for p in pkg.get("props", []) if p.get("body") in {v[0] for v in PROPS.values()} | {"prop:football"}]
        pkg["spawn"] = {"bounds": [-FX, -FY, 0, FX, FY, 2], "clearance": 0.3, "position": [KICKOFF["A"][0], 0.0, 0.0], "site": "spawn", "yaw": 0.0}
        pkg["soccer"] = {"field_half": [FX, FY], "corner_radius": CORNER_R, "goal": {"half_width": GW, "depth": GD, "height": GH}, "ball_rolling_friction": ball_rolling,
                         "teams": {"RED": "robot, defends -x", "BLUE": "robot2, defends +x"}}
        for s in wb.iter("site"):
            if s.get("name") == "spawn": s.set("pos", f"{KICKOFF['A'][0]} 0 0")
        readme = ("# Jumper Crab Soccer field (SIM-ONLY VISION CONCEPT)\n\n" + pkg["description"] +
                  "\n\nBuilt by `jhs/soccer/field.py` from `jumper-hide-seek-v7.map` (see that map's README for the provenance of the "
                  "official pieces). The ball's rolling friction is raised to %.4f so it slows down like a ball on turf.\n" % ball_rolling)
        with zipfile.ZipFile(dst_map, "w") as zo:
            for it in zi.infolist():
                f = it.filename
                if f == "scene.xml": zo.writestr(it, xml)
                elif f == "scene-package.json": zo.writestr(it, json.dumps(pkg, indent=1))
                elif f == "README.md": zo.writestr(it, readme)
                elif f.startswith("assets/") and f not in files_used: continue
                elif f.startswith("preview/"): continue
                else: zo.writestr(it, zi.read(f))
    return dst_map


if __name__ == "__main__":
    import sys
    here = Path(__file__).resolve().parents[2]
    out = build(here / "maps" / "jumper-hide-seek-v7.map", here / "maps" / "jumper-soccer-field.map", *(float(a) for a in sys.argv[1:2]))
    print("wrote", out, out.stat().st_size)
