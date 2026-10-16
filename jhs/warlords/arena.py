"""Crab Warlords arena .map: the official room v7 floor/walls (via the Crab Tanks builder, outer box pushed out of the way), then the
equilateral-triangle wall (bouncy, like the soccer walls), three castles (13 breakable bricks + a king block each), stone floor,
torch lights and team banners (no collision). SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import hashlib, json, math, os, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from . import game as G

STONE_FLOOR = "0.30 0.28 0.25 1"
WALL_RGBA = "0.46 0.43 0.38 1"
BRICK_RGBA = {"A": "0.62 0.36 0.30 1", "B": "0.36 0.42 0.60 1", "C": "0.38 0.55 0.36 1"}
KING_RGBA = {"A": "0.95 0.20 0.10 1", "B": "0.20 0.40 1.00 1", "C": "0.15 0.85 0.25 1"}


def _g(parent, name, **a): return ET.SubElement(parent, "geom", {"name": name, **{k: str(v) for k, v in a.items()}})


def build(src_map, dst_map):
    from jhs.tanks import field as TF
    keep = (TF.FX, TF.FY, TF.BALL_PARK)
    tmp = Path(dst_map).with_suffix(".tk.map")
    try:
        TF.FX, TF.FY, TF.BALL_PARK = 4.6, 5.0, (0.0, 0.0)
        TF.build(src_map, tmp, pieces=(), start={"A": G.home("A"), "B": G.home("B")}, label="Warlords")
    finally:
        TF.FX, TF.FY, TF.BALL_PARK = keep
    with zipfile.ZipFile(tmp) as zi:
        root = ET.fromstring(zi.read("scene.xml").decode()); pkg = json.loads(zi.read("scene-package.json"))
        wb = root.find("worldbody"); field = next(b for b in wb.iter("body") if b.get("name") == "tanks_arena"); field.set("name", "warlords_arena")
        ref = next(g for g in field if (g.get("name") or "").startswith("wall_side"))
        wkw = {k: ref.get(k) for k in ("type", "friction", "condim", "solref", "priority") if ref.get(k)}
        for g in list(field):                                     # the rectangular box walls are not part of this game
            if (g.get("name") or "").startswith(("wall_", "corner_", "kick_")): field.remove(g)
        for k in G.KEYS:                                          # triangle walls: the face opposite corner k
            c = -(G.R_IN + G.WALL_T / 2) * G.U[k]; yaw = math.degrees(G.ANG[k]) + 90.0
            _g(field, f"wl_wall_{k}", pos=f"{c[0]:.4f} {c[1]:.4f} {G.WALL_H / 2}", size=f"{G.SIDE / 2 + 0.25:.3f} {G.WALL_T / 2} {G.WALL_H / 2}",
               euler=f"0 0 {yaw:.3f}", rgba=WALL_RGBA, **wkw)
        bk = dict(wkw); bk["type"] = "box"
        for k in G.KEYS:
            for i, (x, y, a) in enumerate(G.bricks(k)):
                _g(field, f"wl_brick_{k}_{i}", pos=f"{x:.4f} {y:.4f} {G.BRICK_H}", size=f"{G.BRICK_L} {G.BRICK_T} {G.BRICK_H}",
                   euler=f"0 0 {math.degrees(a):.2f}", rgba=BRICK_RGBA[k], **bk)
            K = G.king(k)
            _g(field, f"wl_king_{k}", pos=f"{K[0]:.4f} {K[1]:.4f} {G.KING_H}", size=f"{G.KING_H} {G.KING_H} {G.KING_H}",
               euler=f"0 0 {math.degrees(G.ANG[k]):.2f}", rgba=KING_RGBA[k], **bk)
            # banner (team colour, behind the king near the corner) + torch: drawing only
            V = G.VERT[k] - 0.35 * G.U[k]
            _g(field, f"wl_banner_{k}", type="box", pos=f"{V[0]:.3f} {V[1]:.3f} 0.45", size="0.02 0.12 0.18", euler=f"0 0 {math.degrees(G.ANG[k]):.1f}",
               rgba=KING_RGBA[k], contype="0", conaffinity="0", group="1")
            _g(field, f"wl_pole_{k}", type="cylinder", pos=f"{V[0]:.3f} {V[1]:.3f} 0.32", size="0.012 0.32", rgba="0.25 0.2 0.15 1", contype="0", conaffinity="0", group="1")
            for sgn in (1, -1):
                side = G.U[k] @ [[0, -1], [1, 0]]
                T = G.VERT[k] - 1.7 * G.U[k] + sgn * (1.7 * math.tan(math.pi / 6) - 0.08) * side
                _g(field, f"wl_torch_{k}{'p' if sgn > 0 else 'm'}", type="sphere", pos=f"{T[0]:.3f} {T[1]:.3f} 0.30", size="0.035",
                   rgba="1.0 0.62 0.15 1", contype="0", conaffinity="0", group="1")
                ET.SubElement(wb, "light", {"name": f"wl_torchlight_{k}{'p' if sgn > 0 else 'm'}", "pos": f"{T[0]:.3f} {T[1]:.3f} 0.6", "dir": "0 0 -1",
                                            "diffuse": "0.45 0.30 0.12", "specular": "0.05 0.04 0.02", "attenuation": "1 0.1 0.25", "castshadow": "false"})
        for g in wb.iter("geom"):
            if g.get("name") == "ground": g.set("rgba", STONE_FLOOR)
        for b in wb.iter("body"):
            if b.get("name") == "prop:football":
                b.set("pos", f"0 0 {G.BALL_R + 0.003:.3f}")
                for x in b.iter("geom"):
                    if x.get("name") == "prop:football_geom_geom": x.set("friction", "0.4 0.003 0.0004")   # low rolling friction
        for m in root.find("asset"):
            if m.get("name") == "shelf_maze_appearance_floor": m.set("rgba", STONE_FLOOR)
        root.set("model", "jumper-warlords")
        xml = ET.tostring(root, encoding="unicode").encode()
        pkg["files"]["scene.xml"] = {"bytes": len(xml), "sha256": hashlib.sha256(xml).hexdigest()}
        pkg["id"] = "jumper-warlords"; pkg["title"] = "Jumper Crab Warlords arena (SIM-ONLY VISION CONCEPT)"
        pkg["description"] = (f"SIM-ONLY VISION CONCEPT. An equilateral-triangle arena ({G.SIDE:.0f} m sides, bouncy stone walls) with three castles "
                              "(13 breakable bricks around a king block) after Atari Warlords, on the official room v7 floor.")
        pkg.pop("tanks", None); pkg["warlords"] = {"side": G.SIDE, "castles": {k: dict(king=list(map(float, G.king(k))), bricks=len(G.bricks(k))) for k in G.KEYS}}
        with zipfile.ZipFile(dst_map, "w") as zo:
            for it in zi.infolist():
                if it.filename == "scene.xml": zo.writestr(it, xml)
                elif it.filename == "scene-package.json": zo.writestr(it, json.dumps(pkg, indent=1))
                elif it.filename == "README.md": zo.writestr(it, "# Jumper Crab Warlords arena (SIM-ONLY VISION CONCEPT)\n\n" + pkg["description"] + "\n\nBuilt by `jhs/warlords/arena.py`.\n")
                else: zo.writestr(it, zi.read(it.filename))
    os.remove(tmp)
    return dst_map
