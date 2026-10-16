"""Crab Warlords arena .map: the official room v7 floor/walls (via the Crab Tanks builder, outer box pushed out of the way), then the
equilateral-triangle wall (bouncy, like the soccer walls), three castles (13 breakable bricks + a king block each), stone floor,
torch lights and team banners (no collision). SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import hashlib, json, math, os, zipfile
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from . import game as G

STONE_FLOOR = "0.30 0.28 0.25 1"
WALL_RGBA = "0.34 0.27 0.20 1"
KICK_RGBA = "0.30 0.24 0.18 1"     # kick strips: a shade darker than the stone     # warm dark stone: saturated + dim enough that the white-ball detector never merges the ball with a wall behind it
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
               euler=f"0 0 {yaw:.3f}", rgba=WALL_RGBA, **{**wkw, "solref": G.WALL_SOLREF_OUT})
        # ---- ball-proof walls (ported from the soccer field): rounded corners, junction fillets, kick strips ----
        sw_ = {**wkw, "solref": G.WALL_SOLREF_OUT, "type": "box"}
        for k in G.KEYS:                                          # rounded corner: 8 tangent stone segments (inner face at CORNER_R)
            C = G.CORNER_C[k]; a0 = G.ANG[k]; n_ = 8; span = math.radians(120); da = span / n_
            half = (G.CORNER_R + G.WALL_T / 2) * math.tan(da / 2) * 1.15
            for i in range(n_):
                a = a0 - span / 2 + (i + 0.5) * da; nx, ny = math.cos(a), math.sin(a); r = G.CORNER_R + G.WALL_T / 2
                _g(field, f"wl_corner_{k}_{i}", pos=f"{C[0] + nx * r:.4f} {C[1] + ny * r:.4f} {G.WALL_H / 2}", size=f"{G.WALL_T / 2} {half:.4f} {G.WALL_H / 2}",
                   euler=f"0 0 {math.degrees(a):.2f}", rgba=WALL_RGBA, **sw_)
        for (k, j, A, B) in G.FILLETS:                            # junction fillets: stone deflectors sealing the wall/castle wedge
            AB = B - A; L = float(np.linalg.norm(AB)); nf = np.array([-AB[1], AB[0]]) / L; nf = nf if (-A) @ nf > 0 else -nf
            M = (A + B) / 2 - nf * G.WALL_T / 2
            _g(field, f"wl_fillet_{k}{j}", pos=f"{M[0]:.4f} {M[1]:.4f} {G.WALL_H / 2}", size=f"{L / 2 + 0.03:.4f} {G.WALL_T / 2} {G.WALL_H / 2}",
               euler=f"0 0 {math.degrees(math.atan2(AB[1], AB[0])):.2f}", rgba=WALL_RGBA, **sw_)
        th = math.radians(G.KICK_DEG); st = 0.01
        def strip(name, c, n, hl):                                # soccer kick strip: (c) on the wall's inner face, (n) unit normal into play
            h0 = G.KICK_W * math.tan(th); s0 = G.KICK_W / 2; zt = h0 - s0 * math.tan(th)
            P_ = c + n * s0 - n * math.sin(th) * st; pz = zt - math.cos(th) * st
            X = (n[0] * math.cos(th), n[1] * math.cos(th), -math.sin(th)); Y = (-n[1], n[0], 0.0)
            _g(field, name, type="box", pos=f"{P_[0]:.4f} {P_[1]:.4f} {pz:.4f}", size=f"{s0 / math.cos(th):.4f} {hl:.4f} {st}",
               xyaxes=" ".join(f"{v:.5f}" for v in (*X, *Y)), friction="0.8 0.005 0.0001", condim="3", rgba=KICK_RGBA)
        for j in G.KEYS: strip(f"wl_kick_wall_{j}", -G.R_IN * G.U[j], G.U[j], G.SIDE / 2 - 1.0)
        for (k, j, A, B) in G.FILLETS:
            AB = B - A; L = float(np.linalg.norm(AB)); nf = np.array([-AB[1], AB[0]]) / L; nf = nf if (-A) @ nf > 0 else -nf
            strip(f"wl_kick_fillet_{k}{j}", (A + B) / 2, nf, L / 2)
        bk = dict(wkw); bk["type"] = "box"; bk["solref"] = G.WALL_SOLREF   # bouncy, stiff stone (fast ball: restitution ~0.9)
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
                                            "diffuse": "0.22 0.15 0.07", "specular": "0.05 0.04 0.02", "attenuation": "1 0.1 0.25", "castshadow": "false"})
        for bd in list(wb):                                       # the room's static stair / planter blocks are not part of this game
            if bd.tag == "body" and (bd.get("name") or "").startswith(("stair_", "planter")): wb.remove(bd)
        for g in wb.iter("geom"):
            if g.get("name") == "ground": g.set("rgba", STONE_FLOOR)
        for b in wb.iter("body"):
            if b.get("name") == "prop:football":
                b.set("pos", f"0 0 {G.BALL_R + 0.003:.3f}")
                for x in b.iter("geom"):
                    if x.get("name") == "prop:football_geom_geom": x.set("friction", "0.4 0.003 0.0002")   # low rolling friction
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
