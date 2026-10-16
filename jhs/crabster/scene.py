"""Crabster stage -- SIM-ONLY THRUSTER + SUCTION CONCEPT. Builds the night-show map from the official open-field map:
a tall curved concrete arch (a half loop the crab climbs into and hangs from), a row of five small stylized cars
(blocky, no logos), stage lights and crowd silhouettes. Static geometry only; the cars and arch collide for real."""
from __future__ import annotations
import hashlib, json, math, zipfile, xml.etree.ElementTree as ET
from pathlib import Path

ARCH_X, ARCH_R, ARCH_W, ARCH_T, ARCH_N = 1.0, 1.00, 1.40, 0.06, 40   # loop starts (tangent to the floor) at x = ARCH_X
CARS_X = -0.85                                                          # the row (cars side by side along y, broadside to the crab)
SPRINT_Y = -1.6                                                         # sprint lane
START = {"ramp": (-0.2, 0.0, 180.0), "kickflip": (0.3, 0.0, 180.0), "sprint": (-2.6, SPRINT_Y, 0.0)}
#            name,       length, width, height, cabin(len, h, offset), rgba
CARS = [("Model 3", 0.36, 0.15, 0.07, (0.18, 0.05, -0.01), "0.75 0.05 0.06 1"),
        ("Model Y", 0.38, 0.16, 0.08, (0.22, 0.06, -0.02), "0.88 0.88 0.9 1"),
        ("Semi", 0.62, 0.20, 0.10, (0.16, 0.12, 0.20), "0.55 0.58 0.62 1"),
        ("Cybercab", 0.30, 0.15, 0.07, (0.20, 0.05, 0.0), "0.78 0.70 0.20 1"),
        ("Cybertruck", 0.44, 0.17, 0.09, (0.26, 0.06, -0.03), "0.70 0.72 0.75 1")]
CAR_GAP = 0.08


def arch_point(th):
    """inner surface point at loop angle th (0 = floor tangent, pi/2 = vertical wall, pi = ceiling) and inward normal."""
    return (ARCH_X + ARCH_R * math.sin(th), ARCH_R - ARCH_R * math.cos(th)), (-math.sin(th), math.cos(th))


def car_boxes():
    """(name, cx, cy, half-x, half-y, height): cars point along y, so the crab (moving along -x) jumps their width."""
    out = []; y = -(sum(c[1] for c in CARS) + CAR_GAP * 4) / 2
    for n, L, W, H, cab, rgba in CARS:
        out.append((n, CARS_X, y + L / 2, W / 2, L / 2, H + cab[1])); y += L + CAR_GAP
    return out


def _g(parent, name, **a): return ET.SubElement(parent, "geom", {"name": name, **{k: str(v) for k, v in a.items()}})


def build(src_map, dst_map):
    src_map, dst_map = Path(src_map), Path(dst_map)
    with zipfile.ZipFile(src_map) as zi:
        root = ET.fromstring(zi.read("scene.xml").decode()); pkg = json.loads(zi.read("scene-package.json"))
        wb = root.find("worldbody")
        for e in list(wb):                                    # keep only the floor: the show brings its own set
            n = e.get("name") or ""
            if e.tag == "body" and (n.startswith(("stair", "planter", "prop:", "tanks_arena", "soccer_field"))): wb.remove(e)
            if e.tag == "light": e.set("active", "true"); e.set("diffuse", "0.12 0.12 0.16")
        for g in wb.iter("geom"):
            if g.get("name") == "ground": g.set("rgba", "0.10 0.10 0.12 1"); g.attrib.pop("material", None)
        vis = root.find("visual") or ET.SubElement(root, "visual")
        for t in ("headlight", "quality"):
            for e in vis.findall(t): vis.remove(e)
        ET.SubElement(vis, "headlight", ambient="0.06 0.06 0.09", diffuse="0.15 0.15 0.2", specular="0 0 0")
        ET.SubElement(vis, "quality", shadowsize="4096")
        st = ET.SubElement(wb, "body", name="crabster_stage")
        conc = dict(rgba="0.62 0.62 0.6 1", friction="0.30 0.02 0.002", condim="3", priority="2")   # polished concrete; priority so its friction wins over the feet (MuJoCo takes the max otherwise)
        for i in range(ARCH_N):                                 # the arch: tangent slabs just outside the loop circle
            a0, a1 = math.pi * i / ARCH_N, math.pi * (i + 1) / ARCH_N; am = (a0 + a1) / 2
            (px, pz), (nx, nz) = arch_point(am); half = ARCH_R * math.tan((a1 - a0) / 2) + 0.004
            cx, cz = px - nx * ARCH_T / 2, pz - nz * ARCH_T / 2
            _g(st, f"arch_{i}", type="box", pos=f"{cx:.4f} 0 {cz:.4f}", euler=f"0 {-math.degrees(am):.3f} 0",
               size=f"{half:.4f} {ARCH_W / 2} {ARCH_T / 2}", **conc)
        for s in (-1, 1):                                        # end pillars / buttresses (drawn, collide)
            _g(st, f"arch_pillar_{s}", type="box", pos=f"{ARCH_X + ARCH_R + ARCH_T + 0.08:.3f} {s * (ARCH_W / 2 - 0.06):.3f} {ARCH_R:.3f}",
               size=f"0.08 0.06 {ARCH_R:.3f}", rgba="0.5 0.5 0.48 1")
        for i, (n, L, W, H, cab, rgba) in enumerate(CARS):       # the cars
            _, cx, cy, hx, hy, top = car_boxes()[i]
            _g(st, f"car_{i}_body", type="box", pos=f"{cx} {cy:.3f} {0.02 + H / 2:.3f}", size=f"{W / 2} {L / 2} {H / 2}", rgba=rgba)
            _g(st, f"car_{i}_cabin", type="box", pos=f"{cx} {cy + cab[2]:.3f} {0.02 + H + cab[1] / 2:.3f}", size=f"{W / 2 - 0.012} {cab[0] / 2} {cab[1] / 2}",
               rgba="0.05 0.06 0.08 1")
            for sx in (-1, 1):
                for sy in (-1, 1):
                    _g(st, f"car_{i}_wheel_{sx}{sy}", type="cylinder", pos=f"{cx + sx * W / 2:.3f} {cy + sy * (L / 2 - 0.05):.3f} 0.025", euler="0 90 0",
                       size="0.025 0.012", rgba="0.02 0.02 0.02 1", contype="0", conaffinity="0")
        for k in range(46):                                     # crowd silhouettes behind a barrier (y = +2.6 .. +3.2, drawn only)
            row = k // 23; x = -3.0 + (k % 23) * 0.27 + row * 0.13; y = 2.7 + row * 0.35; h = 0.40 + 0.06 * ((k * 7) % 5) / 4 + row * 0.15
            _g(st, f"crowd_{k}_b", type="capsule", pos=f"{x:.3f} {y:.3f} {h / 2:.3f}", size=f"0.07 {h / 2 - 0.07:.3f}", rgba="0.02 0.02 0.03 1", contype="0", conaffinity="0")
            _g(st, f"crowd_{k}_h", type="sphere", pos=f"{x:.3f} {y:.3f} {h + 0.05:.3f}", size="0.055", rgba="0.02 0.02 0.03 1", contype="0", conaffinity="0")
        _g(st, "barrier", type="box", pos="0 2.45 0.12", size="3.6 0.02 0.12", rgba="0.9 0.25 0.05 1", contype="0", conaffinity="0")
        _g(st, "launch_pad", type="cylinder", pos=f"{CARS_X - 0.75} 0 0.001", size="0.35 0.001", rgba="0.85 0.75 0.1 1", contype="0", conaffinity="0")
        _g(st, "sprint_lane", type="box", pos=f"0 {SPRINT_Y} 0.0008", size="3.2 0.18 0.0008", rgba="0.18 0.18 0.22 1", contype="0", conaffinity="0")
        for j in range(9):
            _g(st, f"sprint_tick_{j}", type="box", pos=f"{-2.6 + j * 0.5:.2f} {SPRINT_Y} 0.0016", size="0.01 0.18 0.0008", rgba="0.9 0.9 0.9 1", contype="0", conaffinity="0")
        for i, (x, y, rgb, tgt) in enumerate([(-2.5, -2.4, "1 0.85 0.6", (ARCH_X, 0, ARCH_R)), (3.0, -2.2, "0.5 0.6 1", (ARCH_X, 0, ARCH_R)),
                                              (-2.8, 2.2, "1 0.4 0.3", (CARS_X, 0, 0.3)), (2.6, 2.2, "0.9 0.9 1", (CARS_X, 0, 0.3)), (0, -3.0, "0.8 0.8 0.8", (0, 0, 0.3))]):
            d = [tgt[0] - x, tgt[1] - y, tgt[2] - 3.2]; m = math.sqrt(sum(v * v for v in d))
            ET.SubElement(wb, "light", name=f"stage_{i}", pos=f"{x} {y} 3.2", dir=" ".join(f"{v / m:.3f}" for v in d), directional="false",
                          diffuse=rgb, specular="0.3 0.3 0.3", cutoff="28", exponent="6", castshadow="true" if i < 2 else "false", attenuation="1 0 0")
            _g(st, f"light_mast_{i}", type="cylinder", pos=f"{x} {y} 1.6", size="0.03 1.6", rgba="0.15 0.15 0.17 1", contype="0", conaffinity="0")
            _g(st, f"light_can_{i}", type="sphere", pos=f"{x} {y} 3.2", size="0.09", rgba=rgb + " 1", contype="0", conaffinity="0")
        for tx in root.iter("texture"):                         # night sky: deep navy to black, with a faint horizon glow
            if tx.get("type") == "skybox": tx.set("rgb1", "0.015 0.02 0.05"); tx.set("rgb2", "0.07 0.06 0.10")
        root.set("model", "jumper-crabster")
        xml = ET.tostring(root, encoding="unicode").encode()
        pkg["files"]["scene.xml"] = {"bytes": len(xml), "sha256": hashlib.sha256(xml).hexdigest()}
        pkg["id"] = "jumper-crabster"; pkg["title"] = "Crabster stunt stage (SIM-ONLY THRUSTER + SUCTION CONCEPT)"
        pkg["description"] = "Night-show stage: concrete half-loop arch, five stylized model cars, stage lights, crowd silhouettes, sprint lane."
        for k in ("soccer", "tanks"): pkg.pop(k, None)
        pkg["crabster"] = {"arch": [ARCH_X, ARCH_R, ARCH_W], "cars": car_boxes(), "start": START}
        with zipfile.ZipFile(dst_map, "w") as zo:
            for it in zi.infolist():
                if it.filename == "scene.xml": zo.writestr(it, xml)
                elif it.filename == "scene-package.json": zo.writestr(it, json.dumps(pkg, indent=1))
                elif it.filename == "README.md": zo.writestr(it, "# Crabster stunt stage (SIM-ONLY THRUSTER + SUCTION CONCEPT)\n\n" + pkg["description"] + "\n\nBuilt by `jhs/crabster/scene.py`.\n")
                else: zo.writestr(it, zi.read(it.filename))
    return dst_map


if __name__ == "__main__":
    import sys; print(build(sys.argv[1], sys.argv[2]))


def unit():
    out = []
    (x0, z0), n0 = arch_point(0.0); (x1, z1), n1 = arch_point(math.pi)
    assert abs(x0 - ARCH_X) < 1e-9 and abs(z0) < 1e-9 and abs(z1 - 2 * ARCH_R) < 1e-9 and n1[1] < -0.99   # floor tangent -> ceiling, normal points down
    cb = car_boxes(); assert len(cb) == 5 and [c[0] for c in cb][2] == "Semi"
    for a, b in zip(cb, cb[1:]): assert a[2] + a[4] < b[2] - b[4]                                          # side by side, no overlap
    assert all(abs(c[1] - CARS_X) < 1e-9 for c in cb) and START["kickflip"][0] > CARS_X + 0.62
    out.append("stage: arch half loop (floor tangent -> inverted ceiling), five cars in a row, Semi in the middle")
    import tempfile, mujoco
    src = Path(__file__).resolve().parents[2] / "maps" / "jumper-hide-seek-v7.map"
    if src.exists():
        d = Path(tempfile.mkdtemp())
        from jhs.tanks import field as TF
        TF.build(src, d / "f.map", pieces=(), label="Crabster stage"); build(d / "f.map", d / "s.map")
        with zipfile.ZipFile(d / "s.map") as z: z.extractall(d / "x")
        m = mujoco.MjModel.from_xml_path(str(d / "x" / "scene.xml"))
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        assert sum(n.endswith("arch_%d" % i) for n in names for i in range(ARCH_N)) == ARCH_N and any("car_2_body" in n for n in names)
        assert m.nlight >= 5
        out.append(f"stage map builds and loads in MuJoCo ({m.ngeom} geoms, {m.nlight} lights, night sky)")
    return out
