"""A moonlit ballroom and two crabs. The room is drawn; the crabs are the real model, twice.

Kinematic only: the show sets qpos and asks for poses. Nothing is simulated, so the
crabs cannot pull each other over, and they also cannot hold claws.
"""
from __future__ import annotations
import math
import re
from pathlib import Path
import numpy as np

from jhs import TOOLKIT

XML_CRAB = TOOLKIT / "assets" / "jumper" / "jumper.xml"
SEPARATION = 0.68          # centre to centre, face to face and side by side
HALF = SEPARATION / 2

# Shell, belt, limb grey. Moon is silver in moonlight; Candle is the warm one.
PALETTE = {
    "moon": dict(shell=(0.74, 0.82, 0.95, 1), belt=(0.90, 0.80, 0.46, 1), limb=(0.48, 0.54, 0.64, 1)),
    "candle": dict(shell=(0.86, 0.46, 0.16, 1), belt=(0.98, 0.82, 0.32, 1), limb=(0.55, 0.36, 0.20, 1)),
}
VISOR = (0.04, 0.05, 0.07, 1)
STOCK_RED = (0.85, 0.08, 0.05)


def _quat_z(yaw: float) -> str:
    h = yaw / 2
    return f"{math.cos(h):.5f} 0 0 {math.sin(h):.5f}"


def _room_xml(crab: Path) -> str:
    R, wall_h, col_h = 3.55, 2.15, 2.85
    geoms = []
    for i in range(8):
        a0 = i * math.pi / 4 - math.pi / 8
        a1 = a0 + math.pi / 4
        mid = (a0 + a1) / 2
        mx, my = math.cos(mid) * R, math.sin(mid) * R
        length = math.hypot(math.cos(a1) - math.cos(a0), math.sin(a1) - math.sin(a0)) * R
        # The moon sits to the south. That side of the room is glass.
        south = abs((mid + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi) < 0.55
        if south:
            geoms.append(
                f'<geom name="pane{i}" type="box" size="{length/2:.3f} 0.015 {wall_h*0.46:.3f}" '
                f'pos="{mx:.3f} {my:.3f} {wall_h*0.52:.3f}" quat="{_quat_z(mid + math.pi/2)}" '
                f'material="glass"/>'
            )
            for s, sign in (("l", -1), ("r", 1)):
                cx = mx + sign * math.cos(mid + math.pi / 2) * (length * 0.46)
                cy = my + sign * math.sin(mid + math.pi / 2) * (length * 0.46)
                geoms.append(
                    f'<geom name="curtain{i}{s}" type="box" size="0.16 0.04 {wall_h*0.5:.3f}" '
                    f'pos="{cx:.3f} {cy:.3f} {wall_h*0.5:.3f}" quat="{_quat_z(mid + math.pi/2)}" '
                    f'rgba="0.32 0.07 0.12 1" contype="0" conaffinity="0"/>'
                )
        else:
            geoms.append(
                f'<geom name="wall{i}" type="box" size="{length/2:.3f} 0.08 {wall_h/2:.3f}" '
                f'pos="{mx:.3f} {my:.3f} {wall_h/2:.3f}" quat="{_quat_z(mid + math.pi/2)}" '
                f'rgba="0.10 0.11 0.16 1"/>'
            )
        ang = i * math.pi / 4
        # Leave the south open so the moon sits in the window, not behind a pillar.
        if abs(((ang + math.pi / 2 + math.pi) % (2 * math.pi)) - math.pi) > 0.4:
            geoms.append(
                f'<geom name="column{i}" type="cylinder" size="0.11 {col_h/2:.3f}" '
                f'pos="{math.cos(ang)*3.05:.3f} {math.sin(ang)*3.05:.3f} {col_h/2:.3f}" rgba="0.16 0.17 0.22 1"/>'
            )
        geoms.append(
            f'<geom name="sconce{i}" type="sphere" size="0.045" contype="0" conaffinity="0" '
            f'pos="{math.cos(ang)*2.85:.3f} {math.sin(ang)*2.85:.3f} 1.65" material="flame"/>'
        )
    drops = []
    for k in range(6):
        a = k * math.pi / 3
        drops.append(
            f'<geom name="drop{k}" type="sphere" size="0.055" contype="0" conaffinity="0" '
            f'pos="{math.cos(a)*0.38:.3f} {math.sin(a)*0.38:.3f} 2.35" material="flame"/>'
        )
    body = "\n    ".join(geoms + drops)
    return f'''<mujoco model="moonlit_ballroom">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="1280" offheight="720" azimuth="90" elevation="-14"/>
    <headlight ambient="0.30 0.32 0.42" diffuse="0.62 0.66 0.78" specular="0.22 0.22 0.28"/>
    <rgba haze="0.02 0.025 0.06 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.01 0.015 0.05" rgb2="0.20 0.24 0.42" width="512" height="3072"/>
    <texture name="marble" type="2d" builtin="checker" rgb1="0.14 0.16 0.24" rgb2="0.24 0.27 0.40" width="256" height="256"/>
    <material name="marble" texture="marble" texrepeat="9 9" reflectance="0.22" specular="0.35"/>
    <material name="inlay" rgba="0.30 0.36 0.55 1" reflectance="0.4" specular="0.6"/>
    <material name="brass" rgba="0.72 0.56 0.24 1" reflectance="0.4" specular="0.8"/>
    <material name="flame" rgba="1 0.84 0.42 1" emission="0.7"/>
    <material name="moonmat" rgba="0.96 0.97 1 1" emission="1"/>
    <material name="glass" rgba="0.55 0.70 0.95 0.08"/>
    <model name="jumper" file="{crab}"/>
  </asset>
  <worldbody>
    <light name="moon" pos="0 -8 6" dir="0 0.55 -1" directional="true" diffuse="0.55 0.62 0.9" specular="0.35 0.4 0.55" castshadow="true"/>
    <light name="front" pos="0 4 2.4" dir="0 -1 -0.45" directional="true" diffuse="0.95 0.96 1" specular="0.3 0.3 0.35" castshadow="false"/>
    <light name="chandelier" pos="0 0 2.4" diffuse="1.0 0.78 0.45" specular="0.45 0.3 0.12"/>
    <geom name="floor" type="plane" size="16 16 0.1" material="marble"/>
    <geom name="brassring" type="cylinder" size="2.22 0.004" pos="0 0 0.001" material="brass" contype="0" conaffinity="0"/>
    <geom name="floorcircle" type="cylinder" size="2.08 0.006" pos="0 0 0.003" material="inlay" contype="0" conaffinity="0"/>
    <geom name="moon_disc" type="sphere" size="0.88" pos="0 -4.40 1.12" material="moonmat" contype="0" conaffinity="0"/>
    <geom name="hub" type="sphere" size="0.10" pos="0 0 2.55" material="flame" contype="0" conaffinity="0"/>
    {body}
    <frame pos="0.34 0 0"><attach model="jumper" prefix="moon/"/></frame>
    <frame pos="-0.34 0 0"><attach model="jumper" prefix="candle/"/></frame>
  </worldbody>
</mujoco>
'''


def _skin(model):
    import mujoco
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        m = re.match(r"(moon|candle)/(.*)$", name)
        if not m or model.geom_rgba[g][3] == 0:
            continue
        who, part = m.group(1), m.group(2)
        pal = PALETTE[who]
        if re.match(r"base_link", part):
            c = pal["belt"]
        elif re.search(r"(display_module|camera|tof_sensor)_link", part):
            c = VISOR
        elif all(abs(a - b) < 0.05 for a, b in zip(model.geom_rgba[g][:3], STOCK_RED)):
            c = pal["shell"]
        elif re.search(r"(shell|shoulder|hip|upper_arm)", part):
            c = pal["shell"]
        else:
            c = pal["limb"]
        model.geom_matid[g] = -1
        model.geom_rgba[g] = c


class Crab:
    def __init__(self, model, prefix: str, home: dict):
        import mujoco
        jn = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base")
        if jn < 0:
            raise RuntimeError(f"the {prefix} crab has no floating base")
        self.free = int(model.jnt_qposadr[jn])
        self.joints = []
        self.ranges = []
        for name in home:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
            if jid < 0:
                raise RuntimeError(f"missing joint {prefix}{name}")
            adr = int(model.jnt_qposadr[jid])
            self.joints.append(adr)
            self.ranges.append((float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])))
        self.home = np.array([home[n] for n in home], dtype=np.float64)

    def stand(self, data, xy, z, yaw):
        self.apply(data, xy, z, _yaw_quat(yaw), self.home)

    def apply(self, data, xy, z, quat, joints):
        data.qpos[self.free:self.free + 3] = [xy[0], xy[1], z]
        data.qpos[self.free + 3:self.free + 7] = quat
        for adr, (lo, hi), v in zip(self.joints, self.ranges, joints):
            data.qpos[adr] = float(np.clip(v, lo, hi))


def _yaw_quat(yaw: float) -> np.ndarray:
    h = yaw / 2
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


def rpy_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll), the order the dance files were recorded in. wxyz."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def build():
    """Return (model, data, moon, candle). The crab file is named by an absolute path."""
    import mujoco, tempfile
    if not XML_CRAB.is_file():
        raise FileNotFoundError(f"crab model missing: {XML_CRAB}")
    tmp = Path(tempfile.mkstemp(suffix=".xml")[1])
    try:
        tmp.write_text(_room_xml(XML_CRAB.resolve()))
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    _skin(model)
    if str(TOOLKIT) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(TOOLKIT))
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tasks.jumper.common.constants import HOME, STAND_Z
    data = mujoco.MjData(model)
    moon, candle = Crab(model, "moon/", HOME), Crab(model, "candle/", HOME)
    moon.stand(data, (HALF, 0.0), STAND_Z, math.pi)
    candle.stand(data, (-HALF, 0.0), STAND_Z, 0.0)
    mujoco.mj_kinematics(model, data)
    return model, data, moon, candle
