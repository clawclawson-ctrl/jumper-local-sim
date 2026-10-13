"""The robot's tripod, under the dance.

The recorded dances keep the four walking legs on the floor and move the body
over them, so on a traveling figure the feet skate. Here those legs step the way
the locomotion tasks step: LF, RM and LR together, then RF, LM and RR. The clock
is the robot's own 3.125 Hz when the stride would otherwise get longer than the
leg can reach, and a slower march when the pair is barely moving. A stance foot
stays where it was put down. A swing foot lifts and lands under the body.

A bow, a wave or a wing still owns the two front claws. The four legs underneath
keep stepping. The solver is not free to flip a knee: each walking leg stays near
the pose the clip (or the standing pose, for a front foot) already solved, which
is the bend this robot actually uses. A claw that would pass through the floor
is lifted with the body, and the feet are put back down.
"""
from __future__ import annotations
import math
import numpy as np

# The tripod groups from tasks/jumper/tripod. Same leg names, same pairing.
TRIPOD_A = ("LF", "RM", "LR")
TRIPOD_B = ("RF", "LM", "RR")
SUPPORT = ("LM", "RM", "LR", "RR")
ALL = TRIPOD_A + TRIPOD_B

# Joints that place the foot. The gripper is not one of them.
_LEG_JOINTS = {
    "LF": ("LF_J0_joint", "LF_J1_joint", "LF_J2_joint", "LF_J3_joint"),
    "RF": ("RF_J0_joint", "RF_J1_joint", "RF_J2_joint", "RF_J3_joint"),
    "LM": ("LM_J0_joint", "LM_J1_joint", "LM_J2_joint"),
    "RM": ("RM_J0_joint", "RM_J1_joint", "RM_J2_joint"),
    "LR": ("LR_J0_joint", "LR_J1_joint", "LR_J2_joint"),
    "RR": ("RR_J0_joint", "RR_J1_joint", "RR_J2_joint"),
}
# The wrist is the spare joint on the front arm. It may move, but less willingly
# than the three joints that match the other legs, or it walks into the stop.
_WEIGHT = {
    "LF": (1.0, 1.0, 1.0, 0.35),
    "RF": (1.0, 1.0, 1.0, 0.35),
}
_FRONT = tuple(j for leg in ("LF", "RF") for j in (
    f"{leg}_J0_joint", f"{leg}_J1_joint", f"{leg}_J2_joint",
    f"{leg}_J3_joint", f"{leg}_J4_joint"))
_LEFT = {"LF", "LM", "LR"}

LIFT = 0.032          # metres. The gait task asks for 25 mm; a wide shot needs a little more.
LIFT_STILL = 0.040
ARC = 0.020           # an outward arc when the body itself is hardly moving
STRIDE = 0.08         # the leg's comfortable step, from the tripod task's own limit
GAIT_HZ = 3.125       # the tripod clock: 16 control steps at 50 Hz
MAX_STRIDE = 0.10
SLEW = 0.45           # rad per frame, so a claw can leave a gesture without a pop
_ITERS = 8
# How far a joint may wander from the pose that already looks like this robot.
# Wide enough for a real step. The wrist is tighter: twisting it digs a claw
# into the floor. The knee and thigh walls below stop a bend from reversing.
_BOX = 0.95
_WRIST_BOX = 0.28
_KNEE_BEND = 0.85     # rad. A walking knee stays at least this far onto its home side.
_THIGH = 0.12         # rad. A walking thigh stays at least this far onto its home side.
_STAND_Z = 0.10647    # the height the standing pose was solved at. Lower sinks the claws.


def _smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _yaw_of(q) -> float:
    qw, qx, qy, qz = (float(v) for v in q)
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


class Steps:
    """One crab's feet. `names` is the joint order `crab.joints` was built in."""

    def __init__(self, model, data, crab, names):
        import mujoco
        self.model = model
        self.crab = crab
        self.names = list(names)
        index = {n: i for i, n in enumerate(self.names)}
        prefix = self._prefix(mujoco, names[0])
        self.legs = {}
        for leg, joints in _LEG_JOINTS.items():
            cols, dofs, qadrs, ranges = [], [], [], []
            for jn in joints:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + jn)
                if jid < 0:
                    raise RuntimeError(f"missing joint {prefix}{jn}")
                cols.append(index[jn])
                dofs.append(int(model.jnt_dofadr[jid]))
                qadrs.append(int(model.jnt_qposadr[jid]))
                ranges.append((float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])))
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + leg)
            if site < 0:
                raise RuntimeError(f"missing foot site {prefix}{leg}")
            self.legs[leg] = dict(
                cols=cols, dofs=np.array(dofs), qadrs=qadrs, ranges=ranges,
                site=int(site), side=1.0 if leg in _LEFT else -1.0,
                w=np.array(_WEIGHT.get(leg, (1.0,) * len(joints))),
            )
        self.front_cols = [index[n] for n in _FRONT]
        self.grip_cols = [index[n] for n in ("LF_J4_joint", "RF_J4_joint")]
        self.q = crab.home.copy()
        self.jac = np.zeros((3, model.nv))
        self.swing = {leg: False for leg in ALL}
        self.swing_from = {}
        self.swing_to = {}
        self.vel = np.zeros(2)
        self.phase = 0.0
        self.last_xy = None
        self.last_t = None
        self.anchored = True
        self._measure(mujoco, data)
        self.floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.claw_ids = []
        self.leg_geoms = {leg: [] for leg in ALL}
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if not name.startswith(prefix) or "meshcol" not in name:
                continue
            short = name[len(prefix):]
            leg = short[:2]
            if leg in self.leg_geoms:
                self.leg_geoms[leg].append(g)
            if short[:3] in ("LF_", "RF_") and any(
                k in short for k in ("finger", "palm", "grip", "forearm")
            ):
                self.claw_ids.append(g)

    def reset(self):
        """Forget a warm-up pass. The next frame plants the feet under the body again."""
        self.q[:] = self.crab.home
        self.swing = {leg: False for leg in ALL}
        self.vel[:] = 0
        self.phase = 0.0
        self.last_xy = None
        self.last_t = None
        self.anchored = False

    def _prefix(self, mujoco, joint0: str) -> str:
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name.endswith(joint0) and int(self.model.jnt_qposadr[j]) == self.crab.joints[0]:
                return name[: -len(joint0)]
        raise RuntimeError("could not find this crab's joints on the model")

    def _measure(self, mujoco, data):
        mujoco.mj_kinematics(self.model, data)
        q = data.qpos[self.crab.free:self.crab.free + 7]
        bxy = np.array(q[:2], dtype=np.float64)
        yaw = _yaw_of(q[3:7])
        c, s = math.cos(yaw), math.sin(yaw)
        self.offset = {}
        zs = []
        for leg, spec in self.legs.items():
            p = data.site_xpos[spec["site"]]
            dx, dy = float(p[0] - bxy[0]), float(p[1] - bxy[1])
            self.offset[leg] = (c * dx + s * dy, -s * dx + c * dy)
            zs.append(float(p[2]))
        self.ground_z = float(np.mean(zs))
        self.plant = self._nominal(bxy, yaw)

    def _nominal(self, xy, yaw) -> dict:
        c, s = math.cos(yaw), math.sin(yaw)
        x, y = float(xy[0]), float(xy[1])
        out = {}
        for leg, (nx, ny) in self.offset.items():
            out[leg] = np.array([x + c * nx - s * ny, y + s * nx + c * ny])
        return out

    def _write_q(self, data):
        for adr, v, (lo, hi) in (
            (self.crab.joints[i], self.q[i], self.crab.ranges[i]) for i in range(len(self.q))
        ):
            data.qpos[adr] = float(min(hi, max(lo, v)))

    def _slew(self, col: int, target: float):
        d = float(target) - float(self.q[col])
        if d > SLEW:
            d = SLEW
        elif d < -SLEW:
            d = -SLEW
        self.q[col] += d

    def _clamp_joint(self, leg: str, k: int, q: float, q_ref: float) -> float:
        """Stay near the pose that already looks right, and keep a walking knee bent the home way.

        The recorded pose is always allowed through, so a clip that leans a little
        past the usual bend is not snapped back into a worse one.
        """
        lim_lo, lim_hi = self.legs[leg]["ranges"][k]
        q_ref = min(lim_hi, max(lim_lo, float(q_ref)))
        home = float(self.crab.home[self.legs[leg]["cols"][k]])
        box = _WRIST_BOX if (leg in ("LF", "RF") and k == 3) else _BOX
        lo = max(lim_lo, q_ref - box)
        hi = min(lim_hi, q_ref + box)
        if leg in SUPPORT and k == 1:
            # A wider swing than this splays the thigh out past the way the
            # robot stands. The step still fits inside it.
            lo = max(lo, home - 0.70)
            hi = min(hi, home + 0.70)
            if home < 0:
                hi = min(hi, -_THIGH)
            else:
                lo = max(lo, _THIGH)
        elif leg in ("LF", "RF") and k == 2:
            # Rolling the forearm through zero turns the claw into the floor.
            if home < 0:
                hi = min(hi, -0.15)
            else:
                lo = max(lo, 0.15)
        elif leg in SUPPORT and k == 2:
            # Straight enough to take the step, not folded into the stop.
            fold = min(q_ref, home)
            if home < 0:
                hi = min(hi, -_KNEE_BEND)
                lo = max(lo, fold - 0.30)
            else:
                lo = max(lo, _KNEE_BEND)
                hi = min(hi, fold + 0.30)
        if lo > hi:
            return q_ref
        return float(min(hi, max(lo, q)))

    def _reach(self, data, walked, targets):
        """Move each walked foot onto its target, without leaving the legal bend."""
        import mujoco
        for _ in range(_ITERS):
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            for leg in walked:
                spec = self.legs[leg]
                err = targets[leg] - data.site_xpos[spec["site"]]
                if float(err @ err) < 1e-8:
                    continue
                self.jac[:] = 0
                mujoco.mj_jacSite(self.model, data, self.jac, None, spec["site"])
                J = self.jac[:, spec["dofs"]] * spec["w"]
                A = J @ J.T + 1e-4 * np.eye(3)
                dq = spec["w"] * (J.T @ np.linalg.solve(A, err))
                np.clip(dq, -0.2, 0.2, out=dq)
                for k, (adr, d) in enumerate(zip(spec["qadrs"], dq)):
                    q_ref = float(spec["ref"][k])
                    data.qpos[adr] = self._clamp_joint(leg, k, float(data.qpos[adr]) + float(d), q_ref)

    def _copy_walked(self, data, walked):
        for leg in walked:
            spec = self.legs[leg]
            for col, adr in zip(spec["cols"], spec["qadrs"]):
                self.q[col] = float(data.qpos[adr])

    def _buried(self, data, leg: str) -> bool:
        """True when this leg's shell, not just its foot site, is through the floor."""
        ids = set(self.leg_geoms[leg])
        worst = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            pair = (int(c.geom1), int(c.geom2))
            if self.floor_id not in pair:
                continue
            other = pair[1] if pair[0] == self.floor_id else pair[0]
            if other in ids:
                worst = min(worst, float(c.dist))
        return worst < -0.006

    def _unwind(self, data, walked, nominal):
        """A foot driven through the floor is a bad reach. Put that leg back on
        the pose that already looks right, and stop chasing the far foothold.

        Re-solving the same target just twists the claw under again, so this
        restores the pose instead of taking another reach.
        """
        import mujoco
        if self.floor_id < 0:
            return
        mujoco.mj_forward(self.model, data)
        buried = [leg for leg in walked if self._buried(data, leg)]
        if not buried:
            return
        for leg in buried:
            spec = self.legs[leg]
            for k, adr in enumerate(spec["qadrs"]):
                q_ref = float(spec["ref"][k])
                data.qpos[adr] = self._clamp_joint(leg, k, q_ref, q_ref)
            self.swing[leg] = False
        self._copy_walked(data, buried)
        self._write_q(data)
        mujoco.mj_kinematics(self.model, data)
        for leg in buried:
            site = data.site_xpos[self.legs[leg]["site"]]
            # Keep the next step from dragging the leg back into the floor.
            here = np.array(site[:2], dtype=np.float64)
            gap = here - nominal[leg]
            far = float(np.hypot(gap[0], gap[1]))
            self.plant[leg] = nominal[leg] + gap * (0.08 / far) if far > 0.08 else here
            self.swing_to[leg] = self.plant[leg]

    def _lift_claws(self, data, walked):
        """A gesture may reach the floor. Raise the body by the overlap and put the feet back."""
        import mujoco
        if self.floor_id < 0 or not self.claw_ids:
            return
        mujoco.mj_forward(self.model, data)
        pen = None
        want = set(self.claw_ids)
        for i in range(data.ncon):
            c = data.contact[i]
            pair = (int(c.geom1), int(c.geom2))
            if self.floor_id not in pair:
                continue
            other = pair[1] if pair[0] == self.floor_id else pair[0]
            if other in want:
                dist = float(c.dist)
                pen = dist if pen is None else min(pen, dist)
        # A claw resting on the floor still reads as sunk, because the drawn
        # shell sticks a little past the contact. Leave a few millimetres.
        if pen is None or pen >= 0.004:
            return
        lift = min(0.025, 0.004 - pen)
        data.qpos[self.crab.free + 2] += lift
        self._reach(data, walked, self._targets)
        self._copy_walked(data, walked)
        self._write_q(data)

    def apply(self, data, xy, yaw, sample, t, beat, gesture: bool, phase_shift: float = 0.0):
        """Stand the body on `sample` (clip joints, sway, height, rpy) and step the feet."""
        import mujoco
        from jhs.ballroom.scene import rpy_quat
        joints, dxy, z, rpy = sample
        # A clip may dip the body a few millimetres below the standing height.
        # The feet are solved for that height, so following the dip pushes the
        # claws through the floor.
        z = max(float(z), _STAND_Z)
        yaw = float(yaw) + float(rpy[2])
        c, s = math.cos(yaw), math.sin(yaw)
        world = np.array([
            float(xy[0]) + c * float(dxy[0]) - s * float(dxy[1]),
            float(xy[1]) + s * float(dxy[0]) + c * float(dxy[1]),
        ])
        if self.last_xy is not None and self.last_t is not None:
            dt = float(t) - self.last_t
            if 1e-4 < dt < 0.5:
                inst = (world - self.last_xy) / dt
                self.vel = 0.55 * inst + 0.45 * self.vel
                speed = float(np.hypot(self.vel[0], self.vel[1]))
                # Keep the stride near STRIDE. A waltz beat is too slow for a long step:
                # the rear foot would be asked to stay down after the leg had straightened.
                freq = min(GAIT_HZ, max(1.0, speed / STRIDE))
                self.phase = (self.phase + freq * dt) % 1.0
        self.last_xy = world.copy()
        self.last_t = float(t)
        phase = (self.phase + float(phase_shift)) % 1.0

        if phase < 0.5:
            group, frac = TRIPOD_A, phase / 0.5
        else:
            group, frac = TRIPOD_B, (phase - 0.5) / 0.5
        walked = SUPPORT if gesture else ALL
        # The clip already solved a bend that matches this lean. Start there
        # (the standing pose, for a front foot that is walking) and only leave
        # it to put the foot where the step asks.
        for leg, spec in self.legs.items():
            use_clip = leg in SUPPORT
            spec["ref"] = np.array([
                float(joints[c] if use_clip else self.crab.home[c]) for c in spec["cols"]
            ])
            if leg in walked:
                # Continue from the last step, pulled back inside the legal bend.
                # Starting over from the reference each frame makes the foot chase
                # the body, which is the skate this step is here to prevent.
                for k, col in enumerate(spec["cols"]):
                    self.q[col] = self._clamp_joint(leg, k, float(self.q[col]), float(spec["ref"][k]))

        free = self.crab.free
        data.qpos[free:free + 3] = [world[0], world[1], float(z)]
        data.qpos[free + 3:free + 7] = rpy_quat(float(rpy[0]), float(rpy[1]), yaw)
        self._write_q(data)
        mujoco.mj_kinematics(self.model, data)

        nominal = self._nominal(world, yaw)
        if not self.anchored:
            self.plant = {
                leg: np.array(data.site_xpos[self.legs[leg]["site"]][:2], dtype=np.float64)
                for leg in ALL
            }
            self.anchored = True
        speed = float(np.hypot(self.vel[0], self.vel[1]))
        freq = min(GAIT_HZ, max(1.0, speed / STRIDE))
        # Land where the foot will be under the body halfway through the next stance.
        lead = self.vel * (0.75 / freq)
        if float(np.hypot(lead[0], lead[1])) > STRIDE:
            lead *= STRIDE / float(np.hypot(lead[0], lead[1]))
        lat = np.array([-math.sin(yaw), math.cos(yaw)])

        for leg in ALL:
            spec = self.legs[leg]
            if leg not in walked:
                self.swing[leg] = False
                continue
            comfort = np.array(data.site_xpos[spec["site"]][:2], dtype=np.float64)
            stepping = leg in group
            if stepping and not self.swing[leg]:
                # Land near the natural foothold. A target further out than the
                # leg can reach is what folds the knee backwards.
                aim = nominal[leg] + lead
                reach = aim - comfort
                dist = float(np.hypot(reach[0], reach[1]))
                if dist > MAX_STRIDE:
                    aim = comfort + reach * (MAX_STRIDE / dist)
                self.swing[leg] = True
                self.swing_from[leg] = np.array(self.plant[leg], dtype=np.float64)
                self.swing_to[leg] = aim
            elif not stepping and self.swing[leg]:
                self.swing[leg] = False
                self.plant[leg] = np.array(self.swing_to[leg])

        targets = {}
        for leg in walked:
            spec = self.legs[leg]
            if self.swing[leg]:
                u = _smooth(frac)
                base = (1 - u) * self.swing_from[leg] + u * self.swing_to[leg]
                span = float(np.hypot(*(self.swing_to[leg] - self.swing_from[leg])))
                lift = LIFT_STILL if span < 0.02 else LIFT
                arc = ARC * spec["side"] * math.sin(math.pi * frac) if span < 0.02 else 0.0
                targets[leg] = np.array([
                    base[0] + lat[0] * arc,
                    base[1] + lat[1] * arc,
                    self.ground_z + lift * math.sin(math.pi * min(1.0, frac)),
                ])
            else:
                p = self.plant[leg]
                targets[leg] = np.array([p[0], p[1], self.ground_z])
        self._targets = targets
        self._reach(data, walked, targets)
        self._copy_walked(data, walked)
        if gesture:
            for col in self.front_cols:
                self._slew(col, float(joints[col]))
        else:
            for col in self.grip_cols:
                self._slew(col, float(self.crab.home[col]))
        self._write_q(data)
        self._unwind(data, walked, nominal)
        if gesture:
            self._lift_claws(data, walked)

    def foot_xy(self, data, leg: str) -> np.ndarray:
        return np.array(data.site_xpos[self.legs[leg]["site"]][:2], dtype=np.float64)

    def foot_z(self, data, leg: str) -> float:
        return float(data.site_xpos[self.legs[leg]["site"]][2])


def check() -> str:
    """A stance foot stays put, a swing foot leaves the floor, and a bow still owns the claws."""
    import mujoco
    from jhs.ballroom.clips import Library
    from jhs.ballroom.scene import build
    model, data, moon, _candle = build()
    lib = Library()
    steps = Steps(model, data, moon, lib.names)
    steps.reset()
    beat = 60.0 / 98.0
    home = (moon.home, np.zeros(2), 0.10647, np.zeros(3))
    slip = 0.0
    lift = 0.0
    held = {}
    for i in range(1, 181):
        t = i / 60.0
        xy = (0.12 * t, 0.0)
        steps.apply(data, xy, 0.0, home, t, beat, gesture=False)
        mujoco.mj_kinematics(model, data)
        for leg in ALL:
            if steps.swing[leg]:
                lift = max(lift, steps.foot_z(data, leg) - steps.ground_z)
                held.pop(leg, None)
            else:
                here = steps.foot_xy(data, leg)
                if leg not in held:
                    held[leg] = here.copy()
                slip = max(slip, float(np.hypot(*(here - held[leg]))))
    if slip > 0.012:
        raise AssertionError(f"a stance foot skated {slip * 1000:.0f} mm")
    if lift < 0.018:
        raise AssertionError(f"a swing foot only rose {lift * 1000:.0f} mm")
    for leg in SUPPORT:
        col = steps.legs[leg]["cols"][2]
        bent = float(steps.q[col])
        home_bend = float(steps.crab.home[col])
        if home_bend * bent < 0 or abs(bent) < _KNEE_BEND:
            raise AssertionError(f"{leg} knee folded the wrong way ({bent:.2f})")
    bow = lib.sample("bow", 2.2, 1.0)
    before = steps.q.copy()
    steps.apply(data, (0.34, 0.0), math.pi, bow, 2.2, beat, gesture=True)
    for col in steps.front_cols:
        want = float(bow[0][col]) - float(before[col])
        moved = float(steps.q[col]) - float(before[col])
        if abs(moved) > SLEW + 1e-6:
            raise AssertionError("a claw jumped instead of following the bow")
        if want * moved < -1e-6:
            raise AssertionError("a claw stepped away from the bow")
    return f"tripod, stance slip {slip * 1000:.0f} mm, swing {lift * 1000:.0f} mm"
