"""Crabster -- SIM-ONLY THRUSTER + SUCTION CONCEPT. The thruster model and the scripted stunt flight controller.
Thrusters are cold-gas nozzles on the crab body: LIFT (two, pointing down -> force along body +z), ROLL pair (side nozzles -> torque about
the body x = long axis), PITCH/YAW pairs (small trim nozzles), all clipped to their max thrust and fed from a finite gas tank
(cold nitrogen, Isp ~ 70 s: mass flow = thrust / (Isp g0)). The controller only commands those nozzles; MuJoCo integrates the
resulting body wrench (applied through the sim's xfrc_applied) together with gravity, contacts and the legs. Nothing is kinematic."""
from __future__ import annotations
import math
import numpy as np

G0 = 9.81
ISP = 70.0                      # s, cold gas (N2)
GAS0 = 0.60                     # kg of gas in the tank at the start of the show
DRY_ADD = 0.30                  # kg added to the crab: suction fan + tank + nozzles (gas mass on top)
LIFT_MAX = 75.0                 # N, both lift nozzles together (~2.7 g for a ~2.9 kg crab + kit)
TQ_MAX = np.array([2.0, 1.0, 1.0])    # N m about body x (roll: side pair), y (pitch trim), z (yaw trim)
SUCTION_N = 45.0                # N, the fan's pull toward the surface under it (fan only works within SUCTION_GAP)
SUCTION_GAP = 0.10              # m, base-to-surface distance up to which the fan holds


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotvec(R):
    """rotation matrix -> axis*angle (log map)."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2)); a = math.acos(c)
    if a < 1e-6: return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if a > math.pi - 1e-3:                                   # near 180 deg: axis from the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0)); i = int(np.argmax(ax)); ax = R[:, i] + np.eye(3)[i]; ax /= np.linalg.norm(ax)
        return ax * a
    return v / (2 * math.sin(a)) * a


def rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def rz(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Tank:
    def __init__(self, gas=GAS0): self.gas = gas; self.gas0 = gas; self.used_ns = 0.0
    def draw(self, thrust_total, dt):
        """take gas for this much total nozzle thrust; returns the fraction that can actually fire."""
        need = thrust_total * dt / (ISP * G0)
        if need <= 1e-12: return 1.0
        f = min(1.0, self.gas / need); self.gas -= need * f; self.used_ns += thrust_total * dt * f; return f
    @property
    def frac(self): return max(0.0, self.gas / self.gas0)


class Flight:
    """attitude (body-rate PD on the rotation error, in the body frame) + vertical-speed control with the lift nozzles."""
    def __init__(self, inertia=(0.03, 0.04, 0.05), mass=2.9):
        self.I = np.array(inertia, float); self.m = mass

    def wrench(self, R, w_body, v, R_des, w_des=np.zeros(3), vz_des=None, lift=None, kp=36.0, kd=9.0):
        """-> (lift thrust N along body +z, body torque N m). R = body->world, w_body body rates, v world linear velocity."""
        e = rotvec(R.T @ R_des)                                  # body-frame rotation error toward the target
        tq = self.I * (kp * e + kd * (np.asarray(w_des) - w_body))
        tq = np.clip(tq, -TQ_MAX, TQ_MAX)
        if lift is None and vz_des is not None:                  # vertical speed loop; only pushes when the belly faces down
            up = R[2, 2]
            a_need = G0 + 4.0 * (vz_des - v[2])
            lift = self.m * a_need / up if up > 0.5 else 0.0
        lift = float(np.clip(lift or 0.0, 0.0, LIFT_MAX))
        return lift, tq

    @staticmethod
    def nozzle_sum(lift, tq):
        """total nozzle thrust (N) for the gas bill: lift + the side pairs (a pair at 0.09 m lever makes torque/0.09 of thrust)."""
        return lift + float(np.sum(np.abs(tq))) / 0.09


def unit():
    out = []
    t = Tank(0.1); f = t.draw(ISP * G0 * 0.05 / 0.1, 0.1); assert abs(t.gas - 0.05) < 1e-9 and f == 1.0
    f = t.draw(1e6, 1.0); assert t.gas == 0.0 and 0 < f < 1
    out.append("gas tank: Isp 70 s mass flow, runs dry")
    # rigid-body check: the controller flips a free body 360 deg about x and brings it back upright, using only clipped torques
    fl = Flight(); R = np.eye(3); w = np.zeros(3); dt = 0.005; T = 0.0
    for i in range(int(2.0 / dt)):
        T = i * dt; phi = 2 * math.pi * min(1.0, max(0.0, (T - 0.1) / 0.6)); wd = np.array([2 * math.pi / 0.6 if 0.1 < T < 0.7 else 0.0, 0, 0])
        _, tq = fl.wrench(R, w, np.zeros(3), rx(phi), wd)
        w = w + tq / fl.I * dt
        a = np.linalg.norm(w) * dt
        if a > 0: k = w / np.linalg.norm(w); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); R = R @ (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K)
    assert np.linalg.norm(rotvec(R)) < 0.1 and np.linalg.norm(w) < 0.3, (rotvec(R), w)
    out.append("kickflip controller: 360 deg about the long axis, back upright, with clipped nozzle torques")
    return out


if __name__ == "__main__": print(unit())
