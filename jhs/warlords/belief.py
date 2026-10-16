"""BALL BELIEF ("knows where it isn't"), one per crab, like Crab Tanks' opponent belief -- but for the ball.
Particles carry (position, velocity) and roll with the same 2D ball model the brain predicts with (rolling deceleration +
bounces off the angled walls); a histogram of them on a CELL grid is the belief grid the brain and the minimap use.
- sighting: particles reset around the seen ball with its estimated velocity (+ noise);
- every tick: particles roll on (so the peak follows the predicted bounce path), and a little probability regrows over
  floor I cannot see (a crab may have kicked the ball there);
- floor my camera sees WITHOUT the ball is cleared (those particles lose almost all their weight; resampled).
Uses only my own camera's field of view and my pose (no sim truth)."""
from __future__ import annotations
import math
import numpy as np
from . import game as G

CELL = 0.25
FOV, RANGE = 0.75, 4.5            # rad half-angle, m: the floor my camera reliably sees a ball on (as the perception uses)
N = 600
VSIG = 0.25                      # m/s: velocity noise at a sighting / regrowth (a kick changes the ball's speed)
REGROW = 0.02                    # share of particles respawned per tick over unseen floor
X0, Y0 = -G.R_OUT, -G.R_IN
NX, NY = int(math.ceil(2 * G.R_OUT / CELL)), int(math.ceil((G.R_OUT + G.R_IN) / CELL))
UK = np.array([G.U[k] for k in G.KEYS])


def _inside(P, m=G.BALL_R): return G.field_cn(P)[0] > m


def visible(P, pose):
    x, y, yaw = pose; d = P - np.array([x, y]); r = np.hypot(d[:, 0], d[:, 1])
    a = np.arctan2(d[:, 1], d[:, 0]) - yaw; a = (a + math.pi) % (2 * math.pi) - math.pi
    return (r < RANGE) & (np.abs(a) < FOV) & (r > 0.25)


class BallBelief:
    def __init__(self, rng, n=N):
        self.rng = rng; self.n = n; self.P = np.zeros((n, 2)); self.V = np.zeros((n, 2)); self.W = np.ones(n) / n
        self.uniform(None)

    def _rand_floor(self, m, pose=None):
        out = np.zeros((0, 2))
        while len(out) < m:
            Q = self.rng.uniform([X0, Y0], [X0 + NX * CELL, Y0 + NY * CELL], (4 * m, 2)); Q = Q[_inside(Q, 0.1)]
            if pose is not None: Q = Q[~visible(Q, pose)]
            out = np.vstack([out, Q])
        return out[:m]

    def uniform(self, pose):
        self.P = self._rand_floor(self.n, pose); self.V = self.rng.normal(0, VSIG, (self.n, 2)); self.W[:] = 1 / self.n

    def at(self, p, v=(0, 0), sig=0.05):
        """a sighting / the referee's public kickoff call: the ball is here, moving at v."""
        self.P = np.asarray(p, float) + self.rng.normal(0, sig, (self.n, 2))
        self.V = np.asarray(v, float) + self.rng.normal(0, VSIG * (0.3 if np.linalg.norm(v) > 0 else 0.1), (self.n, 2)); self.W[:] = 1 / self.n

    def step(self, dt, pose=None):
        """roll on (decel + angled-wall bounces), regrow over unseen floor, clear what I can see."""
        self.V += G.kick_acc(self.P) * dt                      # kick strips at the wall foot
        sp = np.hypot(self.V[:, 0], self.V[:, 1]); f = np.maximum(0.0, sp - G.BALL_DECEL * dt) / np.maximum(sp, 1e-9)
        self.V *= f[:, None]; Q = self.P + self.V * dt
        c, n = G.field_cn(Q)                                   # walls, rounded corners, fillets
        hit = (c < G.BALL_R) & ((self.V * n).sum(1) < 0)
        self.V[hit] -= (1 + G.WALL_E_OUT) * (self.V[hit] * n[hit]).sum(1)[:, None] * n[hit]
        Q = np.where(_inside(Q)[:, None], Q, self.P + self.V * dt); Q = np.where(_inside(Q, 0.0)[:, None], Q, self.P)
        self.P = Q
        m = self.rng.random(self.n) < REGROW
        if m.any(): self.P[m] = self._rand_floor(int(m.sum()), pose); self.V[m] = self.rng.normal(0, VSIG, (int(m.sum()), 2)); self.W[m] = self.W.mean()
        if pose is not None: self.clear(pose)

    def clear(self, pose):
        vis = visible(self.P, pose)
        if not vis.any(): return
        self.W[vis] *= 0.02; s = self.W.sum()
        if s < 1e-12: self.uniform(pose); return
        self.W /= s
        idx = self.rng.choice(self.n, self.n, p=self.W)                 # resample (keep the particle count where the ball can be)
        self.P, self.V = self.P[idx].copy(), self.V[idx].copy(); self.W[:] = 1 / self.n
        j = self.rng.normal(0, 0.03, self.P.shape); self.P = np.where(_inside(self.P + j)[:, None], self.P + j, self.P)

    def grid(self):
        i = np.clip(((self.P[:, 0] - X0) / CELL).astype(int), 0, NX - 1); j = np.clip(((self.P[:, 1] - Y0) / CELL).astype(int), 0, NY - 1)
        g = np.zeros((NX, NY)); np.add.at(g, (i, j), self.W); return g

    def peak(self):
        g = self.grid(); i, j = np.unravel_index(int(np.argmax(g)), g.shape)
        return np.array([X0 + (i + 0.5) * CELL, Y0 + (j + 0.5) * CELL]), float(g[i, j])

    def mass(self, pose): return float(self.W[visible(self.P, pose)].sum())


def unit():
    out = []; rng = np.random.default_rng(3)
    b = BallBelief(rng); pose = (0.0, 0.0, 0.0)
    b.step(0.1, pose); assert b.mass(pose) < 0.05, "seen floor not cleared"
    out.append("ball belief: floor my camera sees without the ball is cleared")
    b.at((-1.5, 0.0), (1.0, 0.0)); look = (-1.5, 2.5, math.pi)               # looking away (west, from up north)
    for _ in range(15): b.step(0.1, look)
    pk, _ = b.peak(); assert abs(pk[0] - 0.0) < 0.4 and abs(pk[1]) < 0.4, pk
    out.append(f"peak follows the predicted roll ({pk.round(2).tolist()} after 1.5 s at 1 m/s)")
    b.at((0.0, -1.5), (0.0, -1.0))                                          # rolls into the bottom wall (y = -R_IN) and bounces back
    for _ in range(30): b.step(0.1, None)
    pk, _ = b.peak(); assert pk[1] > -G.R_IN + 0.1 and np.mean(b.V[:, 1]) > 0, (pk, np.mean(b.V, 0))
    out.append("peak follows the bounce off the angled wall")
    pks = []
    for k in G.KEYS:                                                        # 3-way rotation symmetry
        R = G.rot(k); bb = BallBelief(np.random.default_rng(7)); bb.at(R @ np.array([-1.0, -0.5]), R @ np.array([0.8, 0.2]))
        for _ in range(20): bb.step(0.1, None)
        pks.append(R.T @ bb.peak()[0])
    assert np.allclose(pks[0], pks[1], atol=0.3) and np.allclose(pks[0], pks[2], atol=0.3), pks
    out.append("belief rotation-symmetric (RED/BLUE/GREEN)")
    return out


if __name__ == "__main__": print(unit())
