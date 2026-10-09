"""Soccer brain memory + exploration (SIM-ONLY VISION CONCEPT), after the hide & seek crab (tools/hide_seek_next2.py):
a searched-floor grid filled from this crab's own camera coverage, an obstacle map from its own dToF, next-best-view
selection that prefers unseen floor behind obstacles, and A* that treats sensed PUSHABLE obstacles as costly (not blocked)
when the setup says obstacles are pushable (prior knowledge: every obstacle on this field is an official prop -- crate,
planter, stair step -- and all of them are free bodies when 'Pushable obstacles' is on). Walls and goals stay blocked."""
from __future__ import annotations
import heapq, math
import numpy as np
from .field import FX, FY, wall_clear

CELL = 0.1
NX, NY = int(round(2 * FX / CELL)), int(round(2 * FY / CELL))
XS = -FX + CELL * (np.arange(NX) + 0.5); YS = -FY + CELL * (np.arange(NY) + 0.5)
CX, CY = np.meshgrid(XS, YS, indexing="ij")                    # cell centres [NX, NY]
INSIDE = np.vectorize(wall_clear)(CX, CY) >= 0.0
WALLBLOCK = np.vectorize(wall_clear)(CX, CY) < 0.2             # map knowledge: too close to a wall / goal frame for a crab
HFOV, VIEW_R, SEEN_DECAY, OBST_FORGET = 1.0, 2.6, 25.0, 30.0   # camera half-angle 0.5 rad; floor counts unseen again after 25 s
PUSH_COST = 6.0                                                # extra cost per cell of a sensed pushable obstacle


def cell(xy):
    return (int(np.clip((xy[0] + FX) / CELL, 0, NX - 1)), int(np.clip((xy[1] + FY) / CELL, 0, NY - 1)))


class Memory:
    def __init__(self):
        self.seen = np.full((NX, NY), -1e9); self.obst = np.full((NX, NY), -1e9); self.t = 0.0

    def occ(self): return (self.t - self.obst) < OBST_FORGET

    def add_obstacles(self, pts, ball=None, opp=None):
        for p in pts[::2]:
            if ball is not None and np.hypot(p[0] - ball[0], p[1] - ball[1]) < 0.16: continue
            if opp is not None and np.hypot(p[0] - opp[0], p[1] - opp[1]) < 0.35: continue    # the other crab is not a prop
            if wall_clear(p[0], p[1]) < 0.06: continue                  # that is the wall itself (map knowledge)
            i, j = cell(p); self.obst[max(0, i - 1):i + 2, max(0, j - 1):j + 2] = self.t

    def _los(self, ox, oy, tx, ty):
        """bool array: cells (tx, ty) visible from (ox, oy) without crossing a sensed obstacle cell."""
        occ = self.occ(); d = np.hypot(tx - ox, ty - oy); ok = np.ones(tx.shape, bool)
        for k in range(1, int(VIEW_R / CELL) + 1):
            s = k * CELL; m = s < d - 0.12
            if not m.any(): break
            px = ox + (tx - ox) / np.maximum(d, 1e-6) * s; py = oy + (ty - oy) / np.maximum(d, 1e-6) * s
            ii = np.clip(((px + FX) / CELL).astype(int), 0, NX - 1); jj = np.clip(((py + FY) / CELL).astype(int), 0, NY - 1)
            ok &= ~(m & occ[ii, jj])
        return ok

    def mark_view(self, pose):
        x, y, yaw = pose
        d = np.hypot(CX - x, CY - y); b = np.abs((np.arctan2(CY - y, CX - x) - yaw + math.pi) % (2 * math.pi) - math.pi)
        cand = (d < VIEW_R) & (b < HFOV / 2) & (d > 0.15) & INSIDE
        if cand.any():
            vis = self._los(x, y, CX[cand], CY[cand]); idx = np.argwhere(cand)[vis]; self.seen[idx[:, 0], idx[:, 1]] = self.t

    def unseen(self): return INSIDE & ((self.t - self.seen) > SEEN_DECAY) & ~WALLBLOCK

    def next_view(self, me, plan_len=None, sign=1):
        """next-best-view: lattice viewpoints scored by the unseen floor they would show (360 deg scan on arrival, 1.6 m),
        unseen cells next to obstacles count double (hidden corners behind props), minus a travel cost."""
        un = self.unseen()
        if not un.any(): return None, 0.0
        occ = self.occ()
        near_obst = np.zeros_like(un)
        if occ.any():
            from scipy import ndimage
            near_obst = ndimage.binary_dilation(occ, iterations=4) & ~occ
        w = un.astype(float) * (1.0 + near_obst)
        ux, uy, uw = CX[un], CY[un], w[un]
        best, bs = None, 0.0
        # a lattice centred on the field (mirror-symmetric), visited in team-relative order (sign = my attack direction) so a
        # tie between two equally good viewpoints is broken the same way for RED and BLUE
        kx, ky = int((FX - 0.35) / 0.3), int((FY - 0.35) / 0.3)
        for ix in range(-kx, kx + 1):
            for iy in range(-ky, ky + 1):
                gx, gy = sign * ix * 0.3, sign * iy * 0.3
                i, j = cell((gx, gy))
                if WALLBLOCK[i, j] or occ[i, j] or math.hypot(gx - me[0], gy - me[1]) < 0.5: continue    # a view > 0.5 m away
                m = np.hypot(ux - gx, uy - gy) < 1.6
                if not m.any(): continue
                gain = float((uw[m] * self._los(gx, gy, ux[m], uy[m])).sum())
                dist = plan_len(np.array([gx, gy])) if plan_len else float(np.hypot(gx - me[0], gy - me[1]))
                sc = gain - 1.5 * dist
                if sc > bs: best, bs = np.array([gx, gy]), sc
        return best, bs

    def plan(self, start, goal, pushable, blocked_extra=None, sign=1):
        """A* on the 0.1 m grid. Walls/goals (map knowledge) blocked; sensed obstacles blocked, or +PUSH_COST per cell if pushable.
        blocked_extra(x, y) -> bool adds more blocked cells (centre circle, the other crab)."""
        occ = self.occ(); s0, g0 = cell(start), cell(goal)
        openq = [(0.0, (0, 0), s0)]; came = {s0: None}; cost = {s0: 0.0}; n = 0
        while openq and n < 8000:
            _, _, c = heapq.heappop(openq); n += 1
            if c == g0: break
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not dx and not dy: continue
                    nb = (c[0] + dx, c[1] + dy)
                    if not (0 <= nb[0] < NX and 0 <= nb[1] < NY): continue
                    extra = 0.0
                    if nb != g0:
                        if WALLBLOCK[nb] or (blocked_extra and blocked_extra(XS[nb[0]], YS[nb[1]])): continue
                        if occ[nb]:
                            if not pushable: continue
                            extra = PUSH_COST
                    nc = cost[c] + math.hypot(dx, dy) + extra
                    if nc < cost.get(nb, 1e9):
                        cost[nb] = nc; came[nb] = c; heapq.heappush(openq, (nc + math.hypot(nb[0] - g0[0], nb[1] - g0[1]), (sign * nb[0], sign * nb[1]), nb))   # mirrored tie-break
        if g0 not in came: return []
        path = []; c = g0
        while c is not None: path.append(np.array([XS[c[0]], YS[c[1]]])); c = came[c]
        return path[::-1]

    def through_obstacle(self, path, k=4):
        occ = self.occ(); return any(occ[cell(p)] for p in path[:k])
