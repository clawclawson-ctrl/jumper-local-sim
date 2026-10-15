"""Crab Surround rules + trails + referee geometry (SIM-ONLY VISION CONCEPT). Pure functions, unit-tested.
Arena: the Crab Tanks open field (7.0 x 4.5 m, rounded walls, no interior pieces). Each crab lays a low glowing wall behind
it (a sim add-on: drawn into every camera, physics never touches it). Touching any wall or trail ends the round for that
crab (a crab may walk over its OWN trail freely; only the opponent's trail and the outer wall count); the last crab standing scores. The referee is the only code that uses sim truth."""
from __future__ import annotations
import math, os
os.environ.setdefault("TK_ARENA", "open")          # the tanks open field; must be set before jhs.tanks.* is imported
import numpy as np
from jhs.tanks import field as F


FX, FY, CORNER_R, TEAM = F.FX, F.FY, F.CORNER_R, F.TEAM
START = {"A": (-2.6, 0.0, 0.0), "B": (2.6, 0.0, 180.0)}
TAIL = 0.24          # trail is laid at the rear edge of the crab's footprint (feet reach 0.236 m from the base centre)
TRAIL_W = 0.08       # drawn width of the low wall (m)
TRAIL_H = 0.06       # height (m): low, below the camera, above the floor
LAY_STEP = 0.10      # a new trail point every 10 cm of base travel (gait sway of a few cm lays nothing)
BODY_R = 0.20        # referee contact radius: a trail/wall closer than this to the base centre touches a leg (half-span 0.25 minus gait)
CRASH_R = BODY_R + TRAIL_W / 2
OWN_GRACE = 0.70     # m of my own most recent trail that cannot be "hit" (it starts right behind my legs)
ROUND_MAX = 120.0    # s: a round that runs this long ends as a draw (time backstop; the match clock is the other)
TRAIL_RGBA = {"A": (1.0, 0.10, 0.85, 1.0), "B": (0.0, 0.95, 1.0, 1.0)}    # magenta (RED crab) / cyan (BLUE crab)
NEON = {"A": "MAGENTA", "B": "CYAN"}


class Trail:
    """the wall follows the crab's real (curved, omnidirectional) path: a point every LAY_STEP of base travel, placed TAIL behind the
    base along the direction of travel (so turning on the spot lays nothing, and strafing lays behind the sideways motion)."""
    def __init__(self): self.pts = []; self.cum = [0.0]; self.c = None

    def clear(self): self.pts = []; self.cum = [0.0]; self.c = None

    def tail_of(self, pose):
        x, y, yaw = pose; return (x - TAIL * math.cos(yaw), y - TAIL * math.sin(yaw))

    def lay(self, pose):
        x, y = pose[0], pose[1]
        if self.c is None: self.c = (x, y); return False
        m = math.hypot(x - self.c[0], y - self.c[1])
        if m < LAY_STEP: return False
        ux, uy = (x - self.c[0]) / m, (y - self.c[1]) / m; self.c = (x, y)
        p = (x - TAIL * ux, y - TAIL * uy)
        if not self.pts: self.pts.append(p); self.cum = [0.0]; return True
        d = math.hypot(p[0] - self.pts[-1][0], p[1] - self.pts[-1][1])
        self.pts.append(p); self.cum.append(self.cum[-1] + d); return True

    def segments(self, skip_recent=0.0):
        """[(a, b)] trail segments, leaving out the last skip_recent metres."""
        if len(self.pts) < 2: return []
        lim = self.cum[-1] - skip_recent; out = []
        for i in range(1, len(self.pts)):
            if self.cum[i] > lim: break
            out.append((self.pts[i - 1], self.pts[i]))
        return out

    def length(self): return self.cum[-1] if self.cum else 0.0


def seg_dist(p, a, b):
    ax, ay = a; bx, by = b; dx, dy = bx - ax, by - ay; L = dx * dx + dy * dy
    t = 0.0 if L < 1e-12 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def check_crash(k, pose, trails, other_pose=None):
    """referee (sim truth): -> None or a reason string for crab k at pose."""
    p = pose[:2]
    if F.outer_clear(*p) < BODY_R: return "wall"
    for who, tr in trails.items():
        if who == k: continue                 # rule: a crab may walk over its OWN trail freely
        for a, b in tr.segments(0.0):
            if seg_dist(p, a, b) < CRASH_R: return f"{TEAM[who]} trail"
    return None                       # crab-to-crab contact is harmless (they may bump); other_pose kept for the call signature


def aggression_from_env(env):
    """per-crab aggression (%) from SR_AGGR_A / SR_AGGR_B, falling back to SR_AGGR (both), default 60."""
    base = float(env.get("SR_AGGR") or 60)
    return {k: max(0.0, min(100.0, float(env.get("SR_AGGR_" + k) or base))) for k in ("A", "B")}


# ---- per-round openers (same menu + odds for both, mirrored for BLUE) ----
OPENERS = ("STRAIGHT", "SWEEP_LEFT", "SWEEP_RIGHT", "CUT_MID")


def roll_round(rng, cfg=None):
    cfg = cfg or {}; out = {}
    for k in ("A", "B"):
        if cfg.get("mode") == "locked": op = str(cfg.get(k) or "STRAIGHT").upper()
        else: op = OPENERS[int(rng.integers(len(OPENERS)))]
        if op not in OPENERS: op = "STRAIGHT"
        out[k] = dict(opener=op, jit=(float(rng.uniform(-0.06, 0.06)), float(rng.uniform(-0.06, 0.06)), float(rng.uniform(-6, 6))))
    return out


def jittered_start(k, jit):
    x, y, d = START[k]; return (x + jit[0], y + jit[1], d + jit[2])


def opener_waypoints(k, op):
    """team-relative waypoints (RED's x; BLUE mirrors x and y: point symmetry)."""
    pts = {"STRAIGHT": [(-1.3, 0.0)], "SWEEP_LEFT": [(-1.9, 0.9), (-0.9, 1.3)],
           "SWEEP_RIGHT": [(-1.9, -0.9), (-0.9, -1.3)], "CUT_MID": [(-1.6, 0.5), (-0.6, 0.2)]}[op]
    s = 1 if k == "A" else -1
    return [(s * x, s * y) for x, y in pts]


def unit():
    out = []
    tr = Trail()
    for i in range(40): tr.lay((-1.0 + 0.05 * i, 0.0, 0.0))
    assert tr.length() > 1.5; out.append(f"trail lays {len(tr.pts)} pts")
    sp = Trail()
    for i in range(100): sp.lay((0.0, 0.0, 0.1 * i))
    assert not sp.pts; out.append("turning on the spot lays no wall")
    cv = Trail()
    for i in range(80): a = 0.03 * i; cv.lay((math.sin(a), 1 - math.cos(a), a))
    assert check_crash("A", (math.sin(2.37), 1 - math.cos(2.37), 2.37), {"A": cv}) is None; out.append("curved path: own fresh wall not hit")
    assert check_crash("A", (1.0, 0.0, 0.0), {"A": tr}) is None, "own fresh trail must not count"
    assert check_crash("B", (-0.5, 0.05, 1.57), {"A": tr}) == "RED trail"
    assert check_crash("A", (-0.5, 0.05, 1.57), {"A": tr}) is None, "own trail is passable"
    assert check_crash("A", (FX - 0.1, 0, 0), {}) == "wall"
    assert check_crash("A", (0, 0, 0), {}, (0.3, 0)) is None
    out.append("crash: wall / opponent trail only; own trail passable; crab-to-crab contact harmless")
    r = roll_round(np.random.default_rng(1)); assert set(r) == {"A", "B"}
    l = roll_round(np.random.default_rng(1), {"mode": "locked", "A": "CUT_MID"}); assert l["A"]["opener"] == "CUT_MID" and l["B"]["opener"] == "STRAIGHT"
    assert opener_waypoints("B", "STRAIGHT")[0][0] > 0; out.append("openers random/locked + mirrored")
    return out
