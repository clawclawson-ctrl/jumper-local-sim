"""Crab Warlords rules + arena geometry + ball model (SIM-ONLY VISION CONCEPT). Pure functions, unit-tested.
After Atari Warlords: three crabs (RED, BLUE, GREEN) in an equilateral-triangle arena, one castle per corner = an arc of
breakable bricks (two rows) around a KING block. One low-friction ball; crabs push / deflect it with their bodies (real physics).
A brick the ball touches breaks; the ball touching a king eliminates that crab; the last crab standing takes the round.
The referee is the only code that uses sim truth."""
from __future__ import annotations
import math, os
import numpy as np

KEYS = ("A", "B", "C")
TEAM = {"A": "RED", "B": "BLUE", "C": "GREEN"}
SIDE = 8.0                                   # m: triangle side (inner faces of the walls)
R_OUT = SIDE / math.sqrt(3)                  # 4.62 m centre -> corner
R_IN = R_OUT / 2                             # 2.31 m centre -> wall
ANG = {"A": math.radians(210), "B": math.radians(330), "C": math.radians(90)}   # corner directions (3-way symmetric)
U = {k: np.array([math.cos(a), math.sin(a)]) for k, a in ANG.items()}
VERT = {k: R_OUT * U[k] for k in KEYS}
WALL_H, WALL_T = 0.22, 0.10
KING_D = 1.10                                # king centre: this far from its corner along the bisector
KING_H = 0.13                                # king half size (a 26 cm block)
BRICK_ROWS = (0.80,)                    # brick arc radii around the king
BRICK_L, BRICK_T, BRICK_H = 0.085, 0.05, 0.07  # brick half sizes: along the arc, radial, height
HOME_D = 2.60                                # crab start: this far from its corner (2.0 m from the centre)
BALL_R = 0.065
BODY_R = 0.24                                # crab footprint radius for the kinematic model + clearances (legs reach 0.236 m)
CLEAR = 0.30                                 # leg clearance from walls / bricks / kings
BALL_DECEL = float(os.environ.get("WL_BALL_DECEL", "0.008"))   # m/s^2: low-friction ball (rolls for a long time)
WALL_E = 0.90                                # wall / brick restitution (bouncy stone; MuJoCo solref WALL_SOLREF below)
WALL_E_OUT = 0.975                           # the outer triangle walls: near-elastic (extra bouncy)
WALL_SOLREF_OUT = "-40000 -3"                # zeta ~0.008 -> restitution ~0.975
WALL_SOLREF = "-40000 -12"                   # direct stiffness/damping: zeta ~0.03 -> restitution ~0.9, 1.5 cm max squash at 3 m/s
BALL_SPEED = float(os.environ.get("WL_BALL_SPEED", "1.6"))     # m/s: "Ball speed" setting = the castle-test shot speed (no serves since the kickoff rule)
SERVE_V = 0.0                                # KICKOFF: the ball starts every round AT REST at the centre (no serve, no random kick)
BALL_VMAX = 3.2                              # m/s: no ball goes faster than this (sanity cap, also bounds tunnelling checks)
# PADDLE BOOST (sim setting, default ON, WL_PADDLE=0 turns it off): a crab body that walks INTO the ball hits it like a heavy,
# elastic paddle -- the ball's normal speed leaves at least (1 + PADDLE_E) x the crab's real walking speed along the contact normal
# (PADDLE_E = 1 is the exact elastic-collision result for a paddle much heavier than the 90 g ball). Nothing fakes the crab's speed.
PADDLE = os.environ.get("WL_PADDLE", "1") != "0"
PADDLE_E = 1.0
STALL_S = 10.0                               # KICKOFF rule: the referee only re-centres the ball (at rest) when it has been still this long
                                             # AND no crab has touched it in that time (= no crab can get to it: e.g. wedged near a wall
                                             # where no crab body + 0.30 m leg clearance fits behind it). Otherwise the crabs must fetch it.
REACH = BODY_R + BALL_R + 0.03               # a crab touching the ball counts as "getting to it" (resets the stall clock)
ROUND_MAX = float(os.environ.get("WL_ROUND_MAX", "150"))
CASTLE_RGBA = {"A": (0.85, 0.15, 0.10, 1.0), "B": (0.15, 0.30, 0.90, 1.0), "C": (0.10, 0.65, 0.20, 1.0)}
# eliminated crab: it leaves the field (the referee parks it outside the walls); its castle REMAINS (bricks + king stay as obstacles).


def rot(k):
    """rotation taking RED's frame to crab k's (120-degree symmetry)."""
    a = ANG[k] - ANG["A"]; c, s = math.cos(a), math.sin(a); return np.array([[c, -s], [s, c]])


def tri_clear(x, y):
    """distance to the nearest triangle wall face (positive inside)."""
    p = np.array([x, y]); return float(min(p @ U[k] + R_IN for k in KEYS))


# ---------------- ball-proof walls (ported from the SOCCER field: rounded corners + kick-strip ramps at the wall foot) ----------------
CORNER_R = 0.45                              # rounded corners (inside each castle, behind the king): circle tangent to both walls
FIL_H, FIL_A = float(os.environ.get("WL_FIL_H", "0")), math.radians(35)        # optional junction fillets (WL_FIL_H > 0: a stone deflector from the castle face
                                             # where it is FIL_H off the wall down to the wall at 35 deg). OFF by default: at 0.40 m they shielded
                                             # ~6 of 13 bricks per castle (king-hit rounds 31/60); the wider kick strip alone keeps the junctions clear
KICK_DEG, KICK_W = float(os.environ.get("WL_KICK_DEG", "6.0")), float(os.environ.get("WL_KICK_W", "0.20"))                 # kick strip: 20 cm wide (soccer: 12 cm), 6 deg ramp along every wall foot
KICK_A = 9.81 * math.sin(math.radians(KICK_DEG)) * 5 / 7   # m/s^2 a ball rolling down the strip gains (rolling sphere)


def _fillets():
    out = []
    for k in KEYS:
        K = VERT[k] - KING_D * U[k]; Ro = BRICK_ROWS[-1] + BRICK_T
        for j in KEYS:
            if j == k: continue
            th = np.linspace(0, 2 * math.pi, 3601); P = K + Ro * np.c_[np.cos(th), np.sin(th)]
            c = P @ U[j] + R_IN; i = np.where(np.abs(c - FIL_H) < 0.004)[0]
            A = max((P[q] for q in i), key=lambda q: np.linalg.norm(q - VERT[k]))
            A = K + (Ro - 0.02) * (A - K) / np.linalg.norm(A - K)                     # a little into the castle face: no seam
            F = A - (A @ U[j] + R_IN) * U[j]; t = (F - VERT[k]) / np.linalg.norm(F - VERT[k])
            B = F + t * (A @ U[j] + R_IN) / math.tan(FIL_A) - 0.01 * U[j]             # ends just inside the wall: no seam
            out.append((k, j, A, B))
    return out
FILLETS = _fillets() if FIL_H > 0 else []
CORNER_C = {k: VERT[k] - 2 * CORNER_R * U[k] for k in KEYS}
_UK = np.array([U[k] for k in KEYS])


def field_cn(P):
    """vectorised: clearance of points P (n, 2) to the arena's fixed stone (triangle walls, rounded corners, junction fillets)
    and the inward unit normal there. Behind a fillet (the sealed wedge) the clearance is negative."""
    P = np.atleast_2d(np.asarray(P, float)); UK = _UK
    c = P @ UK.T + R_IN; j = np.argmin(c, 1); best = c[np.arange(len(P)), j]; nrm = UK[j].copy()
    if len(P) == 1 and best[0] > 1.0: return best, nrm      # fast path: corners / fillets lie within 0.45 m of a wall, so the true
                                                            # clearance here is >= 0.55 m (every caller's threshold is below that)
    for k in KEYS:
        r = (P - VERT[k]) @ U[k] > -1.5 * CORNER_R
        if r.any():
            d = P[r] - CORNER_C[k]; dn = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9); cc = CORNER_R - dn
            m = cc < best[r]; ii = np.where(r)[0][m]; best[ii] = cc[m]; nrm[ii] = -(d[m] / dn[m, None])
    for (k, j, A, B) in FILLETS:
        AB = B - A; L2 = AB @ AB; s = np.clip((P - A) @ AB / L2, 0, 1); Q = A + s[:, None] * AB; D = P - Q
        dn = np.maximum(np.hypot(D[:, 0], D[:, 1]), 1e-9)
        nf = np.array([-AB[1], AB[0]]) / math.sqrt(L2)
        if (np.zeros(2) - A) @ nf < 0: nf = -nf                                          # the fillet's face toward the arena
        inside = (s > 0) & (s < 1); side = (P - A) @ nf
        cc = np.where(inside, side, dn); nn = np.where(inside[:, None], nf[None, :], D / dn[:, None])
        m = cc < best; best[m] = cc[m]; nrm[m] = nn[m]
    return best, nrm


_FIL = []
def _fil_prep():
    for (k, j, A, B) in FILLETS:
        AB = B - A; L2 = float(AB @ AB); nf = np.array([-AB[1], AB[0]]) / math.sqrt(L2)
        if (-A) @ nf < 0: nf = -nf
        _FIL.append((float(A[0]), float(A[1]), float(AB[0]), float(AB[1]), L2, float(nf[0]), float(nf[1])))
_fil_prep()
_UKL = [(float(U[k][0]), float(U[k][1])) for k in KEYS]
_VC = [(float(VERT[k][0]), float(VERT[k][1]), float(U[k][0]), float(U[k][1]), float(CORNER_C[k][0]), float(CORNER_C[k][1])) for k in KEYS]


def field_cs(x, y):
    """scalar field_cn for one point (pure python: the predictor / brain call it a lot) -> (clearance, (nx, ny))."""
    best = 1e9; n = (0.0, 0.0)
    for ux, uy in _UKL:
        c = x * ux + y * uy + R_IN
        if c < best: best = c; n = (ux, uy)
    if best > 1.0: return best, n
    for vx, vy, ux, uy, cx, cy in _VC:
        if (x - vx) * ux + (y - vy) * uy > -1.5 * CORNER_R:
            dx, dy = x - cx, y - cy; dn = max(math.hypot(dx, dy), 1e-9); c = CORNER_R - dn
            if c < best: best = c; n = (-dx / dn, -dy / dn)
    for ax, ay, bx, by, L2, nx, ny in _FIL:
        s_ = ((x - ax) * bx + (y - ay) * by) / L2
        if 0 < s_ < 1: c = (x - ax) * nx + (y - ay) * ny; nn = (nx, ny)
        else:
            s_ = min(max(s_, 0.0), 1.0); dx, dy = x - ax - s_ * bx, y - ay - s_ * by; dn = max(math.hypot(dx, dy), 1e-9); c = dn; nn = (dx / dn, dy / dn)
        if c < best: best = c; n = nn
    return best, n


def kick_acc(P):
    """vectorised: the kick strip's push on a ball centred at P (it rolls down the ramp, away from the wall foot)."""
    c, n = field_cn(P); on = (c < BALL_R + KICK_W) & (c > 0)
    return np.where(on[:, None], n * KICK_A, 0.0)


def wall_release_points(step=0.25):
    """spots where a ball rests against the fixed stone: along each wall (outside the castles), on each fillet face, and in the
    wall/fillet and fillet/castle junctions."""
    pts = []
    for j in KEYS:
        t = np.array([-U[j][1], U[j][0]]); F0 = -R_IN * U[j]
        for s_ in np.arange(-SIDE / 2, SIDE / 2, step):
            q = F0 + s_ * t + (BALL_R + 0.002) * U[j]
            if field_clear(*q) > BALL_R - 0.01 and all(np.linalg.norm(q - king(k)) > castle_zone(k) + BALL_R for k in KEYS): pts.append(q)
    for k in KEYS:                                                   # wall / castle junctions: the ball touching both (in the wedge)
        K = king(k); Ro = BRICK_ROWS[-1] + BRICK_T + BALL_R + 0.002
        th = np.linspace(0, 2 * math.pi, 3601); Pc = K + Ro * np.c_[np.cos(th), np.sin(th)]
        for j in KEYS:
            if j == k: continue
            c = Pc @ U[j] + R_IN; i = np.where(np.abs(c - BALL_R - 0.002) < 0.004)[0]
            if len(i) and field_clear(*Pc[i[0]]) > BALL_R - 0.01: pts.append(max((Pc[q] for q in i), key=lambda q: np.linalg.norm(q - VERT[k])))
    for (k, j, A, B) in FILLETS:
        nf = np.array([-(B - A)[1], (B - A)[0]]); nf /= np.linalg.norm(nf); nf = nf if (-A) @ nf > 0 else -nf
        for f in np.linspace(0.0, 1.0, 9):
            q = A + f * (B - A) + (BALL_R + 0.002) * nf
            for _ in range(30):                                      # settle the junction spots out of overlaps
                c, n = wall_at(q, Castles()); 
                if c >= BALL_R: break
                q = q + n * (BALL_R + 0.002 - c)
            pts.append(q)
    return pts


def unreachable_spots(step=0.15):
    """ball spots (outside the castles) no crab body can touch: the crab centre must keep its planning clearance (CLEAR - 0.05) from
    walls, fillets and castle discs and come within BODY_R + BALL_R + 0.03 of the ball."""
    xs = np.arange(-R_OUT, R_OUT, step); P = np.array([(x, y) for x in xs for y in np.arange(-R_IN, R_OUT, step)])
    c, _ = field_cn(P); P = P[c > BALL_R]
    P = P[np.array([all(np.linalg.norm(p - king(k)) > castle_zone(k) + BALL_R for k in KEYS) for p in P])]
    a = np.linspace(0, 2 * math.pi, 24, endpoint=False); ring = (BODY_R + BALL_R + 0.02) * np.c_[np.cos(a), np.sin(a)]
    bad = []
    for p in P:
        Cc = p + ring; cc, _ = field_cn(Cc)
        for k in KEYS: cc = np.minimum(cc, np.linalg.norm(Cc - king(k), axis=1) - castle_zone(k))
        if cc.max() < CLEAR - 0.05: bad.append(tuple(np.round(p, 2)))
    return bad


def field_clear(x, y): return field_cs(x, y)[0]
def field_normal(x, y): return np.array(field_cs(x, y)[1])
def kick1(p):
    c, n = field_cs(p[0], p[1])
    return np.array(n) * KICK_A if 0 < c < BALL_R + KICK_W else np.zeros(2)


def tri_normal(x, y):
    """inward unit normal of the nearest wall."""
    p = np.array([x, y]); k = min(KEYS, key=lambda k: p @ U[k] + R_IN); return U[k].copy()


def king(k): return VERT[k] - KING_D * U[k]


def home(k):
    p = VERT[k] - HOME_D * U[k]; return (float(p[0]), float(p[1]), math.degrees(math.atan2(-U[k][1], -U[k][0])))


def bricks(k):
    """[(x, y, yaw)] brick centres of castle k: two arcs around the king, facing the arena, from wall to wall (no gaps)."""
    K = king(k); out = []
    for r in BRICK_ROWS:
        n = int(2 * math.pi * r / (2 * BRICK_L * 0.98)); 
        for i in range(n):
            a = 2 * math.pi * (i + 0.5) / n; d = np.array([math.cos(a), math.sin(a)]); p = K + r * d
            if d @ (-U[k]) < -0.35: continue                      # the back arc (between the king and its corner)
            if tri_clear(*p) < -0.03: continue                    # into the wall: the arc ends at the wall
            out.append((float(p[0]), float(p[1]), a + math.pi / 2))
    return out


def brick_dist(p, b):
    """distance from point p to brick b's footprint (oriented box)."""
    c, s = math.cos(b[2]), math.sin(b[2]); dx, dy = p[0] - b[0], p[1] - b[1]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return math.hypot(max(0.0, abs(lx) - BRICK_L), max(0.0, abs(ly) - BRICK_T))


def king_dist(p, k):
    K = king(k); return math.hypot(max(0.0, abs(p[0] - K[0]) - KING_H), max(0.0, abs(p[1] - K[1]) - KING_H))


def castle_zone(k, margin=0.0):
    """radius of the castle's outer face around its king (planning)."""
    return BRICK_ROWS[-1] + BRICK_T + margin


class Castles:
    """brick state of the three castles (referee truth). alive[k][i] for bricks(k)[i]."""
    def __init__(self):
        self.B = {k: bricks(k) for k in KEYS}; self.alive = {k: [True] * len(self.B[k]) for k in KEYS}

    def reset(self): self.alive = {k: [True] * len(self.B[k]) for k in KEYS}
    def left(self, k): return int(sum(self.alive[k]))
    def total(self, k): return len(self.B[k])

    def gaps(self, k):
        """broken-brick spots of castle k (nearest the arena centre first): where a push gets through to the king."""
        g = [self.B[k][i][:2] for i, a in enumerate(self.alive[k]) if not a]
        g.sort(key=lambda q: math.hypot(*q)); return g

    def ball_hit(self, p, r=BALL_R + 0.005):
        """bricks the ball at p touches -> [(k, i)]; a king it touches -> k (or None)."""
        hits = [(k, i) for k in KEYS for i, b in enumerate(self.B[k]) if self.alive[k][i] and brick_dist(p, b) < r]
        kh = next((k for k in KEYS if king_dist(p, k) < r), None)
        return hits, kh


def rest_e(p, castles=None):
    """restitution of the solid nearest to p: outer wall WALL_E_OUT, bricks / kings WALL_E."""
    return WALL_E_OUT if field_clear(*p) <= wall_at(p, castles)[0] + 1e-9 else WALL_E


def wall_at(p, castles=None):
    """the solid things a ball can bounce off at p: (clearance, inward normal). Walls + intact bricks + kings."""
    c_, n_ = field_cs(float(p[0]), float(p[1]))
    best = (c_, np.array(n_))
    if castles is not None:
        for k in KEYS:
            K = king(k)
            if np.linalg.norm(p - K) > castle_zone(k, 0.3): continue
            d = king_dist(p, k)
            if d < best[0]: v = p - K; best = (d, v / max(1e-9, np.linalg.norm(v)))
            for i, b in enumerate(castles.B[k]):
                if not castles.alive[k][i]: continue
                d = brick_dist(p, b)
                if d < best[0]:
                    v = p - np.array(b[:2]); best = (d, v / max(1e-9, np.linalg.norm(v)))
    return best


def predict_ball(p, v, horizon, castles=None, step=0.05):
    """the ball's track: rolling deceleration + bounces off the ANGLED triangle walls (and intact bricks when castles is given).
    -> list of (t, p, v)."""
    p = np.array(p, float); v = np.array(v, float); t = 0.0; out = []
    while t < horizon - 1e-9:
        v = v + kick1(p) * step                    # kick strip at the wall foot
        sp = float(np.linalg.norm(v))
        if sp < 1e-3: out.append((t, p.copy(), v.copy())); break
        v = v * max(0.0, sp - BALL_DECEL * step) / sp; q = p + v * step
        d, n = wall_at(q, castles)
        if d < BALL_R and v @ n < 0:
            v = v - (1 + rest_e(q, castles)) * (v @ n) * n; q = p + v * step
            if wall_at(q, castles)[0] < BALL_R: q = p
        p = q; t += step; out.append((t, p.copy(), v.copy()))
    return out


def threat(k, track, margin=0.25):
    """first (t, p) of the track that reaches castle k's outer face (+margin), else None."""
    K = king(k); R = castle_zone(k, margin)
    for t, p, v in track:
        if np.linalg.norm(p - K) < R: return t, p
    return None


def tiebreak(castles, out):
    """time backstop: among crabs still in, the most bricks left takes the round; a tie stays a draw."""
    alive = [k for k in KEYS if not out.get(k)]
    if not alive: return None
    b = {k: castles.left(k) for k in alive}; m = max(b.values()); top = [k for k in alive if b[k] == m]
    return top[0] if len(top) == 1 else None


OPENERS = ("rush", "left", "right", "guard")   # kickoff: straight at the centre ball, via a left / right arc (strike from an angle), or guard first


def roll_round(rng, cfg=None):
    """per-round openers + start jitter (Random or Locked)."""
    cfg = dict(cfg or {}); mode = cfg.get("mode", "random"); out = {}
    for k in KEYS:
        op = cfg.get(k) if mode == "locked" and cfg.get(k) in OPENERS else str(rng.choice(OPENERS))
        out[k] = dict(opener=op, jit=(float(rng.uniform(-0.08, 0.08)), float(rng.uniform(-0.08, 0.08)), float(rng.uniform(-8, 8))))
    return out


def jittered_start(k, jit):
    x, y, d = home(k); return (x + jit[0], y + jit[1], d + jit[2])


def opener_waypoints(k, op):
    """waypoints in the world frame, defined in RED's frame and rotated (fair 3-way)."""
    H = np.array(home("A")[:2]); inward = -U["A"]; side = np.array([-inward[1], inward[0]])
    P = {"rush": [], "guard": [], "left": [H + 1.1 * inward + 0.55 * side], "right": [H + 1.1 * inward - 0.55 * side]}[op]
    return [tuple(rot(k) @ q) for q in P]


def aggression_from_env(env):
    base = float(env.get("WL_AGGR", "60"))
    return {k: max(0.0, min(100.0, float(env.get(f"WL_AGGR_{k}", base)))) for k in KEYS}


# ---------------- 2D ball/crab contact model (kinematic harness + unit tests; the real sim uses MuJoCo contacts) ----------------
def wedged(p, castles=None):
    """the ball at p is jammed where no crab body can get round it (against a wall / brick / king, or in a corner)."""
    return wall_at(np.asarray(p, float), castles)[0] < BALL_R + 0.04       # resting against a wall / brick / king


def paddle(v, cv, n):
    """ball velocity v after a crab moving at cv touches it along unit normal n (crab -> ball)."""
    v = np.asarray(v, float); vc = float(np.asarray(cv) @ n)
    tgt = (1 + PADDLE_E) * vc if PADDLE else vc
    if vc > 0.02 and v @ n < tgt: v = v + (tgt - v @ n) * n
    sp = float(np.linalg.norm(v))
    return v * (BALL_VMAX / sp) if sp > BALL_VMAX else v


def ball_step(p, v, dt, crabs, castles, crab_v=None, e_crab=0.6):
    """substepped so the ball never moves more than BALL_R/3 per step (no tunnelling through 10 cm bricks)."""
    n = max(1, int(math.ceil(float(np.linalg.norm(v)) * dt / (BALL_R / 3)))); br = []; kh = None
    for _ in range(n):
        p, v, b, k = _ball_step(p, v, dt / n, crabs, castles, crab_v, e_crab); br += b; kh = kh or k
    return p, v, br, kh


def _ball_step(p, v, dt, crabs, castles, crab_v=None, e_crab=0.6):
    """advance the ball dt: deceleration, crab pushes (disc BODY_R; the ball takes the crab's normal speed), walls/bricks.
    -> p, v, broken [(k, i)], king_hit k|None."""
    p = np.array(p, float); v = np.array(v, float) + kick1(p) * dt
    sp = float(np.linalg.norm(v))
    if sp > 1e-6: v = v * max(0.0, sp - BALL_DECEL * dt) / sp
    for j, (c, cv) in enumerate(zip(crabs, crab_v or [np.zeros(2)] * len(crabs))):
        if c is None: continue
        d = p - np.asarray(c[:2]); dn = float(np.linalg.norm(d))
        if dn < BODY_R + BALL_R:
            n = d / max(1e-9, dn); rel = v - cv
            if rel @ n < 0: v = v - (1 + e_crab) * (rel @ n) * n
            v = paddle(v, cv, n)                                    # the body pushes / kicks the ball ahead of it
            p = np.asarray(c[:2]) + n * (BODY_R + BALL_R)
    q = p + v * dt; dd, n = wall_at(q, castles)
    broken, kh = castles.ball_hit(q, BALL_R + 0.01)
    if dd < BALL_R and v @ n < 0:
        v = v - (1 + rest_e(q, castles)) * (v @ n) * n; q = p + v * dt
        if wall_at(q, castles)[0] < BALL_R: q = p
    for (k, i) in broken: castles.alive[k][i] = False
    return q, v, broken, kh


def unit():
    out = []
    assert abs(tri_clear(0, 0) - R_IN) < 1e-9 and abs(tri_clear(*VERT["A"])) < 1e-9
    # angled-wall bounce: ball heading straight at RED's-side... the wall opposite GREEN's corner (y = -R_IN) and an angled wall
    global KICK_A
    _ka = KICK_A; KICK_A = 0.0                              # pure restitution check (kick strip off)
    tr = predict_ball((0, 0), (0, -0.5), 8.0)
    ys = [p[1] for _, p, _ in tr]; assert min(ys) > -R_IN and tr[-1][2][1] > 0, "bounces off the bottom wall"
    d = -U["C"] * 0 + np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])   # toward the right wall (normal -U_A)
    tr = predict_ball((0, 0), 0.6 * d, 10.0); n = -U["A"]
    hit = next((i for i, (_, p, v) in enumerate(tr) if v @ U["A"] > 0), None)
    assert hit is not None and all(tri_clear(*p) > BALL_R - 0.02 for _, p, _ in tr), "reflects off the angled wall"
    v_in, v_out = tr[hit - 1][2], tr[hit][2]; assert abs(abs(v_out @ U["A"]) / abs(v_in @ U["A"]) - WALL_E_OUT) < 0.2, "restitution along the angled normal"
    KICK_A = _ka
    out.append("angled-wall bounce prediction (normal reflection, restitution)")
    C = Castles(); nb = C.total("A")
    assert nb == C.total("B") == C.total("C") and nb >= 10, nb
    b0 = C.B["A"][0]; hits, kh = C.ball_hit(np.array(b0[:2])); assert hits and kh is None
    p, v, br, kh = ball_step(np.array(b0[:2]) - 0.2 * (np.array(b0[:2]) - king("A")) / np.linalg.norm(np.array(b0[:2]) - king("A")) * -1,
                             -(np.array(b0[:2]) - king("A")) / np.linalg.norm(np.array(b0[:2]) - king("A")) * 2.0, 0.08, [], C)
    out.append(f"brick break ({nb} bricks per castle)")
    _, kh = C.ball_hit(king("B") + np.array([KING_H + 0.03, 0])); assert kh == "B"
    out.append("king hit -> elimination")
    for k in KEYS:                                            # 3-way mirror symmetry
        R = rot(k); assert np.allclose(R @ king("A"), king(k)) and np.allclose(R @ np.array(home("A")[:2]), np.array(home(k)[:2]))
        assert sorted(np.round([np.linalg.norm(np.array(b[:2])) for b in C.B[k]], 4)) == sorted(np.round([np.linalg.norm(np.array(b[:2])) for b in C.B["A"]], 4))
        for op in OPENERS: assert np.allclose(R @ np.array(opener_waypoints("A", op) or [(0, 0)]).T, np.array(opener_waypoints(k, op) or [tuple(R @ (0, 0))]).T)
    assert tri_clear(*home("A")[:2]) > 1.0 and min(np.linalg.norm(np.array(home("A")[:2]) - np.array(b[:2])) for b in C.B["A"]) > BODY_R + 0.25
    out.append("3-way symmetry: castles, homes, openers")
    C2 = Castles(); C2.alive["A"] = [False] * C2.total("A"); assert tiebreak(C2, {}) is None
    C2.alive["B"][0] = False; assert tiebreak(C2, {}) == "C"
    out.append("tiebreak: most bricks left")
    C3 = Castles(); b0 = np.array(C3.B["A"][3][:2]); K = king("A"); u = (b0 - K) / np.linalg.norm(b0 - K)
    p, v, br, kh = ball_step(b0 + 0.5 * u, -BALL_VMAX * u, 0.25, [], C3)     # 80 cm in one call: substeps must still hit the brick
    assert br == [("A", 3)] and kh is None and v @ u > 0, "no tunnelling at top speed"
    v2 = paddle(np.zeros(2), np.array([0.5, 0.0]), np.array([1.0, 0.0])); assert abs(v2[0] - (1 + PADDLE_E) * 0.5) < 1e-9 if PADDLE else True
    p, v, br, kh = ball_step(np.array([0.0, -R_IN + 0.4]), np.array([0.0, -BALL_VMAX]), 0.3, [], Castles())
    assert tri_clear(*p) > 0 and v[1] > 0 and abs(v[1] / BALL_VMAX - WALL_E_OUT) < 0.05, (p, v)
    out.append("fast ball: no tunnelling through bricks at 3.2 m/s; paddle boost = elastic heavy paddle")
    out.append(f"outer wall near-elastic (e={WALL_E_OUT}), no tunnelling at 3.2 m/s")
    # ball-proof walls: a ball released AT REST against every wall stretch, rounded corner, fillet and junction rolls >= 0.3 m into play
    rel = wall_release_points(); C4 = Castles()
    for q in rel:
        p, v = np.array(q), np.zeros(2)
        for _ in range(80): p, v, _, _ = ball_step(p, v, 0.05, [], C4)
        assert np.linalg.norm(p - q) >= 0.3, (q, p)
    out.append(f"ball at rest against {len(rel)} wall / corner / fillet / junction spots rolls >= 0.3 m into play")
    bad = unreachable_spots()
    assert not bad, bad[:5]
    out.append("no unreachable pockets: every resting ball spot outside the castles can be touched by a crab")
    return out


if __name__ == "__main__": print(unit())
