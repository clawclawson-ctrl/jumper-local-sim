"""Crab Surround brain (SIM-ONLY VISION CONCEPT). One per crab; uses ONLY its own camera + pose (+ the game rules).
 * own trail: proprioception (I know where my own tail has been, the same rule the referee uses)
 * opponent trail: glowing trail pixels in my own camera projected onto the floor, plus the line between my sightings
   of the opponent (a crab always lays a trail, a game rule)
 * opponent: detected by colour in my camera; heading/velocity from a line fit over recent sightings; when out of view,
   the Crab Tanks opponent-belief grid (where it is NOT) gives a likely spot
 * route: every 0.4 s score 12 candidate headings on a 0.25 m grid: a short probe must stay clear (0.30 m leg clearance),
   then a Voronoi race (BFS from my probe end vs the opponent's predicted spot) counts the floor I reach first, plus a
   self-trap check (floor reachable at all) and a small turn cost. Separated from the opponent -> fill my own pocket.
 * per-round opener (Random/Locked + seed), small start jitter, 3 s spin watchdog."""
from __future__ import annotations
import math, os
from collections import deque
import numpy as np
from scipy import ndimage
from jhs.soccer import vision
from jhs.soccer.brain import wrap, colour_blobs, TEAMFAM, OTHER, WALK
from . import game as G

CELL = 0.25
NX, NY = int(round(2 * G.FX / CELL)), int(round(2 * G.FY / CELL))
XS = -G.FX + CELL * (np.arange(NX) + 0.5); YS = -G.FY + CELL * (np.arange(NY) + 0.5)
CX, CY = np.meshgrid(XS, YS, indexing="ij")
CLEAR = 0.30                                   # leg clearance kept from walls and trails
WALLS = np.vectorize(G.F.outer_clear)(CX, CY) < CLEAR
HUE = {"A": (285, 325), "B": (168, 198)}       # trail hue windows: magenta (RED's trail), cyan (BLUE's trail)
NH = 16
SPIN_MAX = 3.0
REPLAN = 0.4
TRAP = 40                                      # cells (0.25 m) ~ 2.5 m^2: below this my pocket is a trap
ESC_S = 2.5                                    # s of escape walking before normal planning resumes
TURN_RATE = 0.65                               # rad/s max change of travel direction: ~0.45 m turning radius at walking speed
                                               # (a tighter hairpin would bring my legs back onto my own fresh wall)


def cell(xy):
    return (int(np.clip((xy[0] + G.FX) / CELL, 0, NX - 1)), int(np.clip((xy[1] + G.FY) / CELL, 0, NY - 1)))


def trail_points(img, selfmask, cam_pos, cam_R, cam, who, step=3, rmax=3.5):
    """floor points (N x 2) of crab `who`'s glowing trail seen in this image."""
    h, s, v = vision.hsv(img[::step, ::step]); lo, hi = HUE[who]
    m = (h >= lo) & (h <= hi) & (s >= 0.45) & (v >= 0.45) & ~selfmask[::step, ::step]
    ys, xs = np.nonzero(m)
    if len(xs) == 0: return np.zeros((0, 2))
    u = (xs * step + 0.5 - cam.W / 2) / cam.f; w = -(ys * step + 0.5 - cam.H / 2) / cam.f
    d = (cam_R @ np.stack([u, w, -np.ones_like(u)])).T
    ok = d[:, 2] < -1e-3; d = d[ok]
    t = (G.TRAIL_H * 0.6 - cam_pos[2]) / d[:, 2]
    P = cam_pos[None, :2] + t[:, None] * d[:, :2]
    r = np.hypot(P[:, 0] - cam_pos[0], P[:, 1] - cam_pos[1])
    P = P[(t > 0) & (r < rmax) & (np.abs(P[:, 0]) < G.FX) & (np.abs(P[:, 1]) < G.FY)]
    return P


def perceive(team, img, selfmask, cam_pos, cam_R, cam):
    other = OTHER[team]
    cb = colour_blobs(img, cam_pos, cam_R, cam, selfmask, TEAMFAM[other])
    return dict(opp=[d for d in cb if d["kind"] == "crab"], trail=trail_points(img, selfmask, cam_pos, cam_R, cam, other),
                ntrail_px=0)


def bfs(free, src):
    D = np.full(free.shape, 10 ** 6, np.int32)
    if src is None or not free[src]: return D
    q = deque([src]); D[src] = 0
    while q:
        i, j = q.popleft(); d = D[i, j] + 1
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < NX and 0 <= b < NY and free[a, b] and D[a, b] > d: D[a, b] = d; q.append((a, b))
    return D


class Brain:
    def __init__(self, team, rng=None, aggression=0.6):
        self.aggr = float(min(1.0, max(0.0, aggression)))
        self.team = team; self.rng = rng or np.random.default_rng(0); self.op_name = ""
        self.counts = {"escapes": 0, "replans": 0, "cutoffs": 0, "fills": 0, "unspins": 0, "opp_sightings": 0, "trail_cells_seen": 0}
        self.reset(0.0)

    def reset(self, t):
        self.t = t; self.own = G.Trail(); self.opp_cells = np.zeros((NX, NY), bool)
        self.opp = None; self.opp_t = -1e9; self.opp_v = np.zeros(2); self.sight = deque(maxlen=8)
        self.escaping = False; self.esc_until = -1e9; self.esc_h = None; self.reach_hist = deque(maxlen=60); self.reach_now = None
        self.tdir = None; self.head = None; self.plan_t = -1e9; self.state = "start"; self.sub = ""; self.spin_t0 = None
        self.opening = None; self.op_wps = []; self.op_t0 = t; self.ret = None; self.ret_noturn = False; self.info = {}
        from jhs.tanks.brain import OppBelief
        if not hasattr(self, "bel"): self.bel = OppBelief()
        self.bel.reset(G.START[OTHER[self.team]][:2], sigma=0.3); self.bel_t = t
        if not hasattr(self, "mem"):
            from jhs.tanks.explore import Memory
            self.mem = Memory()

    def set_opening(self, o, t):
        self.op_name = o["opener"]; self.opening = o; self.op_wps = list(G.opener_waypoints(self.team, o["opener"])); self.op_t0 = t

    # ---------------- perception ----------------
    def observe(self, t, per, pose):
        dt = max(0.0, t - self.t); self.t = t; self.pose = pose
        self.own.lay(pose)
        for p in per["trail"]:
            c = cell(p); self.opp_cells[c] = True
        self.counts["trail_cells_seen"] = int(self.opp_cells.sum())
        self.bel.spread(max(0.0, t - self.bel_t)); self.bel_t = t
        if per["opp"]:
            d = min(per["opp"], key=lambda d: d["rng"]); z = np.asarray(d["xy"], float)
            if self.opp is not None and t - self.opp_t < 1.0:          # it walked from there to here: its trail lies between
                for f in np.linspace(0, 1, 6): self.opp_cells[cell(self.opp + f * (z - self.opp))] = True
            self.sight.append((t, z)); self.opp = z; self.opp_t = t; self.counts["opp_sightings"] += 1
            self.bel.reset(z, sigma=0.12)
            S = [(a, b) for a, b in self.sight if t - a < 2.0]
            if len(S) >= 3:                                              # line fit: velocity (robust to one noisy frame)
                ts = np.array([a for a, _ in S]); P = np.array([b for _, b in S]); tc = ts - ts.mean()
                if (tc ** 2).sum() > 1e-4:
                    v = (tc[:, None] * (P - P.mean(0))).sum(0) / (tc ** 2).sum()
                    if np.linalg.norm(v) < 0.6: self.opp_v = v
        else:
            self.bel.seen_empty(self.mem.visible(pose, 3.0, 1.4))

    # ---------------- planning ----------------
    def blocked(self, predict=True):
        """walls + the opponent's trail (seen, + its predicted next ~2 s while it is tracked). My own trail is passable (rule)."""
        opp = self.opp_cells.copy()
        if predict and self.opp is not None and self.t - self.opp_t < 2.0:
            q0 = self.opp + self.opp_v * (self.t - self.opp_t)
            for f in np.linspace(0.3, 2.0, 6): opp[cell(q0 + self.opp_v * f)] = True
        return WALLS | ndimage.binary_dilation(opp, iterations=1)      # 1 cell (0.25 m) ~ the 0.30 m leg clearance

    def opp_guess(self):
        if self.opp is not None and self.t - self.opp_t < 4.0:
            return self.opp + self.opp_v * min(1.5, self.t - self.opp_t + 1.0), True
        g, _ = self.bel.peak(self.pose[:2], sign=1 if self.team == "A" else -1, dist_w=0.0)
        return np.asarray(g, float), False

    def choose(self, pose):
        x, y, yaw = pose; me = np.array([x, y]); occ = self.blocked()
        ov, seen = self.opp_guess(); oc = cell(ov)
        free = ~occ; free_all = free.copy()
        if self.opp is not None and self.t - self.opp_t < 1.0:          # the opponent's body + a step ahead are off-limits
            free &= np.hypot(CX - self.opp[0], CY - self.opp[1]) > 0.35     # bumping is harmless; just do not plan through its body
        best = None; out = []; esc = None
        for i in range(NH):
            h = 2 * math.pi * i / NH + (0.0 if self.team == "A" else math.pi)     # mirrored candidate order (fair ties)
            dirv = np.array([math.cos(h), math.sin(h)]); ok = True
            td = getattr(self, "tdir", None); td = yaw if td is None else td; q = me.copy(); c0 = cell(me)
            for n_ in range(16):                                            # probe the arc I would really walk (turn-rate limited)
                td += float(np.clip(wrap(h - td), -TURN_RATE * 0.25, TURN_RATE * 0.25)); q = q + 0.08 * np.array([math.cos(td), math.sin(td)])
                c = cell(q)
                if not free[c] and c != c0: ok = False; break
            if not ok:
                if esc is None or n_ > esc[0]: esc = (n_, h)
                continue
            end = cell(q); run = 16
            q2, td2 = q.copy(), td
            for _ in range(14):                                             # how far the arc stays clear beyond the probe (room to turn)
                td2 += float(np.clip(wrap(h - td2), -TURN_RATE * 0.25, TURN_RATE * 0.25)); q2 = q2 + 0.08 * np.array([math.cos(td2), math.sin(td2)])
                if not free[cell(q2)]: break
                run += 1
            Dm = bfs(free, end); reach = int((Dm < 10 ** 6).sum())
            fo = free_all | (np.hypot(CX - ov[0], CY - ov[1]) < 0.45)
            Do = bfs(fo, oc)
            mine = int(((Dm < Do) & (Dm < 10 ** 6)).sum()); theirs = int(((Do < Dm) & (Do < 10 ** 6)).sum())
            sep = not ((Dm < 10 ** 6) & (Do < 10 ** 6)).any()
            ref = getattr(self, 'tdir', None) if getattr(self, 'tdir', None) is not None else yaw
            turn = abs(wrap(h - ref))
            if turn > math.radians(100) and not (self.escaping or getattr(self, 'free_turn', False)): continue    # smooth play; ESCAPE may double back (own trail is passable)
            a = self.aggr
            if sep:      # sealed off from each other: fill my pocket, hug the walls (wall-follow keeps the pocket compact)
                nb = ndimage.binary_dilation(~free, iterations=1)[end]
                sc = reach * 1.0 + 1.0 * nb - 15.0 * turn
            else:
                safe = (mine - theirs) + 0.35 * reach - 15.0 * turn
                # HUNT: get ahead of the opponent's predicted path and lay my wall across it (shrinks its territory)
                ex, ey = XS[end[0]], YS[end[1]]; vn = float(np.linalg.norm(self.opp_v))
                ahead = self.opp_v / vn * 0.9 if vn > 0.05 else np.zeros(2)
                tgt = ov + ahead if seen else ov
                hunt = 0.5 * (mine - theirs) + 0.2 * reach - 300.0 * math.hypot(ex - tgt[0], ey - tgt[1]) - 8.0 * turn
                sc = (1 - a) * safe + a * hunt
            sc += 6.0 * run
            if reach < 6: sc -= 500                                         # self-trap
            if self.opp is not None and self.t - self.opp_t < 2.0:          # head-on: predicted closest approach over the next 3 s
                ts = np.linspace(0, 2.0, 9); q0 = self.opp + self.opp_v * (self.t - self.opp_t)
                dm = float(np.min(np.hypot(me[0] + dirv[0] * 0.32 * ts - q0[0] - self.opp_v[0] * ts, me[1] + dirv[1] * 0.32 * ts - q0[1] - self.opp_v[1] * ts)))
                if dm < 0.45: sc -= 200 * (1 - a) * (0.45 - dm)                   # bumps are harmless: only a cautious crab shies away
            out.append((round(math.degrees(h)), round(sc, 1)))
            if best is None or sc > best[0]: best = (sc, h, sep, mine, theirs, reach)
        self.info = dict(cands=out, seen=seen)
        if best is None and not getattr(self, "_retry", False):          # no smooth route: any walking direction is legal (own trail passable)
            self._retry = True; self.free_turn = True; self.tdir = None
            try: r = self.choose(pose)
            finally: self._retry = False
            if r is not None: return r
        if best is None and esc is not None: return (-1e3, esc[1], True, 0, 0, 0)     # boxed in: the longest clear arc
        return best

    def _drive(self, pose, h, speed=WALK, strafe=False, face=None):
        """omnidirectional: walk along h at once (a crab strafes; no stop-and-turn wobble), while the body turns toward h so the
        camera looks where I am going."""
        x, y, yaw = pose; e = wrap((h if face is None else face) - yaw); c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = math.cos(h), math.sin(h)
        vx, vy = (c * dx + s * dy) * speed, (-s * dx + c * dy) * speed
        p = {"ly": -float(np.clip(vx, -0.45, 0.45)), "lx": -float(np.clip(vy, -0.45, 0.45))}
        if not strafe and abs(e) > 0.15: p["rx"] = (-1 if e > 0 else 1) * max(0.62, 0.5 + 0.3 * min(1.0, abs(e)))
        return p

    def decide(self, pose):
        x, y, yaw = pose; me = np.array([x, y]); t = self.t
        if self.opening is not None:
            while self.op_wps and np.linalg.norm(np.array(self.op_wps[0]) - me) < 0.3: self.op_wps.pop(0)
            near = self.opp is not None and t - self.opp_t < 0.5 and np.linalg.norm(self.opp - me) < 2.6
            if not self.op_wps or t - self.op_t0 > 1.5 + 6.5 * (1 - self.aggr) or near: self.opening = None   # an aggressive crab leaves its opener early
            else:
                wp = np.array(self.op_wps[0]); h = math.atan2(wp[1] - y, wp[0] - x)
                occ = self.blocked(); probe = [cell(me + np.array([math.cos(h), math.sin(h)]) * r) for r in (0.3, 0.5)]
                if not any(occ[c] for c in probe):
                    self.state = "opener"; self.sub = f"OPENER {self.op_name}"; self.head = h
                    return self._guard(self._drive(pose, self._slew(h)), pose)
                self.opening = None
        if self.escaping:
            d_ = np.array([math.cos(self.esc_h), math.sin(self.esc_h)]); occ_ = self.blocked()
            if any(occ_[cell(me + d_ * r)] for r in (0.3, 0.55)): self.esc_until = t     # wall / trail ahead: stop escaping, replan
            if t >= self.esc_until: self.escaping = False; self.plan_t = -1e9; self.tdir = None; self.free_turn = True
            else:
                self.state = "escape"; self.tdir = self.esc_h; self.tdir_t = t; self.head = self.esc_h
                return self._guard(self._drive(pose, self.esc_h), pose)
        if self.head is None or t - self.plan_t > REPLAN:
            self.plan_t = t; self.counts["replans"] += 1; b = self.choose(pose); self.free_turn = False
            e = self.check_escape(pose, b)
            if e is not None:
                self.escaping = True; self.esc_h = e[0]; self.esc_until = t + ESC_S; self.counts["escapes"] += 1
                self.state = "escape"; self.sub = f"ESCAPE: pocket {e[2]} -> open {e[1]} cells"
                self.tdir = self.esc_h; self.tdir_t = t; self.head = self.esc_h
                return self._guard(self._drive(pose, self.esc_h), pose)
            if b is None:
                self.state = "trapped"; self.sub = "TRAPPED: no clear heading"
                self.head = self.head if self.head is not None else yaw
            else:
                sc, h, sep, mine, theirs, reach = b; self.head = h
                if sep: self.state = "fill"; self.sub = f"FILL own pocket ({reach} cells)"; self.counts["fills"] += 1
                else:
                    self.state = ("hunt" if self.aggr >= 0.5 else "cutoff") if mine > theirs else ("hunt" if self.aggr >= 0.5 else "race")
                    self.sub = f"{'HUNT' if self.aggr >= 0.5 else 'CUT OFF' if mine > theirs else 'RACE'}: mine {mine} vs theirs {theirs} ({self.aggr:.0%})"
                    if mine > theirs: self.counts["cutoffs"] += 1
        hd = self._slew(self.head); face = None
        if self.aggr >= 0.5 and self.state == "hunt":            # a hunter keeps its camera on the opponent while it strafes to cut it off
            ov, _ = self.opp_guess(); fa = math.atan2(ov[1] - y, ov[0] - x)
            if abs(wrap(fa - hd)) < math.radians(80): face = fa
        return self._guard(self._drive(pose, hd, face=face), pose)

    def reach_from(self, xy, free=None):
        if free is None: free = ~self.blocked()
        f = free.copy(); c = cell(xy); f[c] = True
        return int((bfs(f, c) < 10 ** 6).sum())

    def check_escape(self, pose, best):
        """trap watch: flood fill of the floor I can still reach (walls + opponent trail + its predicted path; my own trail is passable).
        Escape when the pocket is small, shrinking fast, or the best forward route leads into a much smaller part of it; the escape
        heading may point anywhere, even straight back over my own trail (legal: own trail is passable; the crab walks omnidirectionally,
        so it backs out at once with no turn on the spot)."""
        me = np.array(pose[:2]); free = ~self.blocked(); R0 = self.reach_from(me, free); self.reach_now = R0
        self.reach_hist.append((self.t, R0))
        old = [r for (tt, r) in self.reach_hist if self.t - tt >= 3.0]
        shrink = bool(old) and R0 < 0.6 * old[-1] and R0 < 150
        fwd = best[5] if best is not None else 0
        closing = fwd < TRAP and R0 >= 2 * TRAP                          # the corridor ahead closes, but there is room elsewhere
        if self.t - self.esc_until < 4.0: return None                   # cool-down: no escape ping-pong
        if not (R0 < TRAP or shrink or closing): return None
        cand = None
        for i in range(NH):
            h = 2 * math.pi * i / NH + (0.0 if self.team == "A" else math.pi); d = np.array([math.cos(h), math.sin(h)]); ok = True
            for r in np.arange(0.15, 0.9, 0.08):
                c = cell(me + d * r)
                if not free[c] and c != cell(me): ok = False; break
            if not ok: continue
            R = self.reach_from(me + d * 0.8, free)
            if cand is None or R > cand[1]: cand = (h, R)
        if os.environ.get("SR_NO_ESCAPE") == "1": return None          # test switch (before/after comparisons)
        if cand is None or cand[1] < 1.3 * max(fwd, 1) or cand[1] < TRAP: return None
        return cand[0], cand[1], max(fwd, 0)

    def _slew(self, h):
        """travel direction eases toward the plan at TURN_RATE (smooth curved trail, no hairpins)."""
        if getattr(self, "tdir", None) is None or self.t - getattr(self, "tdir_t", -1e9) > 0.5: self.tdir = h
        else: self.tdir = self.tdir + float(np.clip(wrap(h - self.tdir), -TURN_RATE * (self.t - self.tdir_t), TURN_RATE * (self.t - self.tdir_t)))
        self.tdir_t = self.t; return self.tdir

    def _guard(self, cmd, pose):
        spin = abs(cmd.get("rx", 0)) > 0.1 and abs(cmd.get("lx", 0)) + abs(cmd.get("ly", 0)) < 0.08
        if not spin: self.spin_t0 = None; return cmd
        if self.spin_t0 is None: self.spin_t0 = self.t
        if self.t - self.spin_t0 <= SPIN_MAX: return cmd
        self.spin_t0 = None; self.counts["unspins"] += 1; self.sub = "SPIN WATCHDOG: strafe"
        return self._drive(pose, self.head if self.head is not None else pose[2], strafe=True)

    # ---------------- walk back between rounds (trails are gone) ----------------
    def begin_return(self, spot, t): self.ret = spot; self.ret_noturn = False; self.state = "return"; self.sub = "walking back to start"

    def return_cmd(self, pose):
        x, y, yaw = pose; sx, sy, sd = self.ret; d = math.hypot(sx - x, sy - y)
        if d > 0.12:
            h = math.atan2(sy - y, sx - x); c = self._drive(pose, h, speed=min(WALK, max(0.18, d * 1.4)))
            return self._guard(c, pose)
        e = wrap(math.radians(sd) - yaw)
        if abs(e) < math.radians(11) or self.ret_noturn: return {}
        c = {"rx": (-1 if e > 0 else 1) * 0.62}
        g = self._guard(c, pose)
        if "rx" not in g or g is not c: self.ret_noturn = True; return {}
        return g

    def grid(self):
        """overlay: 0 free, 1 own trail, 2 seen opponent trail, 3 wall band (coarse, as this crab believes)."""
        own = np.zeros((NX, NY), np.int8)
        for a, b in self.own.segments(0.0): own[cell(a)] = 1; own[cell(b)] = 1
        g = np.where(self.opp_cells, 2, own); return g.tolist()


def unit():
    out = []; b = Brain("A"); b.pose = (-2.6, 0.0, 0.0)
    best = b.choose(b.pose); assert best is not None and abs(wrap(best[1])) < math.radians(100), best
    out.append(f"open field: heads into the arena ({math.degrees(best[1]):.0f} deg)")
    for i in range(30): b.own.lay((-1.0, -1.5 + 0.1 * i, math.pi / 2))
    b.pose = (-1.0, 1.6, math.pi / 2); b.own.lay(b.pose)
    best = b.choose(b.pose); assert best is not None
    assert not b.blocked()[cell((-1.0, 0.0))], "own trail must be passable"
    out.append("own trail passable in the planner")
    # shrinking pocket: opponent trail closes a U ahead of me (open behind) -> ESCAPE backwards
    e = Brain("A", aggression=0.6); e.t = 5.0; e.pose = (0.0, 0.0, 0.0)
    for yy in np.arange(-0.75, 0.76, 0.25): e.opp_cells[cell((0.8, yy))] = True
    for xx in np.arange(-0.3, 0.81, 0.25): e.opp_cells[cell((xx, 0.75))] = True; e.opp_cells[cell((xx, -0.75))] = True
    for xx in np.arange(-3.5, -0.3, 0.25): e.opp_cells[cell((xx, 0.75))] = True; e.opp_cells[cell((xx, -0.75))] = True
    e.opp_cells[cell((-3.3, 0.0))] = True
    for yy in np.arange(-2.25, -0.75, 0.25): e.opp_cells[cell((0.8, yy))] = True
    e.head = 0.0; e.tdir = 0.0; e.tdir_t = 5.0; e.plan_t = -1e9
    for k in range(4): e.t = 5.0 + 0.5 * k; e.reach_hist.append((e.t - 4, 400)); c = e.decide(e.pose)
    assert e.counts["escapes"] >= 1 and abs(wrap(e.esc_h)) > math.pi / 2, (e.counts, e.esc_h, e.reach_now)
    out.append(f"shrinking pocket -> ESCAPE at {math.degrees(e.esc_h):.0f} deg (backs out over its own trail)")
    import os as _os
    ag = G.aggression_from_env({"SR_AGGR": "60", "SR_AGGR_A": "20", "SR_AGGR_B": "80"}); assert ag == {"A": 20.0, "B": 80.0}
    lo, hi = Brain("A", aggression=ag["A"] / 100), Brain("B", aggression=ag["B"] / 100)
    for q in (lo, hi):
        q.t = 1.0; q.pose = (0.0, 0.0, 0.0); q.opp = np.array([2.0, 1.0]); q.opp_t = 1.0; q.opp_v = np.array([0.0, -0.3])
    bl, bh = lo.choose(lo.pose), hi.choose(hi.pose); assert lo.aggr == 0.2 and hi.aggr == 0.8 and bl[0] != bh[0]
    out.append("per-crab aggression reaches the planner (20% / 80%)")
    b2 = Brain("B"); b2.pose = (0.0, 0.0, 0.0); b2.opp_cells[cell((1.3, 0.0))] = True; b2.opp_cells[cell((1.3, 0.25))] = True; b2.opp_cells[cell((1.3, -0.25))] = True
    best = b2.choose(b2.pose); assert best is not None and abs(wrap(best[1])) > 0.4, best; out.append("avoids a seen opponent trail")
    b3 = Brain("A"); b3.t = 1.0; b3.pose = (0.0, 0.0, 0.0)
    for k in range(4): b3.t = 1.0 + 0.1 * k; b3.observe(b3.t, dict(opp=[dict(xy=np.array([1.5, 0.3 * 0.1 * k]), rng=1.5)], trail=np.zeros((0, 2))), b3.pose)
    assert b3.opp_v[1] > 0.1, b3.opp_v; out.append("opponent velocity line fit")
    c = {"rx": 0.7}; b3.head = 0.5; b3.spin_t0 = 0.0; b3.t = 5.0; g = b3._guard(c, b3.pose); assert "rx" not in g; out.append("spin watchdog")
    return out
