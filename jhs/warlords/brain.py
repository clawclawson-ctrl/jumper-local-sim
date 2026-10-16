"""Crab Warlords brain (one crab). SIM-ONLY VISION CONCEPT: it sees the ball (and the other crabs) only in its own camera, knows
its own pose and the fixed arena map, and reads the public scoreboard (bricks left per castle, who is out) like a player
watching the score. It drives the official walking policy through a virtual gamepad.
DEFEND: predict the ball's track (rolling decel + bounces off the angled walls) and stand on my guard ring where it would hit my
castle. ATTACK: walk around behind the ball and push it at the weakest enemy castle / exposed king. HOLD: guard spot facing the
ball, scan when it is lost. Per-crab aggression (how far from home it goes after the ball), random openers, jitter, spin watchdog,
and a final safety layer (look-ahead: never walk into a wall, brick or king; slide along them)."""
from __future__ import annotations
import math
from collections import deque
import numpy as np
from . import game as G

WALK, PUSH = 0.45, 0.42             # stick (x0.8 m/s): ~0.36 m/s walking
GUARD = G.castle_zone("A") + 0.42   # guard ring radius around my king (front face of the bricks + body + margin)
SPIN_MAX = 4.0
DEF_H = float(__import__('os').environ.get('WL_DEF_H', '2.5'))   # s: defend against threats this soon
LOOK = 0.45                          # m: safety look-ahead along the commanded walk


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi
def unitv(v): n = float(np.linalg.norm(v)); return v / n if n > 1e-9 else np.zeros(2)


def obstacle(p, castles_alive=None):
    """(clearance from my body centre to the nearest wall / castle, outward normal). Castles as solid discs (planning)."""
    best = (G.tri_clear(*p), G.tri_normal(*p))
    for k in G.KEYS:
        K = G.king(k); d = float(np.linalg.norm(p - K)) - G.castle_zone(k)
        if d < best[0]: best = (d, unitv(p - K))
    return best


class Brain:
    def __init__(self, team, rng=None, aggression=0.6):
        self.team = team; self.rng = rng or np.random.default_rng(0); self.aggr = float(min(1.0, max(0.0, aggression)))
        self.K = G.king(team); self.H = np.array(G.home(team)[:2]); self.inward = -G.U[team]
        self.counts = dict(defends=0, attacks=0, unspins=0, safety=0, scans=0)
        self.scoreboard = {k: 13 for k in G.KEYS}; self.out = {k: False for k in G.KEYS}
        self.reset(0.0)

    def reset(self, t):
        self.t = t; self.obs = deque(maxlen=12); self.ball = None; self.ball_v = np.zeros(2); self.ball_t = -1e9
        self.state = "hold"; self.sub = ""; self.spin_t0 = None; self.opening = None; self.op_wps = []; self.op_name = "-"
        self.pose = None; self.target = None; self.track = []; self.jit = self.rng.normal(0, 0.05, 2); self.crabs = []

    def set_opening(self, rnd, t):
        self.op_name = rnd["opener"]; self.op_wps = G.opener_waypoints(self.team, rnd["opener"]); self.opening = t
        self.jit = self.rng.normal(0, 0.05, 2)

    # ---------------- perception ----------------
    def observe(self, t, p, pose):
        self.t = t; self.pose = pose
        bs = p.get("ball") or []
        if bs:
            b = np.asarray(min(bs, key=lambda d: d.get("rng", 1.0))["xy"], float)
            if self.ball is not None and t - self.ball_t < 1.0 and np.linalg.norm(b - self.ball) > 1.0: self.obs.clear()   # a jump: new track
            self.obs.append((t, b)); self.ball = b; self.ball_t = t
            o = [(tt, q) for tt, q in self.obs if t - tt < 1.0]
            if len(o) >= 3:
                T = np.array([tt for tt, _ in o]); Q = np.array([q for _, q in o]); T = T - T.mean()
                if (T ** 2).sum() > 1e-6: self.ball_v = (T[:, None] * (Q - Q.mean(0))).sum(0) / (T ** 2).sum(); self.ball = Q.mean(0) + self.ball_v * T[-1]
        self.crabs = [np.asarray(d["xy"], float) for d in (p.get("crabs") or [])]

    def ball_now(self):
        if self.ball is None or self.t - self.ball_t > 4.0: return None, None
        dt = self.t - self.ball_t; tr = G.predict_ball(self.ball, self.ball_v, dt) if dt > 0.05 else []
        return (tr[-1][1], tr[-1][2]) if tr else (self.ball.copy(), self.ball_v.copy())

    # ---------------- decisions ----------------
    def decide(self, pose):
        return self._safety(self._guard(self._decide(pose)), pose)

    def _decide(self, pose):
        x, y, yaw = pose; me = np.array([x, y]); t = self.t
        b, bv = self.ball_now()
        self.track = G.predict_ball(b, bv, 6.0) if b is not None else []
        th = G.threat(self.team, self.track, 0.35) if b is not None else None
        enemies = [k for k in G.KEYS if k != self.team and not self.out.get(k)]
        if th is not None and th[0] < DEF_H:                    # DEFEND: stand where the ball would reach my castle
            q = th[1]; g = self.K + GUARD * unitv(q - self.K)
            if np.linalg.norm(q - self.K) < 1e-6: g = self.K + GUARD * self.inward
            g = self._ring(g); self.state = "defend"; self.sub = f"DEFEND: ball at my castle in {th[0]:.1f}s"; self.counts["defends"] += 1
            return self._go(pose, g, face=b, speed=WALK)
        if self.opening is not None:
            if self.op_wps and t - self.opening < 4.0 + 3.0 * (1 - self.aggr):
                wp = np.array(self.op_wps[0])
                if np.linalg.norm(wp - me) < 0.15: self.op_wps.pop(0)
                else:
                    self.state = "opener"; self.sub = f"OPENER {self.op_name}"
                    return self._go(pose, wp, face=b if b is not None else np.zeros(2))
            self.opening = None
        a_ = min(1.0, self.aggr + (0.3 if len(enemies) == 1 else 0.0))   # duel: press harder
        reach = 1.6 + 2.6 * a_                         # how far from home I go after the ball
        slow = bv is not None and float(np.linalg.norm(bv)) < 0.12
        if b is not None and enemies and (np.linalg.norm(b - self.H) < reach or np.linalg.norm(b - me) < 0.8 or (slow and np.linalg.norm(b - me) < 3.5)):
            tgt = min(enemies, key=lambda k: (self.scoreboard.get(k, 26), np.linalg.norm(G.king(k) - b)))
            aim = self.gap_aim(tgt) + G.rot(self.team) @ self.jit; d = unitv(aim - b)
            behind = b - (G.BODY_R + G.BALL_R + 0.10) * d
            rel = me - b; along = rel @ d; lat = abs(rel[0] * d[1] - rel[1] * d[0])
            self.state = "attack"; self.counts["attacks"] += 1; self.target = tgt
            if along < -0.12 and lat < 0.16:                   # lined up behind the ball: push it through toward the target
                self.sub = f"ATTACK {G.TEAM[tgt]} ({self.scoreboard.get(tgt, 0)} bricks): push"
                return self._go(pose, b + 0.6 * d, face=aim, speed=PUSH)
            if along > -0.05 and lat < G.BODY_R + 0.12:        # ball between me and my approach spot: go round it
                side = np.array([-d[1], d[0]]); side = side if rel @ side > 0 else -side
                wp = b + side * (G.BODY_R + 0.25) - d * 0.15
            else: wp = behind
            self.sub = f"ATTACK {G.TEAM[tgt]}: line up behind the ball"
            return self._go(pose, wp, face=b)
        g = self.K + GUARD * unitv((b if b is not None else np.zeros(2)) - self.K)
        g = self._ring(g); self.state = "hold"
        if b is None:
            self.sub = "SCAN: ball lost"; self.counts["scans"] += 1
            if np.linalg.norm(g - me) < 0.2: return {"rx": 0.7}
            return self._go(pose, g)
        self.sub = "HOLD guard spot"; return self._go(pose, g, face=b)

    def gap_aim(self, k):
        """aim point on castle k: its king, through the widest gap of broken bricks when there is one (public scoreboard)."""
        gaps = self.gaps.get(k) if hasattr(self, "gaps") else None
        if not gaps: return G.king(k)
        return np.asarray(gaps[0], float)

    def _ring(self, g):
        """guard spots stay on the floor in front of my castle (wall-clear)."""
        for _ in range(20):
            c, n = obstacle(g)
            if c >= G.CLEAR + 0.02: break
            g = g + n * (G.CLEAR + 0.03 - c)
        return g

    def _go(self, pose, goal, face=None, speed=WALK):
        x, y, yaw = pose; d = np.asarray(goal, float) - np.array([x, y]); dist = float(np.linalg.norm(d))
        sp = speed * min(1.0, dist / 0.35) if dist > 0.04 else 0.0
        h = math.atan2(d[1], d[0]); c, s = math.cos(yaw), math.sin(yaw)
        vx, vy = (c * math.cos(h) + s * math.sin(h)) * sp, (-s * math.cos(h) + c * math.sin(h)) * sp
        cmd = {"ly": -float(vx), "lx": -float(vy)}
        fa = math.atan2(face[1] - y, face[0] - x) if face is not None else (h if dist > 0.2 else None)
        if fa is not None:
            e = wrap(fa - yaw)
            if abs(e) > 0.18: cmd["rx"] = (-1 if e > 0 else 1) * min(0.8, 0.45 + 0.4 * abs(e))
        return cmd

    def _guard(self, cmd):
        spin = abs(cmd.get("rx", 0)) > 0.1 and abs(cmd.get("lx", 0)) + abs(cmd.get("ly", 0)) < 0.08
        if not spin: self.spin_t0 = None; return cmd
        if self.spin_t0 is None: self.spin_t0 = self.t
        if self.t - self.spin_t0 <= SPIN_MAX: return cmd
        self.spin_t0 = None; self.counts["unspins"] += 1; self.sub = "SPIN WATCHDOG: step"
        return {"ly": -0.3}

    def _safety(self, cmd, pose):
        """FINAL SAFETY LAYER: look ahead LOOK m along the commanded walk; a wall / castle within CLEAR there removes the
        into-obstacle part of the step (slide along it)."""
        x, y, yaw = pose; vxb, vyb = -cmd.get("ly", 0.0), -cmd.get("lx", 0.0)
        c, s = math.cos(yaw), math.sin(yaw); w = np.array([c * vxb - s * vyb, s * vxb + c * vyb]); sp = float(np.linalg.norm(w))
        if sp < 0.03: return cmd
        me = np.array([x, y]); changed = False
        for r in (0.15, 0.3, LOOK):
            q = me + w / sp * r; cl, n = obstacle(q)
            if cl < G.CLEAR - 0.05 and w @ n < 0:
                w = w - (w @ n) * n; changed = True
                if float(np.linalg.norm(w)) < 0.03: break
        if not changed: return cmd
        self.counts["safety"] += 1
        out = dict(cmd); out["ly"] = -float(c * w[0] + s * w[1]); out["lx"] = -float(-s * w[0] + c * w[1]); return out


def unit():
    out = []
    b = Brain("A", aggression=0.6); b.reset(1.0); b.pose = G.home("A")[:2] + (0.0,)
    # ball rolling at RED's castle -> DEFEND on the guard ring between ball and king
    K = G.king("A"); p0 = K + 2.5 * (-G.U["A"]) + np.array([0.3, 0.0]); v0 = -0.45 * unitv(p0 - K)
    for i in range(5): b.observe(1.0 + 0.1 * i, dict(ball=[dict(xy=p0 + v0 * 0.1 * i, rng=1.0)]), b.pose)
    b.decide(b.pose); assert b.state == "defend", (b.state, b.sub)
    out.append("ball heading at my castle -> DEFEND")
    # ball near me, not threatening -> ATTACK the weakest enemy
    c = Brain("A", aggression=0.6); c.reset(1.0); c.scoreboard = {"A": 26, "B": 5, "C": 20}; pose = (0.0, -0.8, 0.0)
    for i in range(5): c.observe(1.0 + 0.1 * i, dict(ball=[dict(xy=np.array([0.3, -0.5]), rng=0.4)]), pose)
    c.opening = None; c.decide(pose); assert c.state == "attack" and c.target == "B", (c.state, c.sub)
    out.append("loose ball -> ATTACK the weakest castle")
    # safety: walking into the wall is turned into a slide
    s = Brain("A"); s.reset(1.0); x, y = np.array([0.0, -G.R_IN + 0.33]); cmd = s._safety({"ly": 0.0, "lx": -0.45}, (x, y, 0.0))
    assert -cmd["lx"] > -0.05, cmd                   # lx<0 = walk +y body-left; heading pose yaw 0, wall below at -y -> lx>0 means walk -y
    cmd = s._safety({"ly": 0.0, "lx": 0.45}, (x, y, 0.0)); assert abs(cmd["lx"]) < 0.05, cmd
    out.append("safety layer: no walking into the wall")
    # 3-way fairness: rotated situations give rotated commands
    res = []
    for k in G.KEYS:
        R = G.rot(k); bb = Brain(k, aggression=0.6); bb.reset(1.0); hp = R @ np.array([0.2, -0.9]); pose = (hp[0], hp[1], G.ANG[k] - G.ANG["A"])
        for i in range(5): bb.observe(1.0 + 0.1 * i, dict(ball=[dict(xy=R @ np.array([0.4, -0.6]), rng=0.4)]), pose)
        bb.opening = None; res.append((bb.state, np.round(list(bb.decide(pose).values()), 3).tolist(), bb.decide(pose) and bb.state))
    assert res[0][1] == res[1][1] == res[2][1], res
    out.append("3-way mirror: identical commands in rotated situations")
    return out
