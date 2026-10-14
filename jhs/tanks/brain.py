"""Crab Tanks brain (SIM-ONLY VISION CONCEPT): hunt, take cover, lead the shot, dodge incoming shots.
Each crab sees only with its own camera + dToF + pose. Turret aim is independent of walking (within the camera FOV).
Leg clearance: half-span 0.25 m + 0.05 m margin = INFL 0.30 m (same as soccer). Mirror-symmetric."""
from __future__ import annotations
import math, os
import numpy as np
from scipy import ndimage
from jhs.soccer import vision
from jhs.tanks.explore import Memory, CRAB_R, INFL, CELL, XS, YS, CX, CY, NX, NY, INSIDE, cell
from jhs.soccer.brain import wrap, fam_mask, colour_blobs, FAM, TEAMFAM, OTHER, WALK
from jhs.tanks.field import FX, FY, wall_clear, wall_normal, piece_dist
from . import arena as A

FOV = math.radians(70)             # onboard camera half-width of view is ~35 deg each side; keep aim inside
SCAN_MAX, SCAN_WATCH = 10.0, 8.0
SHOT_HEAD_R = 0.045               # drawn projectile head radius (jhs/tanks/fx.py), known to the crabs as a game rule
SHOT_LEAD = os.environ.get("TK_LEAD", "1") == "1"


def perceive(team, img, selfmask, cam_pos, cam_R, cam, tof, tof_pos, tof_R, tof_cam):
    """-> dict(opp=[dets], shots=[dets], obst=Nx3). Own sensors only."""
    opp_fam = TEAMFAM[OTHER[team]]
    cb = colour_blobs(img, cam_pos, cam_R, cam, selfmask, opp_fam)
    opp = [d for d in cb if d["kind"] == "crab"]
    shots = []
    # a projectile is a small round glowing-cyan blob in the upper half of the crab's view (drawn as a glowing sphere)
    h, s, v = vision.hsv(img)
    m = (h >= 165) & (h <= 195) & (s >= 0.45) & (v >= 0.55)        # glowing cyan: no other cyan in the arena
    m &= ~selfmask; m[: img.shape[0] // 5, :] = False     # skip the sky strip
    lab, n = ndimage.label(m)
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None: continue
        blob = lab[sl] == i + 1; npx = int(blob.sum())
        if not (4 <= npx <= 1500): continue
        ys, xs = np.nonzero(blob); ys = ys + sl[0].start; xs = xs + sl[1].start
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if max(bw, bh) > 2.5 * min(bw, bh) or npx < 0.4 * bw * bh: continue      # round, filled
        uc, vc = int(np.median(xs)), int(np.median(ys))
        d = cam_R @ np.array([(uc + 0.5 - cam.W / 2) / cam.f, -(vc + 0.5 - cam.H / 2) / cam.f, -1.0])
        Dn = d / max(1e-6, np.linalg.norm(d))
        # range from the head's known size (its vertical extent; the trail only stretches it sideways), which stays well
        # conditioned far away, unlike intersecting a near-horizontal ray with the flight-height plane
        t = (2 * SHOT_HEAD_R) * cam.f / max(1.0, float(bh)) * float(np.linalg.norm(d))
        if t <= 0.05 or t > 7.5: continue
        g = cam_pos + t * Dn
        if abs(g[0]) > FX + 0.3 or abs(g[1]) > FY + 0.3: continue
        # reject the other crab's body (big colour blob nearby)
        if any(np.hypot(g[0] - o["xy"][0], g[1] - o["xy"][1]) < 0.25 for o in opp): continue
        shots.append(dict(xy=np.array([g[0], g[1]], float), rng=float(t),
                          bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), npx=npx))
    keep = []                                  # head + trail: one shot (the largest blob of a group within 0.3 m)
    for q in sorted(shots, key=lambda q: -q["npx"]):
        if all(np.linalg.norm(q["xy"] - k_["xy"]) > 0.3 for k_ in keep): keep.append(q)
    obst = vision.tof_obstacles(tof, tof_pos, tof_R, tof_cam, rmax=0.7) if tof is not None else np.zeros((0, 3))
    return dict(opp=opp, shots=keep, obst=obst)


# ---- randomized opening strategies (same menu + same odds for RED and BLUE; layout and pockets fixed) ----
OPENERS = ("PEEK_NEAR", "PEEK_FAR", "HOLD", "PUSH_MID")
OPEN_P = (0.25, 0.25, 0.25, 0.25)
JIT_XY, JIT_YAW = 0.06, math.radians(6)     # start-pose jitter inside the pocket (still behind the bracket, roughly facing the centre)
HOLD_S, OPEN_MAX_S = 9.0, 16.0              # ambush wait / opener cap before the normal search takes over


def roll_opening(rng, team, mode="random", locked=None):
    """one crab's opening for one round. Same menu and odds for both teams; each crab rolls independently."""
    op = locked if (mode == "locked" and locked in OPENERS) else OPENERS[int(rng.choice(len(OPENERS), p=OPEN_P))]
    jx, jy = rng.uniform(-JIT_XY, JIT_XY, 2); jyaw = rng.uniform(-JIT_YAW, JIT_YAW)
    return dict(opener=op, scan=int(rng.choice((-1, 1))), bias=int(rng.choice((-1, 1))), jit=(float(jx), float(jy), float(jyaw)),
                side=int(np.sign(jy)) or int(rng.choice((-1, 1))))


def roll_round(rng, cfg=None):
    """both crabs' openings for one round. cfg = {mode: random|locked, A: name, B: name}. A and B draw from the same rng in turn."""
    cfg = cfg or {}; mode = str(cfg.get("mode", "random")).lower()
    return {k: roll_opening(rng, k, mode, cfg.get(k)) for k in ("A", "B")}


def jittered_start(team, jit, base=None):
    sx, sy, sh = base if base is not None else A.START[team]
    return (round(sx + jit[0], 3), round(sy + jit[1], 3), round(sh + math.degrees(jit[2]), 1))


def opener_waypoints(team, opener, side):
    """team-relative waypoints (x mirrored for BLUE): list of (x, y, face_deg or None). 'side' is the y side of the near bracket end."""
    dx = -1.0 if team == "A" else 1.0; fwd = 0.0 if team == "A" else 180.0
    W = lambda x, y, f: (dx * x, y, f)
    if opener == "PEEK_NEAR" or opener == "PEEK_FAR":
        s = side if opener == "PEEK_NEAR" else -side
        return [W(2.80, 1.12 * s, None), W(1.85, 1.20 * s, fwd)]
    if opener == "HOLD":
        return [W(2.85, 0.42 * side, math.degrees(math.atan2(1.0 * side, -dx * 0.6)))]
    return [W(2.80, 1.12 * side, None), W(1.85, 1.20 * side, None), W(0.55, 0.0, fwd)]   # PUSH_MID: out round the end, into the mid gap


def _seg_blocked(a, b, n=8):
    """a fixed wall piece lies between a and b (a shot there cannot reach my body)."""
    return any(piece_dist(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) < 0.01 for k in range(1, n))


class Brain:
    def __init__(self, team, rng=None, pushable=False, bounce=True):
        self.team = team; self.dirx = +1 if team == "A" else -1; self.pushable = bool(pushable); self.bounce = bool(bounce)
        self.mem = Memory(); self.rng = rng or np.random.default_rng(0)
        self.op_name = ""
        self.counts = {"shots": 0, "hits_scored": 0, "dodges": 0, "searches": 0, "covers": 0, "hunts": 0, "explores": 0, "backoffs": 0}
        self.reset(kickoff=True, t=0.0)

    def reset(self, kickoff, t):
        self.opp = None; self.opp_t = -1e9; self.opp_v = np.zeros(2)
        self.shots = []; self.shot_t = -1e9
        self.turret = 0.0; self.last_fire = -1e9; self.own_in_flight = False
        self.state = "kickoff"; self.sub = ""; self.t = t; self.hold = None; self.hold_until = 0.0
        self.stuck_ref = None; self.scan_acc = 0.0; self.scan_prev = None; self.scan_t0 = t
        self.searching = False; self.exploring = False; self.exp_goal = None; self.exp_path = []; self.exp_t = -1e9
        self.wait_until = t + 0.6; self.obst = np.zeros((0, 3)); self.last_cmd = {}; self.wd = (t, np.zeros(2))
        self.cover = None; self.stun_until = 0.0; self.ret_spot = None; self.own = None; self.dodge_hold = -1e9; self.dodge_goal = None; self.threat_path = np.zeros((0, 3))
        opp_start = A.START["B" if self.team == "A" else "A"]           # rules / map knowledge: where the opponent starts
        if not hasattr(self, "bel"): self.bel = OppBelief()
        self.bel.reset(opp_start[:2] if kickoff else None, sigma=0.3); self.bel_t = t; self.sgoal = None; self.spath = []; self.splan_t = -1e9
        self.opening = None; self.op_wps = []; self.op_t0 = t; self.op_arrive = None; self.op_path = []; self.op_plan_t = -1e9

    def set_opening(self, o, t):
        """start a rolled opener (dict from roll_opening). Ends at the first sighting, a hit/stun, or after OPEN_MAX_S."""
        self.op_name = o["opener"]; self.opening = dict(o); self.op_wps = list(opener_waypoints(self.team, o["opener"], o["side"])); self.op_t0 = t; self.op_arrive = None
        self.op_path = []; self.op_plan_t = -1e9
        self.bel.p *= np.clip(1.0 + 0.25 * o["bias"] * CY / FY, 0.5, 1.5); self.bel._norm()      # slight side bias in the belief

    def _opener(self, pose):
        x, y, yaw = pose; me = np.array([x, y]); t = self.t; o = self.opening; name = o["opener"]
        if t - self.op_t0 > OPEN_MAX_S or not self.op_wps: self.opening = None; return None
        gx, gy, face = self.op_wps[0]; g = np.array([gx, gy]); dg = float(np.linalg.norm(g - me))
        self.state = "opener"
        if dg < 0.15:
            if len(self.op_wps) > 1: self.op_wps.pop(0); self.op_path = []; return self._opener(pose)
            if self.op_arrive is None: self.op_arrive = t
            dwell = HOLD_S if name == "HOLD" else 2.5
            if t - self.op_arrive > dwell: self.opening = None; return None
            # first look: sweep the turret to the rolled side first, then back
            base = math.radians(face if face is not None else math.degrees(yaw)); ph = (t - self.op_arrive) / 1.2
            self._aim(pose, base + o["scan"] * 0.7 * math.sin(ph))
            self.sub = f"{name}: {'ambush, waiting' if name == 'HOLD' else 'peeking'} (scan {'left' if o['scan'] > 0 else 'right'} first)"
            return self._drive(pose, me, face_yaw=base)
        if t - self.op_plan_t > 1.0 or not self.op_path:
            self.op_path = self.mem.plan(me, g, self.pushable, self._opp_block, sign=self.dirx) or [g]; self.op_plan_t = t
        self._aim(pose, math.atan2(g[1] - y, g[0] - x) + o["scan"] * 0.35)
        self.sub = f"{name}: to ({gx:.1f}, {gy:.1f}) {dg:.1f} m"
        return self._follow(pose, self.op_path, g, "opener")

    def observe(self, t, per, pose):
        self.t = t; self.obst = per["obst"]; x, y, yaw = pose
        self.mem.t = t
        self.mem.add_obstacles(per["obst"], None, (per["opp"][0]["xy"] if per["opp"] else self.opp if self.opp is not None and t - self.opp_t < 1.5 else None))
        self.mem.mark_view(pose)
        # opponent belief: spread, then clear what I can see (or reset to a sighting)
        self.bel.spread(max(0.0, t - self.bel_t)); self.bel_t = t
        if per["opp"]: self.bel.reset(min(per["opp"], key=lambda d: d["rng"])["xy"], sigma=0.12); self.opening = None
        else: self.bel.seen_empty(self.mem.visible(pose, OppBelief.BEL_R, OppBelief.BEL_HFOV))
        if per["opp"]:
            d = min(per["opp"], key=lambda d: d["rng"]); z = np.asarray(d["xy"], float)
            if self.opp is not None and 0.05 < t - self.opp_t < 1.5:
                v = (z - self.opp) / (t - self.opp_t)
                if np.linalg.norm(v) < 1.5: self.opp_v = 0.6 * self.opp_v + 0.4 * v
            self.opp = z; self.opp_t = t
        # my own shot: dead-reckoned with the arena rules (map knowledge) so I do not dodge my own projectile
        if self.own is not None:
            if not self.own_in_flight: self.own = None
            else:
                dt = max(0.0, t - self.own["t"]); n = max(1, int(dt / 0.02))
                for _ in range(n):
                    x_, y_ = self.own["x"] + self.own["vx"] * dt / n, self.own["y"] + self.own["vy"] * dt / n
                    self.own["x"], self.own["y"], self.own["vx"], self.own["vy"], _b = A.reflect(x_, y_, self.own["vx"], self.own["vy"])
                self.own["t"] = t
        def _tracked(d):
            return any(np.linalg.norm(q["xy"] + q["v"] * (t - q["t"]) - d["xy"]) < 0.45 for q in self.shots)
        prev = getattr(self, "prev_raw", None) or (t, [])
        def _vel2(d):                                       # frame-to-frame velocity of a raw shot detection (None if no match)
            dt_ = t - prev[0]
            if not (0.03 < dt_ < 0.35): return None
            c = [z for z in prev[1] if np.linalg.norm(z - d["xy"]) < 0.25 + A.SHOT_SPEED * dt_]
            if not c: return None
            z = min(c, key=lambda z: np.linalg.norm(z - d["xy"])); v_ = (d["xy"] - z) / dt_
            return v_ if np.linalg.norm(v_) < 1.5 else None
        def _mine(d):
            if self.own is None or math.hypot(d["xy"][0] - self.own["x"], d["xy"][1] - self.own["y"]) > 0.35: return False
            if _tracked(d): return False                    # an enemy shot I was already tracking stays a threat
            v2 = _vel2(d); vo_ = np.array([self.own["vx"], self.own["vy"]])
            if v2 is not None and float(v2 @ vo_) < 0: return False     # it moves against my shot's direction: not mine (seed 33)
            u = np.array([d["xy"][0] - x, d["xy"][1] - y]); vo = np.array([self.own["vx"], self.own["vy"]])
            return float(u @ vo) > -0.05 * float(np.linalg.norm(u)) * A.SHOT_SPEED   # mine flies away from me; one closing in is not mine
        dets = [d for d in per["shots"] if not _mine(d)]
        self.prev_raw = (t, [np.asarray(d["xy"], float).copy() for d in per["shots"]])
        # match vision shots to previous ones to estimate velocity
        seen = []
        for d in dets:
            z = d["xy"]; pred = None
            if self.shots:
                cand = min(self.shots, key=lambda s: np.linalg.norm(s["xy"] + s["v"] * (t - s["t"]) - z))
                if np.linalg.norm(cand["xy"] + cand["v"] * (t - cand["t"]) - z) < 0.45:
                    dt = max(0.05, t - cand["t"]); v = (z - cand["xy"]) / dt
                    if np.linalg.norm(v) < 1.2: pred = 0.5 * cand["v"] + 0.5 * v
                    else: pred = cand["v"]
            if pred is None:
                v2 = _vel2(d)
                if v2 is not None and np.linalg.norm(v2) > 0.25: pred = v2 / np.linalg.norm(v2) * A.SHOT_SPEED
            if pred is None and self.opp is not None and t - self.opp_t < 3.0:     # first sighting: assume it came from the opponent
                u = z - self.opp; nu = float(np.linalg.norm(u))
                if nu > 0.1: pred = u / nu * A.SHOT_SPEED
            seen.append(dict(xy=z.copy(), v=np.asarray(pred if pred is not None else np.zeros(2), float), t=t, rng=d["rng"], bbox=d["bbox"]))
        self.shots = seen; self.shot_t = t if seen else self.shot_t

    def begin_return(self, spot, t):
        self.ret_spot = (float(spot[0]), float(spot[1]), math.radians(spot[2])); self.ret_path = []; self.ret_plan_t = -1e9
        self.state = "return"; self.sub = "returning to start"; self.t = t; self.stuck_ref = None; self.hold = None

    def return_cmd(self, pose):
        dcmd = self._dodge(pose)                 # a seen shot outranks walking back too
        if dcmd is not None: return dcmd
        x, y, yaw = pose; me = np.array([x, y]); sx, sy, sh = self.ret_spot; goal = np.array([sx, sy])
        dg = float(np.linalg.norm(goal - me))
        if dg < 0.12:
            e = wrap(sh - yaw); self.sub = "at start, waiting" if abs(e) < 0.2 else "turning to face the opponent"
            return {"rx": (-1 if e > 0 else 1) * (0.45 + 0.3 * min(1.0, abs(e)))} if abs(e) > 0.12 else {}
        if self.t - self.ret_plan_t > 1.0 or not self.ret_path:
            self.ret_path = self.mem.plan(me, goal, self.pushable, self._opp_block, sign=self.dirx) or [me, goal]; self.ret_plan_t = self.t
        while len(self.ret_path) > 1 and (np.linalg.norm(self.ret_path[0] - me) < 0.22 or self.mem.clear_line(me, self.ret_path[1])): self.ret_path.pop(0)
        return self._drive(pose, self.ret_path[0] if dg > 0.3 else goal, face_yaw=sh if dg < 0.35 else None, speed=WALK if dg > 0.4 else 0.3)

    def _opp_block(self, x, y):
        return self.opp is not None and self.t - self.opp_t < 3.0 and math.hypot(x - self.opp[0], y - self.opp[1]) < 0.4

    def _drive(self, pose, goal_xy, face_yaw=None, speed=WALK, avoid=True):
        x, y, yaw = pose; me = np.array([x, y])
        d = np.asarray(goal_xy) - me; n = float(np.linalg.norm(d))
        want = face_yaw if face_yaw is not None else math.atan2(d[1], d[0])
        e = wrap(want - yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        sp = min(speed, max(0.18, n * 1.4)) if n > 0.04 else 0.0
        vx, vy = (c * d[0] + s * d[1]) / max(n, 1e-6) * sp, (-s * d[0] + c * d[1]) / max(n, 1e-6) * sp
        if face_yaw is None and abs(e) > 1.3: vx, vy = 0.0, 0.0
        if avoid: vx, vy = self._avoid(pose, vx, vy)
        p = {"ly": -float(np.clip(vx, -0.45, 0.45)), "lx": -float(np.clip(vy, -0.45, 0.45))}
        if abs(e) > 0.06: p["rx"] = (-1 if e > 0 else 1) * (0.5 + 0.3 * min(1.0, abs(e)))
        return p

    def _avoid(self, pose, vx, vy):
        x, y, yaw = pose; me = np.array([x, y]); c, s = math.cos(yaw), math.sin(yaw)
        pts = [p[:2] for p in self.obst[::3]]; rads = [0.38] * len(pts)
        if self.opp is not None and self.t - self.opp_t < 1.5: pts.append(self.opp); rads.append(0.45)
        rep = np.zeros(2)
        for p, r in zip(pts, rads):
            dv = np.asarray(p) - me; n = float(np.linalg.norm(dv))
            if n < 1e-3 or n > r: continue
            dl = np.array([c * dv[0] + s * dv[1], -s * dv[0] + c * dv[1]])
            if dl @ np.array([vx, vy]) <= 0: continue
            rep -= dl / n * (r - n) / r
        wc = wall_clear(x, y)
        if wc < INFL:      # outer wall or a fixed layout piece (map knowledge): push away along the wall normal
            gx, gy = wall_normal(x, y); g = np.array([c * gx + s * gy, -s * gx + c * gy]); g /= max(1e-6, np.linalg.norm(g))
            if g @ np.array([vx, vy]) < 0: rep += g * (INFL - wc) / INFL
        if not np.any(rep): return vx, vy
        return vx + 0.9 * rep[0], vy + 0.9 * rep[1]

    def _stuck(self, pose, cmd):
        me = np.array(pose[:2]); walking = abs(cmd.get("ly", 0)) + abs(cmd.get("lx", 0)) > 0.12
        if self.stuck_ref is None or not walking: self.stuck_ref = (self.t, me); return False
        if self.t - self.stuck_ref[0] < 2.5: return False
        moved = float(np.linalg.norm(me - self.stuck_ref[1])); self.stuck_ref = (self.t, me)
        return moved < 0.05

    def _follow(self, pose, path, goal, label):
        x, y, yaw = pose; me = np.array([x, y])
        while len(path) > 1 and (np.linalg.norm(path[0] - me) < 0.22 or self.mem.clear_line(me, path[1])): path.pop(0)
        wp = path[0] if path else goal
        if self.pushable and self.mem.through_obstacle(path):
            self.state = "push_obst"; self.sub = "PUSHING OBSTACLE (" + label + ")"
            return self._drive(pose, wp, avoid=False)
        return self._drive(pose, wp)

    def _explore(self, pose):
        x, y, yaw = pose; me = np.array([x, y])
        if self.exp_goal is None or self.t - self.exp_t > 6.0 or np.linalg.norm(self.exp_goal - me) < 0.25:
            if self.exp_goal is not None and np.linalg.norm(self.exp_goal - me) < 0.25:
                self.exp_goal = None; self.exploring = False; self.scan_acc = 0.0; self.scan_prev = yaw; self.scan_t0 = self.t
                self.searching = True; self.state = "search"; self.sub = "scan: looking for the opponent"; return {"rx": 0.62}
            g, _ = self.mem.next_view(me, sign=self.dirx)
            if g is None: self.mem.seen[:] = -1e9; g, _ = self.mem.next_view(me, sign=self.dirx)
            if g is None or np.linalg.norm(g - me) < 0.5:
                for _ in range(50):
                    q = np.array([self.rng.uniform(-FX + 0.4, FX - 0.4), self.rng.uniform(-FY + 0.4, FY - 0.4)])
                    if np.linalg.norm(q - me) > 0.8 and wall_clear(*q) > 0.3: g = q; break
            self.exp_goal = g if g is not None else np.zeros(2); self.exp_t = self.t
            self.exp_path = self.mem.plan(me, self.exp_goal, self.pushable, self._opp_block, sign=self.dirx) or [self.exp_goal]
            self.counts["explores"] += 1
        self.state = "explore"; self.sub = f"EXPLORE: next best view ({self.exp_goal[0]:.1f}, {self.exp_goal[1]:.1f})"
        return self._follow(pose, self.exp_path, self.exp_goal, "exploring")

    HIT_R_ = 0.30
    DODGE_R = 0.38                   # react when a predicted pass comes this close (hit radius 0.18 + estimate error)
    SAFE = 0.36                      # a sidestep spot must clear the predicted path by this          # body + leg clearance a seen shot must miss by

    def _threat(self, me, horizon=3.5, dt=0.05):
        """the most urgent seen shot whose predicted path (wall / fixed-wall bounces included, arena rules) passes within DODGE_R:
        -> (time, miss, velocity at closest approach, position at closest approach) or None."""
        best = None
        for s in self.shots:
            if float(np.linalg.norm(s["v"])) < 0.1: continue
            x, y = float(s["xy"][0]), float(s["xy"][1]); vx, vy = float(s["v"][0]), float(s["v"][1]); bmin = (9.0, 0.0, (vx, vy), (x, y))
            path = [(0.0, x, y)]
            for i in range(int(horizon / dt)):
                x, y, vx, vy, _b = A.reflect(x + vx * dt, y + vy * dt, vx, vy)
                d = math.hypot(me[0] - x, me[1] - y); path.append(((i + 1) * dt, x, y))
                if d < bmin[0] and not (d < 0.6 and _seg_blocked((x, y), me)): bmin = (d, (i + 1) * dt, (vx, vy), (x, y))
            if bmin[0] < self.DODGE_R and (best is None or bmin[1] < best[0]):
                best = (bmin[1], bmin[0], np.array(bmin[2]), np.array(bmin[3])); self.threat_path = np.array(path)
        return best

    def _step_clear(self, g, tt):
        """how far a spot stays from the threatening shot's predicted path (up to just after it passes me)."""
        P = self.threat_path; P = P[P[:, 0] <= tt + 0.8]
        return float(np.min(np.hypot(P[:, 1] - g[0], P[:, 2] - g[1]))) if len(P) else 9.0

    def _dodge(self, pose):
        """quickest sidestep out of a seen shot's path keeping INFL from walls (or duck behind cover if nearer); held until it passed."""
        x, y, yaw = pose; me = np.array([x, y]); t = self.t
        th = self._threat(me)
        if th is None:
            if t < self.dodge_hold and self.dodge_goal is not None:
                self.state = "dodge"; self.sub = "DODGE: holding until the shot has passed"
                return self._drive(pose, self.dodge_goal, face_yaw=yaw, speed=0.55)
            self.dodge_goal = None; return None
        tt, miss, v, pc = th
        self.dodge_hold = t + tt + 0.3
        side = np.array([-v[1], v[0]]); side /= max(1e-6, np.linalg.norm(side))
        # candidates: perpendicular both ways first, then diagonal / forward / back steps; each must keep INFL from walls (on the
        # way too), be reachable before the shot arrives and clear the predicted path (bounces included) by SAFE.
        best, bc, fb, fbc = None, 1e9, None, -1.0; occ = self.mem.occ()
        fwd = v / max(1e-6, np.linalg.norm(v))
        dirs = [side, -side] + [math.cos(a) * side + math.sin(a) * fwd for a in np.radians((30, -30, 150, -150, 60, -60, 120, -120, 90, -90))]
        for k_, u in enumerate(dirs):
            u = np.asarray(u, float); u /= max(1e-6, np.linalg.norm(u))
            for dist in (0.15, 0.25, 0.35, 0.5, 0.65, 0.8):
                g = me + u * dist
                if wall_clear(*g) < INFL or occ[cell(g)] or wall_clear(*(me + u * dist * 0.5)) < INFL - 0.05: continue
                clr = self._step_clear(g, tt); reach = dist / 0.28
                if clr >= self.SAFE and reach <= tt + 0.1:
                    cost = dist + 0.02 * k_
                    if cost < bc: best, bc = g, cost
                    break
                if clr - 0.2 * max(0.0, reach - tt) > fbc: fb, fbc = g, clr - 0.2 * max(0.0, reach - tt)
        cover = self._cover_spot(me, pc)
        if cover is not None and float(np.linalg.norm(cover - me)) < bc and float(np.linalg.norm(cover - me)) / 0.28 <= tt:
            best, bc = np.asarray(cover, float), float(np.linalg.norm(cover - me))
        if best is None: best = fb if fb is not None else me; bc = float(np.linalg.norm(best - me))
        if self.dodge_goal is None or self.counts.get("_dodge_t", -1) != round(t - tt, 0):
            self.counts["dodges"] += 1
        self.dodge_goal = np.asarray(best, float)
        self.state = "dodge"; self.sub = f"DODGE: shot in {tt:.1f} s, stepping {bc:.2f} m"
        return self._drive(pose, self.dodge_goal, face_yaw=yaw, speed=0.55)       # strafe: no turning first

    def _cover_spot(self, me, threat_from):
        """a free cell behind a sensed obstacle, on the far side from the threat (own dToF memory only)."""
        occ = self.mem.occ(); best, bs = None, -1e9
        for i in range(0, occ.shape[0], 2):
            for j in range(0, occ.shape[1], 2):
                if not occ[i, j]: continue
                p = np.array([XS[i], YS[j]]); v = p - threat_from; n = float(np.linalg.norm(v))
                if n < 0.1: continue
                spot = p + v / n * 0.45
                if wall_clear(*spot) < INFL or np.linalg.norm(spot - me) < 0.3 or occ[cell(spot)]: continue
                sc = -float(np.linalg.norm(spot - me)) + 0.3 * n
                if sc > bs: best, bs = spot, sc
        return best

    def _aim(self, pose, want):
        """slewing the turret toward `want` (world yaw), clamped to the camera FOV around body yaw. Returns turret yaw."""
        x, y, yaw = pose
        # keep want inside [yaw - FOV/2, yaw + FOV/2]
        e = wrap(want - yaw); e = float(np.clip(e, -FOV / 2, FOV / 2)); target = yaw + e
        de = wrap(target - self.turret)
        step = float(np.clip(de, -A.TURRET_RATE * 0.1, A.TURRET_RATE * 0.1))   # vision tick ~0.1 s
        self.turret = wrap(self.turret + step)
        return self.turret

    def shot_clear(self, a, b):
        """a shot from a to b passes no fixed wall (map knowledge, raw geometry) and no sensed obstacle point mid-way."""
        a, b = np.asarray(a, float), np.asarray(b, float); L = float(np.linalg.norm(b - a))
        pts = self.obst[::3, :2] if len(self.obst) else np.zeros((0, 2))
        for f in np.linspace(0, 1, max(2, int(L / 0.05))):
            p = a + (b - a) * f
            if piece_dist(*p) < 0.05: return False
            if 0.45 < f * L < L - 0.45 and len(pts) and float(np.min(np.linalg.norm(pts - p, axis=1))) < 0.1: return False
        return True

    def fire_ready(self, pose, aim):
        """True when cooldown done, no own shot in flight, and the turret is close enough to aim."""
        if self.own_in_flight or self.t - self.last_fire < A.COOLDOWN: return False
        if abs(wrap(aim - self.turret)) > 0.12: return False
        return True

    def decide(self, pose):
        x, y, yaw = pose; me = np.array([x, y]); t = self.t
        if t < self.wait_until: self.state, self.sub = "kickoff", "ready"; return {}, None
        if t < self.stun_until: self.state, self.sub = "stunned", "STUNNED"; self.opening = None; return {}, None
        dcmd = self._dodge(pose)                 # evasion overrides everything (incl. a stuck back-off in progress)
        if dcmd is not None: self.hold = None; self.stuck_ref = None; return dcmd, None
        if self.hold is not None:
            if t < self.hold_until: return dict(self.hold), None
            self.hold = None
        cmd, fire = self._decide(pose)
        if self._stuck(pose, cmd):
            self.counts["backoffs"] += 1; self.state, self.sub = "backoff", "stuck: backing off"
            side = 1 if self.rng.random() < 0.5 else -1
            self.hold = {"ly": 0.35, "lx": 0.25 * side}; self.hold_until = t + 1.2; self.stuck_ref = None
            return dict(self.hold), None
        self.last_cmd = cmd
        return cmd, fire

    def _decide(self, pose):
        x, y, yaw = pose; me = np.array([x, y]); t = self.t; fire = None
        # ---- DODGE (highest priority, above fire / hunt / cover / search / walk-back) ----
        dcmd = self._dodge(pose)
        if dcmd is not None: return dcmd, None
        # ---- hunt / cover / search ----
        have_opp = self.opp is not None and t - self.opp_t < 2.5
        if have_opp:
            self.searching = False; self.exploring = False
            # take cover when a shot is coming and cover is nearer than a dodge
            threat_from = self.opp
            cover = self._cover_spot(me, threat_from) if self.shots else None
            if cover is not None and np.linalg.norm(cover - me) < 1.6 and any(A.threat(me, s["xy"], s["v"]) for s in self.shots):
                self.counts["covers"] += 1; self.state = "cover"; self.sub = "COVER: behind an obstacle"
                cmd = self._drive(pose, cover)
            else:
                # hunt: close to a stand-off range (~1.4 m) while keeping LOS
                d = self.opp - me; n = float(np.linalg.norm(d)); want_r = 1.4
                if n > want_r + 0.25: goal = self.opp - d / n * want_r
                elif n < want_r - 0.35: goal = me - d / n * 0.4     # back off a little
                else: goal = me + np.array([-d[1], d[0]]) / n * 0.25 * self.dirx   # circle
                self.counts["hunts"] += 1; self.state = "hunt"; self.sub = f"HUNT: opponent at {n:.1f} m"
                cmd = self._drive(pose, goal)
            # aim + fire
            aim, lead_t = A.lead_angle(me, self.opp, self.opp_v) if SHOT_LEAD else (math.atan2(*(self.opp - me)[::-1]), None)
            self._aim(pose, aim)
            if self.fire_ready(pose, aim) and self.shot_clear(me, self.opp):
                fire = dict(yaw=self.turret, lead=lead_t); self.last_fire = t; self.own_in_flight = True; self.counts["shots"] += 1
                self.own = dict(x=x + A.MUZZLE * math.cos(self.turret), y=y + A.MUZZLE * math.sin(self.turret),
                                vx=A.SHOT_SPEED * math.cos(self.turret), vy=A.SHOT_SPEED * math.sin(self.turret), t=t)
                self.sub += " · FIRE"
            return cmd, fire

        if self.opening is not None and not have_opp:
            ocmd = self._opener(pose)
            if ocmd is not None: return ocmd, None
        # ---- search (opponent not seen): go to and look at the likeliest spot of my opponent belief ----
        return self._search(pose), None

    def _search(self, pose):
        x, y, yaw = pose; me = np.array([x, y]); t = self.t
        g, conf = self.bel.peak(me, sign=self.dirx)
        if self.sgoal is None or t - self.splan_t > 2.0 or np.linalg.norm(g - self.sgoal) > 0.5:
            self.sgoal = g; self.splan_t = t
            self.spath = self.mem.plan(me, g, self.pushable, self._opp_block, sign=self.dirx) or [g]
            self.counts["explores"] += 1
        self._aim(pose, math.atan2(g[1] - y, g[0] - x))             # turret toward the likeliest spot
        dg = float(np.linalg.norm(g - me)); self.counts["searches"] += 1
        if dg < 0.6:                                                   # close: just face it (my view clears it, the peak moves on)
            self.state = "search"; self.sub = f"SEARCH: looking at the likeliest spot ({conf:.0%})"
            return self._drive(pose, me, face_yaw=math.atan2(g[1] - y, g[0] - x))
        self.state = "search"; self.sub = f"SEARCH: belief peak ({g[0]:.1f}, {g[1]:.1f}) {dg:.1f} m"
        return self._follow(pose, self.spath, g, "searching")

HIT_R = A.HIT_R


_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


class OppBelief:
    """Where the opponent can be (own vision only). p over the arena grid, sums to 1. Seen-and-empty floor drops to ~0, floor hidden
    behind walls / obstacles keeps its value, and every tick the belief spreads at the opponent's top walking speed (so it regrows from
    the last sighting and from the edges of the unseen region). A sighting resets it to that point. Mirror-symmetric (point-reflection
    tie-break for the peak). SIM-ONLY VISION CONCEPT."""
    VMAX = 0.35                      # m/s: the crab's top walking speed
    BEL_R, BEL_HFOV = 3.5, 1.6       # range / full view angle within which a crab in view would be detected
    FREE = INSIDE & (np.vectorize(lambda x, y: __import__("jhs.tanks.field", fromlist=["wall_clear"]).wall_clear(x, y))(CX, CY) >= 0.12)

    def __init__(self, at=None):
        self.p = np.zeros((NX, NY)); self.reset(at); self.acc = 0.0

    def reset(self, at=None, sigma=0.15):
        if at is None: self.p = self.FREE.astype(float)
        else: self.p = np.exp(-((CX - at[0]) ** 2 + (CY - at[1]) ** 2) / (2 * sigma ** 2)) * self.FREE
        self._norm()

    def _norm(self):
        s = float(self.p.sum())
        if s < 1e-9: self.p = self.FREE.astype(float); s = float(self.p.sum())
        self.p /= s

    def spread(self, dt):
        """the opponent walks at most VMAX: the belief front moves one cell per CELL / VMAX s (a grey dilation, so growth is
        linear in time from the last sighting and from the edges of the unseen region, not diffusive)."""
        self.acc += dt; n = int(self.acc * self.VMAX / CELL)
        if n < 1: return
        self.acc -= n * CELL / self.VMAX
        for _ in range(min(n, 20)):
            self.p = np.maximum(self.p, 0.9 * ndimage.maximum_filter(self.p, footprint=_CROSS, mode="constant")) * self.FREE
        self._norm()

    def seen_empty(self, vis):
        self.p[vis] *= 0.02; self._norm()

    def peak(self, me, sign=1, dist_w=0.15):
        """likeliest spot, smoothed, slightly preferring near ones; ties broken team-relative (mirror-symmetric)."""
        sm = ndimage.gaussian_filter(self.p, 2.0, mode="constant")
        sc = sm / max(1e-12, sm.max()) - dist_w * np.hypot(CX - me[0], CY - me[1]) / (2 * FX)
        sc = np.where(self.FREE, sc, -1e9)
        if sign < 0: i, j = np.unravel_index(np.argmax(sc[::-1, ::-1]), sc.shape); i, j = NX - 1 - i, NY - 1 - j
        else: i, j = np.unravel_index(np.argmax(sc), sc.shape)
        return np.array([CX[i, j], CY[i, j]]), float(sm[i, j] / max(1e-12, sm.max()))

    def coarse(self, n=4):
        """downsampled heat (0..1) for the overlay minimap."""
        h = self.p[: NX // n * n, : NY // n * n].reshape(NX // n, n, NY // n, n).sum(axis=(1, 3))
        return np.round(h / max(1e-12, h.max()), 2)


def unit():
    TB_wrap = wrap
    out = []
    b = Brain("A", np.random.default_rng(1))
    # lead + fire when the opponent is in view, clear LOS
    b.reset(True, 1.0); b.opp = np.array([0.95, 0.0]); b.opp_t = 1.0; b.opp_v = np.array([0.0, 0.3]); b.t = 2.0; b.wait_until = 0
    b.turret = 0.0; b.last_fire = -10
    # seed a clear memory so clear_line passes
    cmd, fire = b._decide((0.0, 0.0, 0.0))
    assert b.state in ("hunt", "cover"), b.state; out.append(f"hunt state={b.state}")
    # fire after a few aim ticks
    fired = False
    for i in range(20):
        b.t = 2.0 + 0.1 * i; cmd, fire = b._decide((0.0, 0.0, 0.0))
        if fire: fired = True; break
    assert fired, "should fire"; out.append("fires with a lead angle")
    # dodge overrides firing and hunting: opponent in view, turret on target, cooldown ready -- and a shot coming at me
    b.reset(True, 1.0); b.wait_until = 0; b.t = 3.0; b.opp = np.array([0.95, 0.0]); b.opp_t = 3.0; b.turret = 0.0; b.last_fire = -10
    b.shots = [dict(xy=np.array([0.9, 0.4]), v=np.array([-A.SHOT_SPEED, -0.25]), t=b.t, rng=1.0, bbox=(0, 0, 1, 1))]
    cmd, fire = b.decide((0.0, 0.0, 0.0)); assert b.state == "dodge" and fire is None, (b.state, fire)
    assert abs(cmd.get("lx", 0)) + abs(cmd.get("ly", 0)) > 0.1; out.append("dodge overrides fire + hunt (strafes, no shot)")
    # holds until the shot has passed, then resumes
    b.shots = []; b.t = 3.2; b.decide((0.0, 0.3, 0.0)); assert b.state == "dodge"
    b.t = 3.0 + 6.0; b.decide((0.0, 0.3, 0.0)); assert b.state != "dodge", b.state; out.append("dodge held until the shot passed, then resumes")
    # bounce path: a shot heading AWAY from me toward the top wall, which bounces back down onto me
    bb_ = Brain("A"); bb_.reset(True, 0.0); bb_.wait_until = 0; bb_.t = 1.0
    bb_.shots = [dict(xy=np.array([1.6, 1.6]), v=np.array([0.0, A.SHOT_SPEED]), t=1.0, rng=1.0, bbox=(0, 0, 1, 1))]
    assert bb_._threat(np.array([1.6, 0.9])) is not None; bb_.decide((1.6, 0.9, 0.0)); assert bb_.state == "dodge", bb_.state
    assert bb_._threat(np.array([2.6, 0.9])) is None
    bb_.shots = [dict(xy=np.array([0.7, 1.6]), v=np.array([0.0, A.SHOT_SPEED]), t=1.0, rng=1.0, bbox=(0, 0, 1, 1))]
    assert bb_._threat(np.array([0.7, 0.9])) is None      # the same bounce stopped by the L bracket in between: no dodge needed
    out.append("bounce-path threat (off the top wall) triggers a dodge; one blocked by an L bracket does not")
    bb_.shots = [dict(xy=np.array([1.6, 1.6]), v=np.array([0.0, A.SHOT_SPEED]), t=1.0, rng=1.0, bbox=(0, 0, 1, 1))]
    # walk-back: a seen shot interrupts it
    bb_.begin_return((-3.0, 0.0, 0.0), 1.0); c_ = bb_.return_cmd((0.6, 0.9, 0.0)); assert bb_.state == "dodge"; out.append("dodge overrides the walk-back")
    # belief: seen-empty area cleared, regrows behind a wall, search heads for it
    bl = OppBelief(); bl.reset(None); m0 = Memory()
    vis = m0.visible((-0.6, 0.0, 0.0), OppBelief.BEL_R, OppBelief.BEL_HFOV); bl.seen_empty(vis)
    i0, j0 = cell((0.4, 0.0)); assert bl.p[i0, j0] < 0.05 * bl.p.max(), "seen-empty cell not cleared"
    ib, jb = cell((1.7, 0.0)); assert bl.p[ib, jb] > 0.5 * bl.p.max(), "cell hidden behind the square block lost its belief"
    before = bl.p[i0, j0]
    for _ in range(30): bl.spread(0.1)
    assert bl.p[i0, j0] > 5 * before, "belief did not regrow"
    out.append("belief: seen floor cleared, hidden floor behind the block kept, regrows at walking speed")
    bs = Brain("A", np.random.default_rng(4)); bs.reset(True, 0.0); bs.wait_until = 0; bs.t = 1.0
    bs.bel.reset((2.0, 1.2)); bs.searching = bs.exploring = True
    c, _ = bs._decide((0.0, 0.0, 0.0)); g = bs.sgoal
    assert bs.state == "search" and np.linalg.norm(g - np.array([2.0, 1.2])) < 0.4, (bs.state, g)
    out.append(f"search heads for the belief peak {np.round(g, 1)}")
    bb = Brain("B", np.random.default_rng(4)); bb.reset(True, 0.0); bb.wait_until = 0; bb.t = 1.0; bb.bel.reset((-2.0, -1.2))
    bb._decide((0.0, 0.0, math.pi)); assert np.allclose(bb.sgoal, -g, atol=0.11), (bb.sgoal, g); out.append("belief search mirror-symmetric")
    # opponent unseen at start: belief search walks (no turning on the spot forever)
    b2 = Brain("B", np.random.default_rng(2)); b2.reset(True, 0.0); b2.wait_until = 0; b2.t = 0.5
    yaw = 0.0; spin = 0; walk = 0
    for i in range(80):
        b2.t = 0.5 + 0.1 * i; cmd, _ = b2.decide((0.0, 0.0, yaw))
        yaw = wrap(yaw + (-0.25 if cmd.get("rx", 0) > 0 else 0.25 if cmd.get("rx", 0) < 0 else 0.0))
        if abs(cmd.get("lx", 0)) + abs(cmd.get("ly", 0)) > 0.05: walk += 1
        elif abs(cmd.get("rx", 0)) > 0.1: spin += 1
    assert walk > 0 and spin < 40, (walk, spin)
    out.append(f"unseen opponent: belief search walks {walk} / turns in place {spin} of 80 ticks")
    # mirror: RED/BLUE decide the same under a point reflection
    cases = []
    for seed in (3, 5, 7):
        ba = Brain("A", np.random.default_rng(seed)); bb = Brain("B", np.random.default_rng(seed))
        for br, team, sign in ((ba, "A", +1), (bb, "B", -1)):
            br.reset(True, 1.0); br.wait_until = 0; br.t = 2.0
            br.opp = np.array([sign * 1.4, 0.2 * sign]); br.opp_t = 2.0; br.opp_v = np.array([0.0, 0.2 * sign])
            br.turret = 0.0 if team == "A" else math.pi; br.last_fire = -10
        ca, fa = ba._decide((0.0, 0.0, 0.0)); cb, fb = bb._decide((0.0, 0.0, math.pi))
        cases.append((ba.state, bb.state, bool(fa), bool(fb)))
    assert all(a == b and fa == fb for a, b, fa, fb in cases), cases; out.append(f"mirror-symmetric ({len(cases)} cases)")
    # clearance constant
    assert CRAB_R == 0.25 and INFL == 0.30; out.append("leg clearance INFL=0.30")
    # arena: every gap >= 2 x INFL, spawns in their pockets, and a route out round the bracket ends to the centre (both sides, mirrored)
    from jhs.tanks import field as TF
    bad = [(n, round(w, 2)) for n, w in TF.gaps() if w < 2 * INFL]
    assert not bad, bad
    for k, (sx, sy, _) in TF.START.items():
        assert TF.wall_clear(sx, sy) >= INFL, (k, TF.wall_clear(sx, sy))
    m = Memory(); pa = m.plan(np.array(TF.START["A"][:2]), np.array([0.0, 0.0]), False, sign=1)
    pb = m.plan(np.array(TF.START["B"][:2]), np.array([0.0, 0.0]), False, sign=-1)
    assert pa and pb and abs(len(pa) - len(pb)) <= 1, (len(pa), len(pb))
    assert min(TF.wall_clear(*p) for p in pa) >= INFL - 0.06
    assert max(abs(p[1]) for p in pa) > 0.8      # it walks round a bracket end (the spine blocks the straight line)
    out.append(f"arena {2 * TF.FX:.1f}x{2 * TF.FY:.1f} m: gaps " + ", ".join(f"{w:.2f}" for _, w in TF.gaps()) + f" m; spawn->centre A* {len(pa)}/{len(pb)} cells round the bracket")
    # no shot through a fixed wall
    b3 = Brain("A"); b3.reset(True, 0.0); assert not b3.shot_clear((-3.0, 0.0), (0.0, 0.0)) and b3.shot_clear((-0.6, 0.0), (0.6, 0.0))
    out.append("line of fire blocked by fixed walls")
    # ---- regression: seed 11 (ba76c7c) - first seen 2.1 s out, passing 0.12 m off my centre, 0.27 m from the square block: the old sidestep stepped ACROSS its path ----
    r1 = Brain("B", np.random.default_rng(1)); r1.reset(True, 0.0); r1.wait_until = 0; r1.t = 20.3
    me1 = np.array([0.6, -0.4]); src = np.array([-0.2, 0.15]); aimp = me1 + np.array([0.05, 0.11])
    v1 = (aimp - src) / np.linalg.norm(aimp - src) * A.SHOT_SPEED
    r1.shots = [dict(xy=me1 - v1 * 2.1, v=v1, t=20.3, rng=0.7, bbox=(0, 0, 1, 1))]
    r1.decide((me1[0], me1[1], 2.78)); g1 = r1.dodge_goal
    assert r1.state == "dodge" and g1 is not None and TF.wall_clear(*g1) >= INFL, (r1.state, g1)
    assert r1._step_clear(g1, 2.1) >= r1.SAFE, r1._step_clear(g1, 2.1)
    assert np.linalg.norm(g1 - me1) / 0.28 <= 2.2, g1
    out.append(f"regression seed 11: sidestep clears the shot path by {r1._step_clear(g1, 2.1):.2f} m (old code crossed it), 0.30 m from walls")
    r1.reset(True, 0.0); r1.wait_until = 0; r1.t = 20.3; me2 = np.array([0.93, -0.41])          # the exact tight spot (block + L bracket)
    r1.shots = [dict(xy=me2 + np.array([0.05, 0.11]) - v1 * 2.1, v=v1, t=20.3, rng=0.7, bbox=(0, 0, 1, 1))]
    r1.decide((me2[0], me2[1], 2.78)); g2 = r1.dodge_goal; c0 = r1._step_clear(me2, 2.1); c2 = r1._step_clear(g2, 2.1)
    assert c2 > c0 + 0.15 and TF.wall_clear(*g2) >= INFL, (c0, c2, g2)
    out.append(f"regression seed 11 exact spot (block + L bracket, 0.27 m from walls): the sidestep moves the path clearance {c0:.2f} -> {c2:.2f} m (old code: toward the path)")
    # ---- regression: seed 33 - a tracked enemy shot (after a bounce) next to my own shot was dropped as 'mine' ----
    r3 = Brain("A", np.random.default_rng(3)); r3.reset(True, 0.0); r3.wait_until = 0; r3.t = 15.0
    r3.own_in_flight = True; r3.own = dict(x=-0.55, y=-1.42, vx=A.SHOT_SPEED, vy=0.0, t=15.0)
    r3.shots = [dict(xy=np.array([-0.5, -1.47]), v=np.array([-A.SHOT_SPEED, 0.0]), t=15.0, rng=0.7, bbox=(0, 0, 1, 1))]
    r3.observe(15.1, dict(opp=[], shots=[dict(xy=np.array([-0.57, -1.47]), rng=0.6, bbox=(0, 0, 1, 1))], obst=np.zeros((0, 3))), (-1.15, -1.47, 0.0))
    assert r3.shots, "enemy shot dropped as own"
    r3.decide((-1.15, -1.47, 0.0)); assert r3.state == "dodge", r3.state
    r3b = Brain("A"); r3b.reset(True, 0.0); r3b.own_in_flight = True; r3b.own = dict(x=-0.55, y=-1.42, vx=A.SHOT_SPEED, vy=0.0, t=15.0)
    r3b.observe(15.1, dict(opp=[], shots=[dict(xy=np.array([-0.49, -1.42]), rng=0.6, bbox=(0, 0, 1, 1))], obst=np.zeros((0, 3))), (-1.15, -1.47, 0.0))
    assert not r3b.shots, "own outgoing shot not filtered"
    r4 = Brain("A"); r4.reset(True, 0.0); r4.wait_until = 0; r4.own_in_flight = True
    r4.own = dict(x=-0.70, y=-1.42, vx=A.SHOT_SPEED, vy=0.0, t=14.9)
    r4.observe(14.9, dict(opp=[], shots=[dict(xy=np.array([-0.66, -1.45]), rng=0.58, bbox=(0, 0, 1, 1))], obst=np.zeros((0, 3))), (-1.24, -1.40, 0.0))
    r4.observe(15.0, dict(opp=[], shots=[dict(xy=np.array([-0.72, -1.45]), rng=0.52, bbox=(0, 0, 1, 1))], obst=np.zeros((0, 3))), (-1.24, -1.40, 0.0))
    assert r4.shots and r4.shots[0]["v"][0] < 0, r4.shots
    r4.t = 15.0; r4.decide((-1.24, -1.40, 0.0)); assert r4.state == "dodge", r4.state
    out.append("regression seed 33 (exact): enemy shot re-seen 0.55 m out right next to my own outgoing shot, moving toward me -> dodge")
    out.append("regression seed 33: tracked enemy shot next to my own shot kept -> dodge; my own outgoing shot still ignored")
    # pocket: penned against the walls, a seen shot still gets a sidestep with 0.30 m clearance
    rp = Brain("A"); rp.reset(True, 0.0); rp.wait_until = 0; rp.t = 1.0
    vp = np.array([-0.55, -1.15]); vp = vp / np.linalg.norm(vp) * A.SHOT_SPEED          # in round the bracket end, at the pocket
    rp.shots = [dict(xy=np.array([-2.45, 1.15]), v=vp, t=1.0, rng=1.2, bbox=(0, 0, 1, 1))]
    rp.decide((-3.0, 0.0, 0.0)); gp = rp.dodge_goal
    assert rp.state == "dodge" and TF.wall_clear(*gp) >= INFL and rp._step_clear(gp, 2.1) >= rp.SAFE, (gp, rp._step_clear(gp, 2.1))
    out.append(f"pocket: shot in round the bracket end -> sidestep to ({gp[0]:.2f}, {gp[1]:.2f}), {TF.wall_clear(*gp):.2f} m from walls")
    # ---- openers ----
    rng = np.random.default_rng(11); cnt = {k: {o: 0 for o in OPENERS} for k in ("A", "B")}; worst = 9.0
    for _ in range(4000):
        r = roll_round(rng)
        for k in ("A", "B"):
            o = r[k]; cnt[k][o["opener"]] += 1
            sx, sy, sh = jittered_start(k, o["jit"]); worst = min(worst, TF.wall_clear(sx, sy))
            assert 2.85 <= abs(sx) <= 3.15 and abs(sy) <= 0.12 and np.sign(sx) == (-1 if k == "A" else 1), (k, sx, sy)   # pocket, behind the spine
            assert abs(TB_wrap(math.radians(sh) - math.radians(TF.START[k][2]))) <= JIT_YAW + 1e-9
    assert worst >= INFL, worst
    for k in ("A", "B"):
        for o in OPENERS: assert abs(cnt[k][o] / 4000 - 0.25) < 0.03, cnt
    out.append(f"jitter inside the pocket (min wall clearance {worst:.2f} m over 8000 starts); odds RED " + "/".join(str(cnt['A'][o]) for o in OPENERS) + " BLUE " + "/".join(str(cnt['B'][o]) for o in OPENERS))
    a1, a2 = roll_round(np.random.default_rng(5)), roll_round(np.random.default_rng(5))
    assert a1 == a2
    for op in OPENERS:
        for sd in (-1, 1): assert [(-x, y) for x, y, _ in opener_waypoints("A", op, sd)] == [(x, y) for x, y, _ in opener_waypoints("B", op, sd)]
    lk = roll_round(np.random.default_rng(5), dict(mode="locked", A="HOLD", B="PUSH_MID")); assert lk["A"]["opener"] == "HOLD" and lk["B"]["opener"] == "PUSH_MID"
    ra = [roll_round(np.random.default_rng(sd))["A"]["opener"] for sd in range(12)]; assert len(set(ra)) > 1
    out.append("toggle: Random reproducible per seed + varies across seeds; Locked gives the chosen openers")
    m = Memory(); legs = 0
    for k in ("A", "B"):
        for op in OPENERS:
            for side in (-1, 1):
                cur = np.array(TF.START[k][:2])
                for (gx, gy, _f) in opener_waypoints(k, op, side):
                    g = np.array([gx, gy]); assert TF.wall_clear(gx, gy) >= INFL, (k, op, side, gx, gy, TF.wall_clear(gx, gy))
                    pth = m.plan(cur, g, False, sign=1 if k == "A" else -1); assert pth, (k, op, side, gx, gy)
                    assert min(TF.wall_clear(*q) for q in pth) >= INFL - 0.06, (k, op, side); cur = g; legs += 1
    out.append(f"openers reachable with 0.30 m clearance ({legs} legs, both teams, both sides)")
    bo = Brain("A", np.random.default_rng(3)); bo.reset(True, 0.0); bo.wait_until = 0; bo.t = 1.0
    bo.set_opening(dict(opener="PEEK_NEAR", side=1, scan=1, bias=1, jit=(0, 0, 0)), 1.0)
    cmd, _ = bo.decide((-3.0, 0.0, 0.0)); assert bo.state == "opener" and cmd, bo.state
    bo.op_wps = bo.op_wps[1:]; cmd, _ = bo.decide((-1.85, 1.0, 0.0)); assert bo.state == "opener", bo.state
    bo.shots = [dict(xy=np.array([-1.85, 0.1]), v=np.array([0.0, A.SHOT_SPEED]), t=1.0, rng=1.0, bbox=(0, 0, 1, 1))]
    cmd, _ = bo.decide((-1.85, 1.0, 0.0)); assert bo.state == "dodge" and bo.opening is not None, bo.state
    bo.shots = []; bo.dodge_hold = -1e9; bo.t = 5.0
    bo.observe(5.0, dict(opp=[dict(xy=np.array([-1.0, 1.2]), rng=1.9, bbox=(0, 0, 1, 1))], shots=[], obst=np.zeros((0, 3))), (-2.0, 1.2, 0.0))
    assert bo.opening is None
    out.append("dodge overrides an opener; first sighting drops the crab into the normal brain")
    return out
