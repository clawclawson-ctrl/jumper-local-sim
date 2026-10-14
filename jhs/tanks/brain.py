"""Crab Tanks brain (SIM-ONLY VISION CONCEPT): hunt, take cover, lead the shot, dodge incoming shots.
Each crab sees only with its own camera + dToF + pose. Turret aim is independent of walking (within the camera FOV).
Leg clearance: half-span 0.25 m + 0.05 m margin = INFL 0.30 m (same as soccer). Mirror-symmetric."""
from __future__ import annotations
import math, os
import numpy as np
from scipy import ndimage
from jhs.soccer import vision
from jhs.soccer.explore import Memory, CRAB_R, INFL, CELL, XS, YS, cell
from jhs.soccer.brain import wrap, fam_mask, colour_blobs, FAM, TEAMFAM, OTHER, WALK
from jhs.soccer.field import FX, FY, wall_clear
from . import arena as A

FOV = math.radians(70)             # onboard camera half-width of view is ~35 deg each side; keep aim inside
SCAN_MAX, SCAN_WATCH = 10.0, 8.0
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
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None: continue
        blob = lab[sl] == i + 1; npx = int(blob.sum())
        if not (8 <= npx <= 900): continue
        ys, xs = np.nonzero(blob); ys = ys + sl[0].start; xs = xs + sl[1].start
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if max(bw, bh) > 2.5 * min(bw, bh) or npx < 0.4 * bw * bh: continue      # round, filled
        uc, vc = int(np.median(xs)), int(np.median(ys))
        d = cam_R @ np.array([(uc + 0.5 - cam.W / 2) / cam.f, -(vc + 0.5 - cam.H / 2) / cam.f, -1.0])
        Dn = d / max(1e-6, np.linalg.norm(d))
        # projectiles fly at SHOT_Z; intersect that plane
        if abs(Dn[2]) < 1e-4: continue
        t = (A.SHOT_Z - cam_pos[2]) / Dn[2]
        if t <= 0.05 or t > 6.0: continue
        g = cam_pos + t * Dn
        if abs(g[0]) > FX + 0.3 or abs(g[1]) > FY + 0.3: continue
        # reject the other crab's body (big colour blob nearby)
        if any(np.hypot(g[0] - o["xy"][0], g[1] - o["xy"][1]) < 0.25 for o in opp): continue
        shots.append(dict(xy=np.array([g[0], g[1]], float), rng=float(t),
                          bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), npx=npx))
    obst = vision.tof_obstacles(tof, tof_pos, tof_R, tof_cam, rmax=0.7) if tof is not None else np.zeros((0, 3))
    return dict(opp=opp, shots=shots, obst=obst)


class Brain:
    def __init__(self, team, rng=None, pushable=False, bounce=True):
        self.team = team; self.dirx = +1 if team == "A" else -1; self.pushable = bool(pushable); self.bounce = bool(bounce)
        self.mem = Memory(); self.rng = rng or np.random.default_rng(0)
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
        self.cover = None; self.stun_until = 0.0; self.ret_spot = None; self.own = None

    def observe(self, t, per, pose):
        self.t = t; self.obst = per["obst"]; x, y, yaw = pose
        self.mem.t = t
        self.mem.add_obstacles(per["obst"], None, (per["opp"][0]["xy"] if per["opp"] else self.opp if self.opp is not None and t - self.opp_t < 1.5 else None))
        self.mem.mark_view(pose)
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
        dets = [d for d in per["shots"] if self.own is None or math.hypot(d["xy"][0] - self.own["x"], d["xy"][1] - self.own["y"]) > 0.35]
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
            if pred is None and self.opp is not None and t - self.opp_t < 3.0:     # first sighting: assume it came from the opponent
                u = z - self.opp; nu = float(np.linalg.norm(u))
                if nu > 0.1: pred = u / nu * A.SHOT_SPEED
            seen.append(dict(xy=z.copy(), v=np.asarray(pred if pred is not None else np.zeros(2), float), t=t, rng=d["rng"], bbox=d["bbox"]))
        self.shots = seen; self.shot_t = t if seen else self.shot_t

    def begin_return(self, spot, t):
        self.ret_spot = (float(spot[0]), float(spot[1]), math.radians(spot[2])); self.ret_path = []; self.ret_plan_t = -1e9
        self.state = "return"; self.sub = "returning to start"; self.t = t; self.stuck_ref = None; self.hold = None

    def return_cmd(self, pose):
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
        if wc < INFL:
            gx, gy = -x, -y; g = np.array([c * gx + s * gy, -s * gx + c * gy]); g /= max(1e-6, np.linalg.norm(g))
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

    def fire_ready(self, pose, aim):
        """True when cooldown done, no own shot in flight, and the turret is close enough to aim."""
        if self.own_in_flight or self.t - self.last_fire < A.COOLDOWN: return False
        if abs(wrap(aim - self.turret)) > 0.12: return False
        return True

    def decide(self, pose):
        x, y, yaw = pose; me = np.array([x, y]); t = self.t
        if t < self.wait_until: self.state, self.sub = "kickoff", "ready"; return {}, None
        if t < self.stun_until: self.state, self.sub = "stunned", "STUNNED"; return {}, None
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
        # ---- dodge first ----
        dodge = None
        for s in self.shots:
            th = A.threat(me, s["xy"], s["v"], horizon=3.5)
            if th is None or th[1] > HIT_R + 0.12: continue
            tt, miss, off = th
            # step perpendicular to the shot, away from its path
            side = np.array([-s["v"][1], s["v"][0]]); side /= max(1e-6, np.linalg.norm(side))
            if off[0] * side[0] + off[1] * side[1] < 0: side = -side
            dodge = me + side * 0.55
            self.counts["dodges"] += 1; break
        if dodge is not None:
            self.state = "dodge"; self.sub = "DODGE: stepping out of the shot"
            return self._drive(pose, dodge, speed=0.55), None

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
            if self.fire_ready(pose, aim) and self.mem.clear_line(me, self.opp):
                fire = dict(yaw=self.turret, lead=lead_t); self.last_fire = t; self.own_in_flight = True; self.counts["shots"] += 1
                self.own = dict(x=x + A.MUZZLE * math.cos(self.turret), y=y + A.MUZZLE * math.sin(self.turret),
                                vx=A.SHOT_SPEED * math.cos(self.turret), vy=A.SHOT_SPEED * math.sin(self.turret), t=t)
                self.sub += " · FIRE"
            return cmd, fire

        # ---- search / explore (opponent not seen) ----
        if not self.searching and not self.exploring:
            self.searching = True; self.scan_acc = 0.0; self.scan_prev = yaw; self.scan_t0 = t
        if self.searching:
            if self.scan_prev is not None: self.scan_acc += abs(wrap(yaw - self.scan_prev))
            self.scan_prev = yaw
            moved = float(np.linalg.norm(me - self.wd[1])) if self.wd else 0.0
            if self.scan_acc > 2 * math.pi or t - self.scan_t0 > SCAN_MAX or (t - self.wd[0] > SCAN_WATCH and moved < 0.3):
                self.searching = False; self.exploring = True; self.exp_goal = None; self.wd = (t, me.copy())
            else:
                self.counts["searches"] += 1; self.state = "search"; self.sub = f"scan: looking for the opponent ({math.degrees(self.scan_acc):.0f} deg)"
                return {"rx": 0.62}, None
        return self._explore(pose), None


HIT_R = A.HIT_R


def unit():
    out = []
    b = Brain("A", np.random.default_rng(1))
    # lead + fire when the opponent is in view, clear LOS
    b.reset(True, 1.0); b.opp = np.array([1.5, 0.0]); b.opp_t = 1.0; b.opp_v = np.array([0.0, 0.3]); b.t = 2.0; b.wait_until = 0
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
    # dodge: a shot aimed at me
    b.shots = [dict(xy=np.array([-1.0, 0.0]), v=np.array([A.SHOT_SPEED, 0.0]), t=b.t, rng=1.0, bbox=(0, 0, 1, 1))]
    cmd, fire = b._decide((0.5, 0.0, 0.0)); assert b.state == "dodge", b.state; out.append("dodges an incoming shot")
    # bounded scan then explore (no spinning forever)
    b2 = Brain("B", np.random.default_rng(2)); b2.reset(True, 0.0); b2.wait_until = 0; b2.t = 0.5
    yaw = 0.0; states = []
    for i in range(80):
        b2.t = 0.5 + 0.1 * i; cmd, _ = b2.decide((0.0, 0.0, yaw))
        yaw = wrap(yaw + (-0.25 if cmd.get("rx", 0) > 0 else 0.25 if cmd.get("rx", 0) < 0 else 0.0))
        states.append(b2.state)
    assert "explore" in states or "search" in states; assert states.count("search") < 70, states.count("search")
    out.append(f"scan then explore (search ticks {states.count('search')})")
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
    return out
