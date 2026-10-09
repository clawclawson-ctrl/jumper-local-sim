"""One crab's soccer brain. SIM-ONLY VISION CONCEPT.

Inputs each vision tick (10 Hz): this crab's OWN onboard camera image, its own-body self-mask (from its body model),
its own dToF grid, its own camera/dToF poses and base pose (own kinematics). Map knowledge: the field outline and
where the two goals are (like the room outline in hide & seek). NEVER: the ball's or the other crab's position from
the simulator -- those are only estimated from the camera here. Output: gamepad sticks for the official walking app.

Perception (hide & seek pieces): vision.detect for the football, plus colour blobs for the other crab (team colour,
low) and goals (team colour, tall). A ball candidate next to a detected crab is dropped (so the other crab's parts are
never taken for the ball). dToF points that stick out of the floor are obstacles (vision.tof_obstacles).
Behaviour: search -> stage behind the ball (relative to the goal it attacks, off walls) -> dribble/push -> shoot when
lined up; defend (between ball and own goal) when the other crab is clearly closer to the ball; back off when stuck."""
from __future__ import annotations
import math
import numpy as np
from scipy import ndimage
from . import vision
from .explore import Memory
from .field import FX, FY, GW, BALL_R, ATTACK, wall_clear, goal_centre, own_goal

# the football's top is 13 cm: lets vision range it even when the ball is cut by the image bottom (close, while dribbling)
vision.CLASSES["football"] = dict(vision.CLASSES["football"], top=2 * BALL_R)


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

# colour families (render colours: RED crab vermilion/gold, BLUE crab navy/steel; goals the defending team's colour)
FAM = {"red": dict(h=((0, 18), (342, 360)), s=0.50, v=(0.25, 1.0)), "blue": dict(h=((205, 245),), s=0.45, v=(0.12, 1.0))}
TEAMFAM = {"A": "red", "B": "blue"}
OTHER = {"A": "B", "B": "A"}
WALK = 0.40          # stick for normal walking (lin_vel_x range 0.8 m/s -> ~0.32 m/s)
DRIBBLE = 0.30
SHOOT = 0.70


def fam_mask(h, s, v, fam):
    c = FAM[fam]; m = np.zeros(h.shape, bool)
    for a, b in c["h"]: m |= (h >= a) & (h <= b)
    return m & (s >= c["s"]) & (v >= c["v"][0]) & (v <= c["v"][1])


def colour_blobs(img, cam_pos, cam_R, cam, selfmask, fam):
    """team-colour blobs -> [dict(kind 'crab'|'goal', xy, rng, ht, w, bbox)] by their ground geometry."""
    H, W = img.shape[:2]
    h, s, v = vision.hsv(img)
    m = fam_mask(h, s, v, fam) & ~selfmask
    m = ndimage.binary_closing(ndimage.binary_opening(m, iterations=1), iterations=3)
    lab, n = ndimage.label(m)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None: continue
        blob = lab[sl] == i + 1; npx = int(blob.sum())
        if npx < 25: continue
        ys, xs = np.nonzero(blob); ys = ys + sl[0].start; xs = xs + sl[1].start
        vb = int(ys.max()); ub = int(np.median(xs[ys >= vb - 1]))
        if vb >= H - 2: continue                      # cut by the image bottom: no ground point
        d = cam_R @ np.array([(ub + 0.5 - W / 2) / cam.f, -(vb + 1.0 - H / 2) / cam.f, -1.0])
        if d[2] >= -1e-4: continue
        t = -cam_pos[2] / d[2]; g = cam_pos + t * d
        rng = float(np.hypot(g[0] - cam_pos[0], g[1] - cam_pos[1]))
        if rng > 6.0: continue
        vt = int(ys.min()); ut = int(np.median(xs[ys <= vt + 1]))
        dt = cam_R @ np.array([(ut + 0.5 - W / 2) / cam.f, -(vt + 0.5 - H / 2) / cam.f, -1.0])
        ht = cam_pos[2] + rng * dt[2] / max(1e-6, math.hypot(dt[0], dt[1]))
        D = cam_R @ np.array([[(x + 0.5 - W / 2) / cam.f, 0.0, -1.0] for x in (xs.min(), xs.max())]).T
        az = np.arctan2(D[1], D[0]); w = (abs(wrap(az[1] - az[0])) + 1 / cam.f) * rng
        dirh = np.array([g[0] - cam_pos[0], g[1] - cam_pos[1]]) / max(rng, 1e-6)
        near_goal = abs(abs(g[0]) - FX) < 0.6 and abs(g[1]) < GW + 0.35
        kind = "goal" if (ht > 0.235 or (near_goal and ht > 0.2)) else "crab" if 0.04 < ht <= 0.235 and 0.06 < w < 0.6 else None
        if kind is None: continue
        ctr = g[:2] + dirh * (0.12 if kind == "crab" else 0.0)
        out.append(dict(kind=kind, fam=fam, xy=ctr, rng=rng, ht=round(float(ht), 3), w=round(float(w), 3),
                        bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), npx=npx))
    return out


def perceive(team, img, selfmask, cam_pos, cam_R, cam, tof, tof_pos, tof_R, tof_cam):
    """-> dict(ball=[dets], opp=[dets], goals=[dets], obst=Nx3). Everything from this crab's own sensors."""
    dets = [d for d in vision.detect(img, cam_pos, cam_R, cam, tof=tof, tof_pos=tof_pos, tof_R=tof_R, tof_cam=tof_cam, selfmask=selfmask)
            if d["cls"] == "football"]
    opp_fam = TEAMFAM[OTHER[team]]; own_fam = TEAMFAM[team]
    cb = colour_blobs(img, cam_pos, cam_R, cam, selfmask, opp_fam) + colour_blobs(img, cam_pos, cam_R, cam, selfmask, own_fam)
    opp = [d for d in cb if d["kind"] == "crab" and d["fam"] == opp_fam]
    goals = [dict(d, team=("B" if d["fam"] == "blue" else "A")) for d in cb if d["kind"] == "goal"]
    crabs = [d for d in cb if d["kind"] == "crab"]
    ball = []
    for d in dets:
        x, y = d["xy"]
        if any(np.linalg.norm(np.asarray(d["xy"]) - c["xy"]) < 0.22 for c in crabs): continue   # the other crab's parts
        bx0, by0, bx1, by1 = d["bbox"]
        if any(not (bx1 < c["bbox"][0] - 2 or bx0 > c["bbox"][2] + 2 or by1 < c["bbox"][1] - 2 or by0 > c["bbox"][3] + 2) for c in crabs): continue
        if abs(y) > FY + 0.05 or abs(x) > FX + 0.45: continue                                # outside the field
        ball.append(dict(d, xy=np.asarray(d["xy"], float)))
    obst = vision.tof_obstacles(tof, tof_pos, tof_R, tof_cam, rmax=0.7) if tof is not None else np.zeros((0, 3))
    return dict(ball=ball, opp=opp, goals=goals, obst=obst)


class Brain:
    def __init__(self, team, rng=None, pushable=False):
        self.team = team; self.dirx = ATTACK[team]; self.pushable = bool(pushable); self.mem = Memory()
        self.exp_goal = None; self.exp_path = []; self.exp_t = -1e9
        self.rng = rng or np.random.default_rng(0)
        self.reset(kickoff=False, t=0.0)
        self.counts = {"shots": 0, "touches": 0, "backoffs": 0, "searches": 0, "defend": 0}

    def reset(self, kickoff, t):
        """referee reset: the ball is on the centre spot (the rules say so), both crabs at their kickoff spots."""
        self.ball = np.zeros(2); self.bv = np.zeros(2); self.ball_t = t if kickoff is not None else -1e9; self.ball_known = True
        self.opp = None; self.opp_t = -1e9
        self.state = "kickoff"; self.sub = ""; self.t = t; self.hold = None; self.hold_until = 0.0
        self.stuck_ref = None; self.scan_acc = 0.0; self.scan_prev = None; self.search_goal = None
        self.shoot_until = 0.0; self.side = 1.0; self.last_cmd = {}; self.wait_until = t + 0.6; self.obst = np.zeros((0, 3))
        self.target = None

    # ---- beliefs ----------------------------------------------------------------------------------------------------
    def observe(self, t, per, pose):
        self.t = t; self.obst = per["obst"]
        x, y, yaw = pose
        self.mem.t = t; self.mem.add_obstacles(per["obst"], self.ball if t - self.ball_t < 1.5 else None); self.mem.mark_view(pose)
        if per["ball"]:
            pred = self.ball + self.bv * min(1.0, t - self.ball_t) if t - self.ball_t < 2.0 else None
            d = min(per["ball"], key=lambda d: np.linalg.norm(d["xy"] - pred) if pred is not None else d["rng"])
            z = d["xy"]
            if t - self.ball_t > 1.0 or pred is None:
                self.ball = z.copy(); self.bv[:] = 0
            else:
                dtt = max(0.05, t - self.ball_t)
                v = (z - self.ball) / dtt
                if np.linalg.norm(v) < 2.5: self.bv = 0.6 * self.bv + 0.4 * v
                self.ball = 0.5 * (self.ball + self.bv * dtt) + 0.5 * z
            self.ball_t = t; self.ball_known = True
        else:
            self.bv *= 0.9
        if per["opp"]:
            d = min(per["opp"], key=lambda d: d["rng"]); self.opp = np.asarray(d["xy"], float); self.opp_t = t

    def ball_age(self): return self.t - self.ball_t

    # ---- helpers --------------------------------------------------------------------------------------------------
    def _drive(self, pose, goal_xy, face_yaw=None, speed=WALK, avoid_ball=False, avoid=True):
        x, y, yaw = pose; me = np.array([x, y])
        d = np.asarray(goal_xy) - me; n = float(np.linalg.norm(d))
        want = face_yaw if face_yaw is not None else math.atan2(d[1], d[0])
        e = wrap(want - yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        sp = min(speed, max(0.18, n * 1.4)) if n > 0.04 else 0.0
        vx, vy = (c * d[0] + s * d[1]) / max(n, 1e-6) * sp, (-s * d[0] + c * d[1]) / max(n, 1e-6) * sp
        if face_yaw is None and abs(e) > 1.3: vx, vy = 0.0, 0.0
        if avoid: vx, vy = self._avoid(pose, vx, vy, avoid_ball)
        p = {"ly": -float(np.clip(vx, -0.45, 0.45)), "lx": -float(np.clip(vy, -0.45, 0.45))}
        if abs(e) > 0.06: p["rx"] = (-1 if e > 0 else 1) * (0.5 + 0.3 * min(1.0, abs(e)))
        return p

    def _avoid(self, pose, vx, vy, avoid_ball=False):
        x, y, yaw = pose; me = np.array([x, y]); c, s = math.cos(yaw), math.sin(yaw)
        pts = [p[:2] for p in self.obst[::3] if self.ball_age() > 1.5 or np.linalg.norm(p[:2] - self.ball) > 0.16]
        rads = [0.38] * len(pts)
        if self.opp is not None and self.t - self.opp_t < 1.5: pts.append(self.opp); rads.append(0.45)
        if avoid_ball and self.ball_age() < 3.0: pts.append(self.ball); rads.append(0.24)
        rep = np.zeros(2)
        for p, r in zip(pts, rads):
            dv = np.asarray(p) - me; n = float(np.linalg.norm(dv))
            if n < 1e-3 or n > r: continue
            dl = np.array([c * dv[0] + s * dv[1], -s * dv[0] + c * dv[1]])
            if dl @ np.array([vx, vy]) <= 0: continue
            rep -= dl / n * (r - n) / r
        wc = wall_clear(x, y)
        if wc < 0.25:      # the walls are map knowledge: do not walk into them
            gx, gy = -x, -y
            g = np.array([c * gx + s * gy, -s * gx + c * gy]); g /= max(1e-6, np.linalg.norm(g))
            if g @ np.array([vx, vy]) < 0: rep += g * (0.25 - wc) / 0.25
        if not np.any(rep): return vx, vy
        return vx + 0.9 * rep[0], vy + 0.9 * rep[1]

    # ---- return to kickoff (after a goal + celebration) ---------------------------------------------------------------
    RET_CELL, RET_CENTRE_R = 0.1, 0.33       # A* grid; the centre circle (ball on the spot) is blocked while walking back

    def begin_return(self, spot, t, ball_xy=(0.0, 0.0)):
        """walk back to my kickoff spot (x, y, heading deg) by myself; the referee has put the ball on the centre spot."""
        self.ret_spot = (float(spot[0]), float(spot[1]), math.radians(spot[2])); self.ret_occ = set(); self.ret_path = []
        self.ret_plan_t = -1e9; self.state = "return"; self.sub = "returning to kickoff"; self.t = t
        self.ball = np.array(ball_xy, float); self.bv[:] = 0; self.ball_t = t; self.stuck_ref = None; self.hold = None

    def _ret_cell(self, xy): return (int(round(xy[0] / self.RET_CELL)), int(round(xy[1] / self.RET_CELL)))

    def _ret_blocked(self, c):
        x, y = c[0] * self.RET_CELL, c[1] * self.RET_CELL
        if wall_clear(x, y) < 0.2 or math.hypot(x - self.ball[0], y - self.ball[1]) < self.RET_CENTRE_R: return True
        if self.opp is not None and self.t - self.opp_t < 3.0 and math.hypot(x - self.opp[0], y - self.opp[1]) < 0.4: return True
        return c in self.ret_occ

    def _ret_blocked_xy(self, x, y):
        return math.hypot(x - self.ball[0], y - self.ball[1]) < self.RET_CENTRE_R or self._opp_block(x, y)

    def plan(self, start, goal):
        return self.mem.plan(start, goal, self.pushable, self._ret_blocked_xy, sign=self.dirx)

    def plan_old(self, start, goal):
        """A* on my own occupancy (walls = map knowledge, centre circle, obstacles I have seen with dToF, the other crab as I last saw it)."""
        import heapq
        s0, g0 = self._ret_cell(start), self._ret_cell(goal)
        openq = [(0.0, s0)]; came = {s0: None}; cost = {s0: 0.0}; n = 0
        while openq and n < 6000:
            _, c = heapq.heappop(openq); n += 1
            if c == g0: break
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not dx and not dy: continue
                    nb = (c[0] + dx, c[1] + dy)
                    if nb != g0 and self._ret_blocked(nb): continue
                    nc = cost[c] + math.hypot(dx, dy)
                    if nc < cost.get(nb, 1e9):
                        cost[nb] = nc; came[nb] = c; heapq.heappush(openq, (nc + math.hypot(nb[0] - g0[0], nb[1] - g0[1]), nb))
        if g0 not in came: return []
        path = []; c = g0
        while c is not None: path.append(np.array([c[0], c[1]], float) * self.RET_CELL); c = came[c]
        return path[::-1]

    def return_cmd(self, pose):
        x, y, yaw = pose; me = np.array([x, y]); sx, sy, sh = self.ret_spot; goal = np.array([sx, sy])
        for p in self.obst[::2]:                       # remember obstacles seen on the way (inflated by my half-size)
            if np.linalg.norm(p[:2] - self.ball) > self.RET_CENTRE_R + 0.1:
                c = self._ret_cell(p[:2])
                for dx in (-2, -1, 0, 1, 2):
                    for dy in (-2, -1, 0, 1, 2):
                        if dx * dx + dy * dy <= 5: self.ret_occ.add((c[0] + dx, c[1] + dy))
        dg = float(np.linalg.norm(goal - me))
        if dg < 0.12:                                  # on the spot: turn to face the opponent's goal and wait
            e = wrap(sh - yaw); self.sub = "at kickoff spot, waiting" if abs(e) < 0.2 else "turning to face the goal"
            return {"rx": (-1 if e > 0 else 1) * (0.45 + 0.3 * min(1.0, abs(e)))} if abs(e) > 0.12 else {}
        if self.pushable and self.mem.through_obstacle(self.ret_path):
            self.state = "push_obst"; self.sub = f"PUSHING OBSTACLE (returning, {dg:.1f} m)"
        else: self.state = "return"
        if self.t - self.ret_plan_t > 1.0 or not self.ret_path:
            self.ret_path = self.plan(me, goal) or [me, goal]; self.ret_plan_t = self.t
        while len(self.ret_path) > 1 and np.linalg.norm(self.ret_path[0] - me) < 0.25: self.ret_path.pop(0)
        wp = self.ret_path[0] if dg > 0.3 else goal
        if self.state == "return": self.sub = f"returning to kickoff ({dg:.1f} m)"
        return self._drive(pose, wp, face_yaw=sh if dg < 0.35 else None, speed=WALK if dg > 0.4 else 0.3, avoid=self.state == "return")

    # ---- explore (ball not found by scanning) + path following that may shove pushable props ---------------------------
    def _follow(self, pose, path, goal, label):
        x, y, yaw = pose; me = np.array([x, y])
        while len(path) > 1 and np.linalg.norm(path[0] - me) < 0.22: path.pop(0)
        wp = path[0] if path else goal
        if self.pushable and self.mem.through_obstacle(path):      # the cheap way is through a pushable prop: walk in and shove it
            self.state = "push_obst"; self.sub = "PUSHING OBSTACLE (" + label + ")"; self.counts["pushes"] = self.counts.get("pushes", 0) + 1
            return self._drive(pose, wp, speed=WALK, avoid=False)
        return self._drive(pose, wp)

    def _explore(self, pose):
        x, y, yaw = pose; me = np.array([x, y])
        if self.exp_goal is None or self.t - self.exp_t > 6.0 or np.linalg.norm(self.exp_goal - me) < 0.25:
            if self.exp_goal is not None and np.linalg.norm(self.exp_goal - me) < 0.25:     # arrived: scan here first
                self.exp_goal = None; self.scan_acc = 0.0; self.scan_prev = yaw; return {"rx": 0.62}
            g, _ = self.mem.next_view(me, sign=self.dirx)
            if g is None: self.mem.seen[:] = -1e9; g, _ = self.mem.next_view(me, sign=self.dirx)       # everything searched: start over
            self.exp_goal = g if g is not None else np.zeros(2); self.exp_t = self.t
            self.exp_path = self.mem.plan(me, self.exp_goal, self.pushable, self._opp_block, sign=self.dirx) or [self.exp_goal]
            self.counts["explores"] = self.counts.get("explores", 0) + 1
        self.state = "explore"; self.sub = f"EXPLORE: next best view ({self.exp_goal[0]:.1f}, {self.exp_goal[1]:.1f})"
        return self._follow(pose, self.exp_path, self.exp_goal, "exploring")

    def _opp_block(self, x, y):
        return self.opp is not None and self.t - self.opp_t < 3.0 and math.hypot(x - self.opp[0], y - self.opp[1]) < 0.4

    def _stuck(self, pose, cmd):
        me = np.array(pose[:2]); walking = abs(cmd.get("ly", 0)) + abs(cmd.get("lx", 0)) > 0.12
        if self.stuck_ref is None or not walking: self.stuck_ref = (self.t, me); return False
        if self.t - self.stuck_ref[0] < 2.5: return False
        moved = float(np.linalg.norm(me - self.stuck_ref[1])); self.stuck_ref = (self.t, me)
        return moved < 0.05

    def stage_dir(self, b):
        """push direction u for the ball at b: at the goal it attacks, turned off the walls if the crab can't get behind."""
        G = np.array([self.dirx * (FX + 0.1), float(np.clip(b[1] * 0.4, -GW * 0.5, GW * 0.5))])
        u0 = (G - b) / max(1e-6, np.linalg.norm(G - b))
        best, bs = u0, -1e9
        for da in (0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4):
            ca, sa = math.cos(da), math.sin(da); u = np.array([ca * u0[0] - sa * u0[1], sa * u0[0] + ca * u0[1]])
            S = b - u * (BALL_R + 0.30)
            if wall_clear(*S) < 0.2: continue
            sc = float(u @ u0) - 0.05 * abs(da)
            if sc > bs: bs, best = sc, u
        return best

    # ---- the decision, once per vision tick --------------------------------------------------------------------------
    def decide(self, pose):
        x, y, yaw = pose; me = np.array([x, y]); t = self.t
        if t < self.wait_until: self.state, self.sub = "kickoff", "ready"; return {}
        if self.hold is not None:
            if t < self.hold_until: return dict(self.hold)
            self.hold = None
        cmd = self._decide(pose)
        if self._stuck(pose, cmd):
            self.counts["backoffs"] += 1; self.state, self.sub = "backoff", "stuck: backing off"
            side = 1 if self.rng.random() < 0.5 else -1
            self.hold = {"ly": 0.35, "lx": 0.25 * side}; self.hold_until = t + 1.2; self.stuck_ref = None
            return dict(self.hold)
        self.last_cmd = cmd
        return cmd

    def _decide(self, pose):
        x, y, yaw = pose; me = np.array([x, y]); t = self.t
        # ---- search ----
        if self.ball_age() > 2.5:
            if self.state != "search": self.state = "search"; self.scan_acc = 0.0; self.scan_prev = yaw; self.counts["searches"] += 1; self.search_goal = None
            self.scan_acc += abs(wrap(yaw - self.scan_prev)); self.scan_prev = yaw
            if self.scan_acc < 2 * math.pi + 0.3 and self.search_goal is None:
                self.sub = "scan: turning to look for the ball"
                d = self.ball - me; e = wrap(math.atan2(d[1], d[0]) - yaw) if self.ball_age() < 8 else 1.0
                return {"rx": (-1 if e > 0 else 1) * 0.62}
            return self._explore(pose)
        b = self.ball.copy(); db = float(np.linalg.norm(b - me))
        # ---- defend: the other crab is clearly closer to the ball ----
        opp_ok = self.opp is not None and t - self.opp_t < 2.0
        if opp_ok:
            do = float(np.linalg.norm(b - self.opp))
            OG = np.array(own_goal(self.team))
            opp_goalside = float((OG - b) @ (self.opp - b)) < 0       # opponent behind the ball (pushing toward my goal)
            if do + 0.35 < db and opp_goalside and abs(b[0] - OG[0]) < 3.2:
                if self.state != "defend": self.counts["defend"] += 1
                self.state = "defend"; self.sub = "defend: between the ball and my goal"
                v = OG - b; L = float(np.linalg.norm(v)); D = b + v / L * min(0.7, 0.5 * L)
                D[1] = float(np.clip(D[1], -FY + 0.25, FY - 0.25)); D[0] = float(np.clip(D[0], -FX + 0.25, FX - 0.25))
                face = math.atan2(b[1] - D[1], b[0] - D[0])
                if np.linalg.norm(D - me) < 0.12: return self._drive(pose, me, face_yaw=math.atan2(b[1] - y, b[0] - x))
                return self._drive(pose, D, face_yaw=face if np.linalg.norm(D - me) < 0.5 else None, avoid_ball=True)
        # ---- attack ----
        u = self.stage_dir(b); perp = np.array([-u[1], u[0]])
        rel = me - b; along = float(rel @ u); side = float(rel @ perp)
        hu = math.atan2(u[1], u[0]); ang = wrap(hu - yaw)
        G = np.array(goal_centre(self.team))
        dgoal = float(np.linalg.norm(G - b))
        behind = along < -(BALL_R + 0.08) and abs(side) < 0.11 and abs(ang) < 0.45
        if self.state in ("dribble", "shoot") and along < -0.02 and abs(side) < 0.16 and abs(ang) < 0.7 and db < 0.6:
            behind = True
        if behind and self.ball_age() < 1.2:
            to_goal = (G - b) / max(1e-6, dgoal); aligned = float(to_goal @ u) > 0.97
            if (dgoal < 1.4 and aligned and abs(ang) < 0.25) or t < self.shoot_until:
                if t >= self.shoot_until: self.shoot_until = t + 1.0; self.counts["shots"] += 1
                self.state, self.sub = "shoot", "shoot!"
                sp = SHOOT
            else:
                self.state, self.sub = "dribble", "dribble toward the goal"
                sp = DRIBBLE
            c, s = math.cos(yaw), math.sin(yaw)
            lat = -s * (b[0] - x) + c * (b[1] - y)                    # ball left/right of my centre line
            p = {"ly": -sp}
            if abs(lat) > 0.015: p["lx"] = -float(np.clip(2.5 * lat, -0.25, 0.25))
            want = hu + float(np.clip(-1.2 * side, -0.3, 0.3))
            e = wrap(want - yaw)
            if abs(e) > 0.05: p["rx"] = (-1 if e > 0 else 1) * (0.5 + 0.3 * min(1.0, abs(e) / 0.5))
            return p
        # stage: get behind the ball (round it, never through it)
        self.state = "stage"
        S = b - u * (BALL_R + 0.26)
        if along > -(BALL_R + 0.12):
            sgn = 1.0 if side >= 0 else -1.0
            if wall_clear(*(b + perp * sgn * 0.4)) < 0.22: sgn = -sgn
            P = b + perp * sgn * 0.40 - u * (0.05 if along > 0.05 else 0.25)
            self.sub = "going round the ball"
            return self._drive(pose, P, face_yaw=math.atan2(b[1] - y, b[0] - x) if db < 0.9 else None, avoid_ball=True)
        from .explore import cell as _cell
        if self.pushable and self.mem.occ()[_cell(S)] and self.ball_age() < 2.0:
            self.state = "push_obst"; self.sub = "PUSHING OBSTACLE to free the ball"; self.counts["pushes"] = self.counts.get("pushes", 0) + 1
            return self._drive(pose, S, face_yaw=hu if float(np.linalg.norm(S - me)) < 0.4 else None, avoid=False, speed=WALK)
        self.sub = "getting behind the ball"
        n = float(np.linalg.norm(S - me))
        return self._drive(pose, S, face_yaw=hu if n < 0.25 else (math.atan2(b[1] - y, b[0] - x) if db < 1.2 else None), avoid_ball=True,
                           speed=WALK if n > 0.3 else 0.28)


def unit():
    """quick checks without the simulator."""
    from .field import validate, random_layout, KICKOFF, in_goal
    out = []
    br = Brain("A"); br.reset(kickoff=True, t=0.0); br.t = 1.0
    br.observe(1.0, dict(ball=[dict(xy=np.array([0.5, 0.0]), rng=1.0, bbox=(0, 0, 1, 1))], opp=[], goals=[], obst=np.zeros((0, 3))), (0.0, 0.0, 0.0))
    c = br.decide((0.0, 0.0, 0.0)); assert br.state in ("dribble", "stage"), br.state
    assert c.get("ly", 0) < 0, c; out.append(f"ball ahead toward goal -> {br.state} ok")
    br2 = Brain("A"); br2.reset(kickoff=True, t=0.0)
    br2.observe(1.0, dict(ball=[dict(xy=np.array([-0.5, 0.0]), rng=0.5, bbox=(0, 0, 1, 1))], opp=[], goals=[], obst=np.zeros((0, 3))), (0.0, 0.0, 0.0))
    br2.decide((0.0, 0.0, 0.0)); assert br2.state == "stage" and br2.sub == "going round the ball", (br2.state, br2.sub)
    out.append("ball behind me -> go round it (not through) ok")
    br3 = Brain("B"); br3.reset(kickoff=True, t=0.0)
    br3.observe(1.0, dict(ball=[dict(xy=np.array([0.3, 0.0]), rng=1.0, bbox=(0, 0, 1, 1))], opp=[dict(xy=np.array([0.0, 0.0]), rng=1.2)], goals=[], obst=np.zeros((0, 3))), (1.4, 0.4, math.pi))
    br3.decide((1.4, 0.4, math.pi)); assert br3.state == "defend", br3.state; out.append("opponent closer and behind the ball -> defend ok")
    br4 = Brain("A"); br4.reset(kickoff=True, t=0.0); br4.t = 5.0
    br4.observe(5.0, dict(ball=[], opp=[], goals=[], obst=np.zeros((0, 3))), (0.0, 0.0, 0.0))
    c = br4.decide((0.0, 0.0, 0.0)); assert br4.state == "search" and abs(c.get("rx", 0)) >= 0.5, (br4.state, c); out.append("ball lost -> scan ok")
    u = Brain("A").stage_dir(np.array([1.0, FY - 0.08])); S = np.array([1.0, FY - 0.08]) - u * (BALL_R + 0.30)
    assert wall_clear(*S) >= 0.2, S; out.append("ball on the wall -> staging spot inside the field ok")
    from .field import CORNER_R
    bc = np.array([FX - 0.12, FY - 0.12]); assert wall_clear(*bc) < 0.0, "corner area should be outside the rounded corner"
    bc = np.array([FX - CORNER_R * 0.55, FY - CORNER_R * 0.55]); u = Brain("A").stage_dir(bc); S = bc - u * (BALL_R + 0.30)
    assert wall_clear(*S) >= 0.2, S; out.append("ball in a rounded corner -> staging spot inside the field ok")
    assert in_goal(FX + 0.07, 0.0) == 1 and in_goal(FX + 0.05, 0.0) == 0 and in_goal(FX + 0.2, 0.5) == 0; out.append("goal line ok")
    assert not [e for e in validate({"ball": [0, 0], "A": list(KICKOFF["A"]), "B": list(KICKOFF["B"]), "props": {}}) if e[0] == "error"]
    _gm = {"ball": [0, 0], "A": list(KICKOFF["A"]), "B": list(KICKOFF["B"]), "props": {"crateNE": [FX - 0.2, 0.0, 0]}}
    assert not [e for e in validate(_gm) if e[0] == "error"], "obstacles may go anywhere inside the field (e.g. a goal mouth)"
    assert [e for e in validate(_gm, strict=True) if e[0] == "error"]      # the random layout keeps the goal mouths clear
    lay = random_layout(3); assert lay and not [e for e in validate({"ball": [0, 0], "A": list(KICKOFF["A"]), "B": list(KICKOFF["B"]), "props": lay}) if e[0] == "error"]
    out.append(f"setup validation + random layout ({len(lay)} obstacles) ok")
    br = Brain("A"); br.begin_return((-0.42, 0.0, 0.0), 0.0); pth = br.plan(np.array([1.0, 0.05]), np.array([-0.42, 0.0]))
    assert pth and all(math.hypot(*p) >= Brain.RET_CENTRE_R - 1e-6 for p in pth[1:-1]), "return path must go round the centre circle"
    assert all(wall_clear(*p) >= 0.19 for p in pth[1:-1]); cmd = br.return_cmd((1.0, 0.05, math.pi))
    assert abs(cmd.get("ly", 0)) + abs(cmd.get("lx", 0)) + abs(cmd.get("rx", 0)) > 0
    br.obst = np.zeros((0, 3)); assert br.return_cmd((-0.42, 0.0, 0.05)) == {}, "on the spot + facing the goal -> wait"
    from . import explore as X
    m = X.Memory(); m.t = 100.0
    m.add_obstacles(np.array([[1.0, y, 0.1] for y in np.arange(-0.6, 0.61, 0.05)]))          # a prop wall at x = 1.0
    for yaw in np.linspace(-math.pi, math.pi, 24): m.mark_view((0.3, 0.0, yaw))               # scanned all round from (0.3, 0)
    g, _ = m.next_view(np.array([0.3, 0.0]))
    assert m.unseen()[X.cell((1.4, 0.0))], "floor right behind the prop must still be unseen"
    assert g is not None and (g[0] > 1.0 or abs(g[1]) > 0.6), f"next view should look behind the prop, got {g}"
    out.append(f"explore: floor behind a prop stays unseen, next best view {np.round(g, 2)} ok")
    m2 = X.Memory(); m2.t = 5.0
    m2.add_obstacles(np.array([[0.0, y, 0.1] for y in np.arange(-FY + 0.06, FY - 0.05, 0.05)]))    # props across the whole field
    assert not m2.plan(np.array([-1.0, 0.0]), np.array([1.0, 0.0]), pushable=False), "fixed props across the field -> no path"
    pth2 = m2.plan(np.array([-1.0, 0.0]), np.array([1.0, 0.0]), pushable=True); assert pth2 and m2.through_obstacle(pth2, k=len(pth2))
    m3 = X.Memory(); m3.t = 5.0; m3.add_obstacles(np.array([[0.0, y, 0.1] for y in np.arange(-0.2, 0.21, 0.05)]))  # short prop
    pth3 = m3.plan(np.array([-1.0, 0.0]), np.array([1.0, 0.0]), pushable=True); assert pth3 and not m3.through_obstacle(pth3, k=len(pth3))
    out.append("push-through costing: blocked field -> shove through a pushable prop; short prop -> walk round ok")
    # mirror test: RED in a situation vs BLUE in the same situation turned 180 deg about the centre spot -> the same stick commands
    R = lambda p: np.array([-p[0], -p[1]])
    cases = [((-0.5, 0.1, 0.2), (0.0, 0.05), None), ((0.3, -0.4, 2.5), (0.6, 0.2), None), ((1.0, 0.5, -2.9), (-0.2, 0.3), (-0.4, 0.2)),
             ((-1.8, -1.2, 0.7), (-2.0, -1.3), None), ((0.2, 0.0, 3.1), (1.6, 0.05), (1.9, 0.3))]
    for pose, ball, opp in cases:
        cm = {}
        for team, P, B, O in (("A", pose, ball, opp), ("B", (-pose[0], -pose[1], wrap(pose[2] + math.pi)), R(ball), None if opp is None else R(opp))):
            b_ = Brain(team); b_.reset(kickoff=None, t=0.0)
            for tt in (1.0, 1.1, 1.2):
                b_.observe(tt, dict(ball=[dict(xy=np.asarray(B, float), rng=1.0, bbox=(0, 0, 1, 1))], opp=[] if O is None else [dict(xy=np.asarray(O, float), rng=1.0)],
                                    goals=[], obst=np.zeros((0, 3))), P)
                c_ = b_.decide(P)
            cm[team] = (b_.state, {k: round(float(v), 3) for k, v in c_.items()})
        assert cm["A"][0] == cm["B"][0] and all(abs(cm["A"][1].get(k, 0) - cm["B"][1].get(k, 0)) < 0.02 for k in set(cm["A"][1]) | set(cm["B"][1])), (pose, cm)
    ma, mb = X.Memory(), X.Memory(); ma.t = mb.t = 50.0
    ga, _ = ma.next_view(np.array([-0.9, 0.2]), sign=1); gb, _ = mb.next_view(np.array([0.9, -0.2]), sign=-1)
    assert np.allclose(ga, -gb), (ga, gb)
    pa = ma.plan(np.array([-1.0, 0.3]), np.array([1.0, -0.2]), True, sign=1); pb_ = mb.plan(np.array([1.0, -0.3]), np.array([-1.0, 0.2]), True, sign=-1)
    assert len(pa) == len(pb_) and all(np.allclose(a_, -b2, atol=0.11) for a_, b2 in zip(pa, pb_)), "A* paths should mirror"
    out.append(f"mirror: {len(cases)} RED/BLUE situations -> same state + commands; explore view + A* path mirror ok")
    out.append(f"return to kickoff: A* path {len(pth)} cells round the centre circle, waits on the spot ok")
    return out


if __name__ == "__main__":
    for l in unit(): print(l)
