"""Crab Tanks arena + projectile rules (SIM-ONLY VISION CONCEPT). The arena is the soccer field map (rounded bouncy
walls, the same placeable official obstacles); the goal mouths count as wall for shots. Projectiles are a sim add-on,
slow like Atari Combat missiles so a crab can see one coming and step out of its path. Pure functions, unit-tested."""
from __future__ import annotations
import math
from jhs.tanks import field as F

FX, FY, CORNER_R, PROPS, TEAM = F.FX, F.FY, F.CORNER_R, F.PROPS, F.TEAM
START = F.START                    # behind each '[' / ']' bracket, facing the centre
BALL_PARK = F.BALL_PARK            # the map's football waits outside the arena
SHOT_SPEED = 0.6                   # m/s (the crab walks ~0.3 m/s, so a 2 m shot gives ~3 s to step aside)
SHOT_R = 0.03; SHOT_Z = 0.15       # drawn sphere radius; flight height (= the turret barrel height)
SHOT_LIFE = 6.0; MAX_BOUNCES = 2
COOLDOWN = 2.5                     # s between shots; also only one shot of your own in flight (as in Combat)
HIT_R = 0.18                       # body hit radius around a crab's base (legs span 0.25)
STUN_S = 1.5
TURRET_RATE = math.radians(120)    # turret slew, rad/s
MUZZLE = 0.22


def validate(setup, strict=False):
    aid = str((setup or {}).get("arena") or F.ARENA)
    spec = F.ARENAS.get(aid)
    if spec is None: return [("error", f"unknown arena {aid}")]
    return F.validate(setup, strict, pieces=spec["pieces"], start=spec["start"])


def random_layout(seed, n=None):
    return F.random_layout(seed, n)


def reflect(x, y, vx, vy, r=SHOT_R):
    """keep (x, y) inside the rounded arena; -> (x, y, vx, vy, bounced)."""
    b = False; lx, ly = FX - r, FY - r
    if abs(x) > lx: x = math.copysign(2 * lx - abs(x), x); vx = -vx; b = True
    if abs(y) > ly: y = math.copysign(2 * ly - abs(y), y); vy = -vy; b = True
    cx, cy = abs(x) - (FX - CORNER_R), abs(y) - (FY - CORNER_R)
    if cx > 0 and cy > 0 and math.hypot(cx, cy) > CORNER_R - r:
        h = math.hypot(cx, cy); nx, ny = math.copysign(cx / h, x), math.copysign(cy / h, y)
        d = vx * nx + vy * ny
        if d > 0: vx, vy = vx - 2 * d * nx, vy - 2 * d * ny
        ox, oy = math.copysign(FX - CORNER_R, x), math.copysign(FY - CORNER_R, y)
        x, y = ox + nx * (CORNER_R - r - 1e-3), oy + ny * (CORNER_R - r - 1e-3); b = True
    for cx, cy, hx, hy in F.PIECES:      # the fixed layout walls bounce shots too (axis-aligned boxes)
        px, py = hx + r - abs(x - cx), hy + r - abs(y - cy)
        if px > 0 and py > 0:
            if px < py: x = cx + math.copysign(hx + r + 1e-3, x - cx); vx = -vx if (x - cx) * vx < 0 else vx
            else: y = cy + math.copysign(hy + r + 1e-3, y - cy); vy = -vy if (y - cy) * vy < 0 else vy
            b = True
    return x, y, vx, vy, b


def step_shots(shots, dt, props, crabs, bounce=True):
    """advance shots in place. shots: list of dict(owner, x, y, vx, vy, age, bounces). props {id: (x, y, deg)};
    crabs {k: (x, y)}. Returns events [(kind, shot, who)] with kind in hit/cover/wall/expired; spent shots removed."""
    ev = []; keep = []
    for s in shots:
        n = max(1, int(math.ceil(SHOT_SPEED * dt / 0.02)))
        dead = None
        for _ in range(n):
            s["x"] += s["vx"] * dt / n; s["y"] += s["vy"] * dt / n
            x, y, vx, vy, b = reflect(s["x"], s["y"], s["vx"], s["vy"])
            if b:
                s["bounces"] += 1
                if not bounce or s["bounces"] > MAX_BOUNCES: dead = ("wall", None); break
                s.update(x=x, y=y, vx=vx, vy=vy)
            hit = next((k for k, c in crabs.items() if k != s["owner"] and math.hypot(c[0] - s["x"], c[1] - s["y"]) < HIT_R), None)
            if hit: dead = ("hit", hit); break
            pk = next((k for k, v in props.items() if k in PROPS and F._box_dist((s["x"], s["y"]), k, *v) < SHOT_R), None)
            if pk: dead = ("cover", pk); break
        s["age"] += dt
        if dead is None and s["age"] > SHOT_LIFE: dead = ("expired", None)
        if dead: ev.append((dead[0], s, dead[1]))
        else: keep.append(s)
    shots[:] = keep
    return ev


def lead_angle(me, opp, opp_v, speed=SHOT_SPEED):
    """heading to fire from `me` so a shot meets a target at `opp` moving with `opp_v` (falls back to direct aim)."""
    rx, ry = opp[0] - me[0], opp[1] - me[1]; vx, vy = opp_v
    a = vx * vx + vy * vy - speed * speed; b = 2 * (rx * vx + ry * vy); c = rx * rx + ry * ry
    t = None
    if abs(a) < 1e-9: t = -c / b if b < 0 else None
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            ts = [x for x in ((-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)) if x > 0]
            t = min(ts) if ts else None
    if t is None or t > 6.0: return math.atan2(ry, rx), None
    return math.atan2(ry + vy * t, rx + vx * t), t


def threat(me, p, v, horizon=4.0):
    """(time to closest approach, miss distance, offset vector me - closest point) of a shot at p with velocity v."""
    rx, ry = me[0] - p[0], me[1] - p[1]; vv = v[0] ** 2 + v[1] ** 2
    if vv < 1e-6: return None
    t = (rx * v[0] + ry * v[1]) / vv
    if t <= 0 or t > horizon: return None
    ox, oy = rx - v[0] * t, ry - v[1] * t
    return t, math.hypot(ox, oy), (ox, oy)


def unit():
    out = []
    x, y, vx, vy, b = reflect(FX + 0.05, 0.0, 0.6, 0.0); assert b and vx < 0 and x < FX
    x, y, vx, vy, b = reflect(FX - 0.05, FY - 0.05, 0.4, 0.4); assert b and vx < 0 and vy < 0
    x, y, vx, vy, b = reflect(-1.30 + 0.18, 0.0, -0.6, 0.0); assert b and vx > 0 and x > -1.30 + 0.19
    out.append("walls + rounded corners + fixed layout pieces reflect")
    sh = [dict(owner="A", x=0.0, y=0.0, vx=SHOT_SPEED, vy=0.0, age=0.0, bounces=0)]
    ev = []
    for _ in range(100): ev += step_shots(sh, 0.05, {}, {"A": (-1, 0), "B": (0.9, 0)})
    assert ev and ev[0][0] == "hit" and ev[0][2] == "B", ev; out.append("shot hits the other crab, never its owner")
    sh = [dict(owner="A", x=0.0, y=0.0, vx=SHOT_SPEED, vy=0.0, age=0.0, bounces=0)]
    ev = []
    for _ in range(100): ev += step_shots(sh, 0.05, {"crateNE": (0.6, 0.0, 0.0)}, {"A": (-1, 0), "B": (0.9, 0)})
    assert ev[0][0] == "cover"; out.append("cover blocks shots")
    sh = [dict(owner="A", x=FX - 0.4, y=1.0, vx=SHOT_SPEED, vy=0.0, age=0.0, bounces=0)]
    ev = step_shots(sh, 1.0, {}, {}, bounce=False); assert ev and ev[0][0] == "wall"
    sh = [dict(owner="A", x=FX - 0.4, y=1.0, vx=SHOT_SPEED, vy=0.0, age=0.0, bounces=0)]
    ev = step_shots(sh, 1.0, {}, {}, bounce=True); assert not ev and sh[0]["vx"] < 0; out.append("bounce setting on/off")
    a, t = lead_angle((0, 0), (2, 0), (0, 0.3)); assert t and a > 0.1; out.append(f"lead {math.degrees(a):.0f} deg for a crossing target")
    th = threat((2, 0.1), (0, 0), (SHOT_SPEED, 0)); assert th and th[1] < 0.15
    assert threat((2, 0), (0, 0), (-SHOT_SPEED, 0)) is None; out.append("threat prediction")
    return out
