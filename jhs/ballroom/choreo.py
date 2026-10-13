"""A floor pattern for two crabs, timed to the phrases the music analysis found.

Moon and Candle stay a crab-width apart and never take claws. A traveling
figure walks forward, the way this robot's step is honest, and turns in place
when the figure asks them to face a new way. A pirouette is that turn with the
front claws kept for the pose. Nothing here joins their claws.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

from jhs.ballroom.scene import HALF

STEP = 0.10
# A curve at this speed still leaves the stance foot where it was planted.
WALK = 0.15
# Straight steps in the do-si-do. A straight walk at this speed still holds the stance.
DOSADO_V = 0.18
# In-place turns stay under the rate the tripod can step through.
YAW_RATE = 1.0
YAW_PEAK = 0.95
ROOM = 2.00
# Claws stay up on these. Courtesy is only Moon; Candle is walking the arc.
# The box steps backward and sideways. Four walking legs can do that;
# six legs skate. The claws stay up, the same way they do for a balance.
_BOTH_GESTURE = {"bow", "hello", "waist", "wings", "brazilian", "pirouette", "swing", "balance", "box"}
CLIPS = {
    "bow": ("bow", "bow"),
    "hello": ("hello", "salute"),
    "box": ("maze", "maze"),
    "promenade": ("wings", "waist"),
    "orbit": ("waist", "maze"),
    "wings": ("wings", "wings"),
    "brazilian": ("brazilian", "brazilian"),
    "waist": ("waist", "paw"),
    "pirouette": ("wings", "waist"),
    "dosado": ("maze", "maze"),
    "allemande": ("waist", "maze"),
    "star": ("maze", "waist"),
    "sashay": ("waist", "maze"),
    "swing": ("wings", "hello"),
    "balance": ("paw", "hello"),
    "courtesy": ("salute", "maze"),
}
SHOT = {
    "bow": "side", "hello": "side", "waist": "side", "box": "follow",
    "promenade": "follow", "orbit": "orbit", "wings": "crane", "brazilian": "crane",
    "pirouette": "orbit", "dosado": "side", "allemande": "orbit", "star": "crane",
    "sashay": "follow", "swing": "orbit", "balance": "side", "courtesy": "orbit",
}
ASIDE = {
    "bow": "a bow, from a careful distance",
    "hello": "hello, across the floor",
    "box": "the box step, six beats and home",
    "promenade": "down the floor, side by side",
    "orbit": "a turn, and the room turns with them",
    "wings": "dream wings, under the moon",
    "brazilian": "a brighter step, still in time",
    "waist": "a sway, and one paw offered",
    "pirouette": "a pirouette, turned on their own feet",
    "dosado": "a do-si-do, right shoulders, and home",
    "allemande": "an allemande, halfway round",
    "star": "a right-hand star, no claws joined",
    "sashay": "a sashay, side by side",
    "swing": "a swing, the long way round",
    "balance": "a balance, back and together",
    "courtesy": "a courtesy turn, one walks the arc",
}


def gestures(kind: str) -> tuple[bool, bool]:
    """Whether Moon, then Candle, keep the front claws for a pose."""
    if kind == "courtesy":
        return True, False
    g = kind in _BOTH_GESTURE
    return g, g


# A held pose may lean and sway with the clip. A step or a spin already
# places the body, and the clip's own root on top of that drags the feet.
# Hello, waist, wings and the brazilian sway the recorded body faster than
# a planted foot can follow. The claws still play the gesture. A bow is the
# one lean the stance can hold.
_CLIP_ROOT = {"bow"}


def clip_root(kind: str) -> bool:
    """True when the recording's lean is the pose, rather than a step."""
    return kind in _CLIP_ROOT


@dataclass
class Figure:
    t0: float
    t1: float
    kind: str
    c0: np.ndarray
    h0: float
    c1: np.ndarray
    h1: float
    clip_a: str
    clip_b: str
    rate: float
    delay: float
    shot: str

    @property
    def aside(self) -> str:
        return ASIDE.get(self.kind, self.kind)


def _smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def lerp_angle(a: float, b: float, t: float) -> float:
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * t


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _dir(heading: float) -> np.ndarray:
    return np.array([math.cos(heading), math.sin(heading)])


def _lin(u: float) -> float:
    return min(1.0, max(0.0, u))


def _turn_time(angle: float) -> float:
    """A steady turn at YAW_RATE. The foot clock already eases the first step."""
    return abs(float(angle)) / YAW_RATE


def place(centre, heading: float, form: float):
    """form 0: face to face. form 1: side by side, both facing `heading`."""
    ang = heading + math.pi - form * (math.pi / 2)
    off = HALF * np.array([math.cos(ang), math.sin(ang)])
    c = np.asarray(centre, dtype=np.float64)
    moon, candle = c + off, c - off
    return moon, heading, candle, heading + math.pi * (1.0 - form)


def _box_shift(local_t: float, beat: float, heading: float, dur: float) -> np.ndarray:
    """The pair's shared translation for a waltz box. The last half-second eases back to the start."""
    # forward, then left, close, back, right, close. In units of one step, along the heading.
    shape = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (0, 0)]
    beats = local_t / beat
    b = beats % 6.0
    i = int(b)
    f = b - i
    fwd = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-math.sin(heading), math.cos(heading)])
    def at(k):
        sf, sl = shape[k]
        return (sf * fwd + sl * left) * STEP
    shift = (1 - f) * at(i) + f * at(i + 1)
    if dur - local_t < 0.55:
        shift *= max(0.0, (dur - local_t) / 0.55)
    return shift


def _next_spot(centre: np.ndarray, heading: float, parity: int):
    sweep = 0.9 if parity % 2 == 0 else -0.75
    step = 0.78 * np.array([math.cos(heading + sweep / 2), math.sin(heading + sweep / 2)])
    c1 = centre + step
    h1 = heading + sweep
    if np.linalg.norm(c1) > 1.30:
        back = math.atan2(-centre[1], -centre[0]) if np.linalg.norm(centre) > 0.2 else heading + math.pi
        h1 = back
        c1 = centre * 0.35
    return c1, h1


def _spin_angle(local: float, dur: float) -> float:
    """A whole turn when the phrase is long enough, otherwise out and back."""
    room = YAW_PEAK * dur / 1.5
    turns = min(2, int(room // (2 * math.pi)))
    u = 0.0 if dur <= 1e-6 else min(1.0, max(0.0, local / dur))
    if turns >= 1:
        return turns * 2 * math.pi * _smooth(u)
    amp = min(math.pi, YAW_PEAK * dur / math.pi)
    return amp * math.sin(math.pi * u)


def _pirouette(centre, heading, local, dur):
    ang = _spin_angle(local, dur)
    moon, my, candle, cy = place(centre, heading, 0.0)
    return moon, my + ang, candle, cy + ang


@dataclass
class _Arc:
    angle: float
    tin: float
    twalk: float
    tout: float
    hold: float
    direction: int


def _orbit_plan(kind: str, dur: float) -> _Arc:
    """How far around the shared centre this phrase can walk, facing the way it travels."""
    cap = math.pi if kind == "allemande" else 2 * math.pi
    direction = -1 if kind == "swing" else 1
    tin = _turn_time(math.pi / 2)
    tout = tin
    spare = dur - tin - tout
    if spare < 0.35:
        return _Arc(0.0, 0.0, 0.0, 0.0, dur, direction)
    reach = WALK * spare / HALF
    if reach >= cap:
        return _Arc(cap, tin, spare, tout, 0.0, direction)
    twalk = reach * HALF / WALK
    return _Arc(reach, tin, twalk, tout, max(0.0, dur - tin - twalk - tout), direction)


def _tangent(face: float, direction: int) -> float:
    # CCW travel faces 90° to the right of the partner. CW faces the other way.
    return face - direction * (math.pi / 2)


def _orbit_at(centre, heading, kind, local, dur):
    arc = _orbit_plan(kind, dur)
    if arc.angle <= 1e-4:
        return place(centre, heading, 0.0)
    h = float(heading)
    t0 = arc.tin
    t1 = t0 + arc.twalk
    t2 = t1 + arc.tout

    def inward(psi):
        return place(centre, h + psi, 0.0)

    if local <= t0:
        s = _lin(local / t0) if t0 > 1e-6 else 1.0
        moon, my, candle, cy = inward(0.0)
        my = lerp_angle(my, _tangent(my, arc.direction), s)
        cy = lerp_angle(cy, _tangent(cy, arc.direction), s)
        return moon, my, candle, cy
    if local <= t1:
        u = (local - t0) / arc.twalk if arc.twalk > 1e-6 else 1.0
        average = arc.angle * HALF / arc.twalk if arc.twalk > 1e-6 else WALK
        frac = _smooth(u) if 1.5 * average <= WALK + 1e-6 else u
        psi = arc.direction * arc.angle * frac
        moon, my, candle, cy = inward(psi)
        return moon, _tangent(my, arc.direction), candle, _tangent(cy, arc.direction)
    if local <= t2:
        s = _lin((local - t1) / arc.tout) if arc.tout > 1e-6 else 1.0
        psi = arc.direction * arc.angle
        moon, my, candle, cy = inward(psi)
        face_m, face_c = my, cy
        my = lerp_angle(_tangent(face_m, arc.direction), face_m, s)
        cy = lerp_angle(_tangent(face_c, arc.direction), face_c, s)
        return moon, my, candle, cy
    return inward(arc.direction * arc.angle)


def _courtesy_plan(centre, heading, dur: float) -> _Arc:
    """Candle walks an arc around Moon. The arc stops at the walls."""
    h = float(heading)
    moon = np.asarray(centre, dtype=np.float64) - HALF * _dir(h)
    radius = 2 * HALF
    tin = _turn_time(math.pi / 2)
    tout = tin
    spare = dur - tin - tout

    def inside(angle):
        for k in range(28):
            psi = angle * k / 27
            pos = moon + radius * _dir(h + psi)
            if float(np.hypot(pos[0], pos[1])) > ROOM:
                return False
        return True

    ang = 0.0
    if spare >= 0.35:
        ang = min(math.pi * 0.85, WALK * spare / radius)
        if not inside(ang):
            lo, hi = 0.0, ang
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                if inside(mid):
                    lo = mid
                else:
                    hi = mid
            ang = lo
    if ang < 0.15:
        return _Arc(0.0, 0.0, 0.0, 0.0, dur, 1)
    need = ang * radius / WALK
    if need <= spare:
        twalk = spare
    else:
        twalk = need
        ang = WALK * twalk / radius
    return _Arc(ang, tin, twalk, tout, max(0.0, dur - tin - twalk - tout), 1)


def _courtesy_end(centre, heading, angle):
    h = float(heading)
    moon = np.asarray(centre, dtype=np.float64) - HALF * _dir(h)
    c1 = moon + HALF * _dir(h + angle)
    return c1, h + angle


def _courtesy_spin(local, dur, angle) -> float:
    """Moon ends facing Candle. A full extra turn when the phrase can carry it."""
    net = angle
    if angle > 0.2 and 1.5 * (angle + 2 * math.pi) / max(dur, 1e-3) <= YAW_PEAK:
        net = angle + 2 * math.pi
    elif angle <= 0.2:
        return _spin_angle(local, dur)
    u = 0.0 if dur <= 1e-6 else min(1.0, max(0.0, local / dur))
    return net * _smooth(u)


def _courtesy_at(centre, heading, local, dur):
    arc = _courtesy_plan(centre, heading, dur)
    h = float(heading)
    moon = np.asarray(centre, dtype=np.float64) - HALF * _dir(h)
    radius = 2 * HALF
    my = h + _courtesy_spin(local, dur, arc.angle)
    if arc.angle <= 1e-4:
        candle, cy = place(centre, h, 0.0)[2:]
        return moon, my, candle, cy
    start = moon + radius * _dir(h)
    end_psi = arc.angle
    end = moon + radius * _dir(h + end_psi)
    t0, t1, t2 = arc.tin, arc.tin + arc.twalk, arc.tin + arc.twalk + arc.tout
    face0 = h + math.pi
    tan0 = h + math.pi / 2
    if local <= t0:
        s = _lin(local / t0) if t0 else 1.0
        return moon, my, start, lerp_angle(face0, tan0, s)
    if local <= t1:
        u = (local - t0) / arc.twalk if arc.twalk else 1.0
        average = arc.angle * radius / arc.twalk if arc.twalk else WALK
        frac = _smooth(u) if 1.5 * average <= WALK + 1e-6 else u
        psi = end_psi * frac
        pos = moon + radius * _dir(h + psi)
        return moon, my, pos, h + psi + math.pi / 2
    if local <= t2:
        s = _lin((local - t1) / arc.tout) if arc.tout else 1.0
        y0 = h + end_psi + math.pi / 2
        y1 = h + end_psi + math.pi
        return moon, my, end, lerp_angle(y0, y1, s)
    return moon, my, end, h + end_psi + math.pi


def _segments(actions, speed: float = WALK):
    """Turn and walk pieces in one dancer's frame: forward, left, yaw."""
    f = left = yaw = 0.0
    segs = []
    for kind, mag in actions:
        if kind == "turn":
            dur = _turn_time(mag)
            segs.append(("turn", dur, f, left, yaw, yaw + mag))
            yaw += mag
        else:
            dur = abs(mag) / speed if abs(mag) > 1e-6 else 0.0
            df = math.cos(yaw) * mag
            dl = math.sin(yaw) * mag
            segs.append(("walk", dur, f, left, yaw, df, dl))
            f += df
            left += dl
    return segs


def _fit_walks(actions, dur: float, speed: float):
    """Shorten the steps so the turns still finish inside the phrase. None if the turns alone do not fit."""
    turns = sum(_turn_time(m) for k, m in actions if k == "turn")
    if turns > dur - 0.05:
        return None
    walks = sum(abs(m) for k, m in actions if k == "walk")
    room = dur - turns
    scale = min(1.0, room * speed / walks) if walks > 1e-6 else 1.0
    return tuple((k, m * scale if k == "walk" else m) for k, m in actions)


def _sample_frame(segs, local: float):
    t = 0.0
    last = (0.0, 0.0, 0.0)
    for seg in segs:
        kind, dur = seg[0], seg[1]
        if dur <= 1e-8:
            continue
        if local >= t + dur:
            if kind == "turn":
                last = (seg[2], seg[3], seg[5])
            else:
                last = (seg[2] + seg[5], seg[3] + seg[6], seg[4])
            t += dur
            continue
        u = (local - t) / dur
        if kind == "turn":
            _, _, f, left, y0, y1 = seg
            return f, left, lerp_angle(y0, y1, _lin(u))
        _, _, f, left, yaw, df, dl = seg
        return f + df * u, left + dl * u, yaw
    return last


def _world(origin, facing, forward, left):
    return (np.asarray(origin, dtype=np.float64)
            + forward * _dir(facing)
            + left * np.array([-math.sin(facing), math.cos(facing)]))


# Out to the right shoulder, forward past, and the same path home. Turns sum to zero.
_DOSADO = (
    ("turn", -math.pi / 2),
    ("walk", 0.30),
    ("turn", math.pi / 2 - 0.18),
    ("walk", 0.18),
    ("turn", math.pi),
    ("walk", 0.18),
    ("turn", -math.pi / 2 + 0.18),
    ("walk", 0.30),
    ("turn", -math.pi / 2),
)
# A short phrase only steps out to the right and about-faces home.
_DOSADO_SHORT = (
    ("turn", -math.pi / 2),
    ("walk", 0.22),
    ("turn", math.pi),
    ("walk", 0.22),
    ("turn", -math.pi / 2),
)
_BALANCE = (("walk", -0.10), ("walk", 0.10), ("walk", -0.10), ("walk", 0.10))


def _two_frames(centre, heading, actions, local, speed):
    segs = _segments(actions, speed)
    f, left, yaw = _sample_frame(segs, local)
    h = float(heading)
    moon = _world(np.asarray(centre, dtype=np.float64) - HALF * _dir(h), h, f, left)
    candle = _world(np.asarray(centre, dtype=np.float64) + HALF * _dir(h), h + math.pi, f, left)
    return moon, h + yaw, candle, (h + math.pi) + yaw


def _farthest(actions, centre, heading, speed) -> float:
    segs = _segments(actions, speed)
    total = sum(s[1] for s in segs) or 1.0
    c = np.asarray(centre, dtype=np.float64)
    far = 0.0
    for i in range(25):
        f, left, _yaw = _sample_frame(segs, total * i / 24)
        for origin, face in (
            (c - HALF * _dir(heading), heading),
            (c + HALF * _dir(heading), heading + math.pi),
        ):
            pos = _world(origin, face, f, left)
            far = max(far, float(np.hypot(pos[0], pos[1])))
    return far


def _dosado_actions(centre, heading, dur):
    fitted = _fit_walks(_DOSADO, dur, DOSADO_V)
    if fitted is None:
        fitted = _fit_walks(_DOSADO_SHORT, dur, DOSADO_V)
    if fitted is None:
        a = min(math.pi / 2, YAW_RATE * dur / 2)
        return (("turn", -a), ("turn", a))
    for _ in range(8):
        if _farthest(fitted, centre, heading, DOSADO_V) <= ROOM:
            break
        fitted = tuple((k, m * 0.8 if k == "walk" else m) for k, m in fitted)
    return fitted


def _dosado_at(centre, heading, local, dur):
    actions = _dosado_actions(centre, heading, dur)
    return _two_frames(centre, heading, actions, local, DOSADO_V)


def _balance_cycles(dur: float) -> int:
    one = sum(abs(m) for _, m in _BALANCE) / 0.08
    cycles = 1
    while cycles < 3 and (cycles + 1) * one <= dur + 1e-6:
        cycles += 1
    return cycles


def _balance_at(centre, heading, local, dur):
    cycles = _balance_cycles(dur)
    actions = _BALANCE * cycles
    segs = _segments(actions, 0.08)
    used = sum(s[1] for s in segs)
    if used > dur > 1e-6:
        actions = tuple(("walk", m * dur / used) for _, m in actions)
        segs = _segments(actions, 0.08)
    f, left, yaw = _sample_frame(segs, local)
    h = float(heading)
    moon = _world(np.asarray(centre, dtype=np.float64) - HALF * _dir(h), h, f, left)
    candle = _world(np.asarray(centre, dtype=np.float64) + HALF * _dir(h), h + math.pi, f, left)
    return moon, h + yaw, candle, (h + math.pi) + yaw


def _sashay_at(centre, heading, local, dur):
    """Turn to face one way, travel, about-face, and come home. A short phrase only opens and closes."""
    h = float(heading)
    c = np.asarray(centre, dtype=np.float64)
    full_turns = 2 * _turn_time(math.pi / 2) + _turn_time(math.pi)
    moon0, my0, candle0, cy0 = place(c, h, 0.0)
    side = h + math.pi / 2

    def slide(dist, twalk, tin, about, tout):
        # dist along `side`, then back. twalk is the time for one of those legs.
        marks = [tin, twalk, about, twalk, tout]
        bounds = []
        acc = 0.0
        for m in marks:
            bounds.append(acc)
            acc += m
        if local >= acc:
            return moon0, my0, candle0, cy0
        if local <= bounds[1]:
            s = _lin(local / tin) if tin else 1.0
            my = lerp_angle(my0, side, s)
            cy = lerp_angle(cy0, side, s)
            return moon0, my, candle0, cy
        if local <= bounds[2]:
            u = (local - bounds[1]) / twalk if twalk else 1.0
            shift = dist * u * _dir(side)
            return moon0 + shift, side, candle0 + shift, side
        if local <= bounds[3]:
            s = _lin((local - bounds[2]) / about) if about else 1.0
            back = side + math.pi
            shift = dist * _dir(side)
            return moon0 + shift, lerp_angle(side, back, s), candle0 + shift, lerp_angle(side, back, s)
        if local <= bounds[4]:
            u = (local - bounds[3]) / twalk if twalk else 1.0
            shift = dist * (1 - u) * _dir(side)
            back = side + math.pi
            return moon0 + shift, back, candle0 + shift, back
        s = _lin((local - bounds[4]) / tout) if tout else 1.0
        back = side + math.pi
        return moon0, lerp_angle(back, my0, s), candle0, lerp_angle(back, cy0, s)

    if dur >= full_turns + 0.8:
        spare = dur - full_turns
        dist = min(0.34, WALK * spare / 2)
        twalk = spare / 2 if dist >= 0.34 - 1e-9 or spare <= 1e-6 else dist / WALK
        if 2 * (0.34 / WALK) < spare:
            dist = 0.34
            twalk = spare / 2
        return slide(dist, twalk, _turn_time(math.pi / 2), _turn_time(math.pi), _turn_time(math.pi / 2))
    # A short phrase only turns out to the sashay and turns back. Walking the
    # return without an about-face would be a backward step, and this robot's
    # six legs skate when they do that.
    tin = min(_turn_time(math.pi / 2), dur / 2)
    if local <= tin:
        s = _lin(local / tin) if tin else 1.0
        return moon0, lerp_angle(my0, side, s), candle0, lerp_angle(cy0, side, s)
    if local <= dur - tin:
        return moon0, side, candle0, side
    s = _lin((local - (dur - tin)) / tin) if tin else 1.0
    return moon0, lerp_angle(side, my0, s), candle0, lerp_angle(side, cy0, s)


def _kinds(phrases, meter: int) -> list[str]:
    n = len(phrases)
    kinds = ["bow"] * n
    if n >= 2:
        kinds[-1] = "bow"
    waltz = ["hello", "box", "pirouette", "dosado", "promenade", "sashay",
             "allemande", "swing", "star", "balance", "courtesy", "wings",
             "waist", "brazilian"]
    square = ["hello", "dosado", "star", "sashay", "pirouette", "allemande",
              "promenade", "swing", "balance", "courtesy", "box", "brazilian"]
    order = waltz if meter == 3 else square
    # A very bright phrase may spin; a very quiet one balances. The calling
    # order itself is what makes sure each figure is danced.
    for i in range(1, max(0, n - 1)):
        e = float(phrases[i].get("energy", 0.5))
        k = order[(i - 1) % len(order)]
        if e >= 0.82 and k in ("hello", "balance", "waist"):
            k = "pirouette"
        elif e <= 0.18 and k in ("swing", "star", "brazilian", "courtesy"):
            k = "balance"
        if k == kinds[i - 1]:
            k = "dosado" if k != "dosado" else "sashay"
        kinds[i] = k
    return kinds


def _ending(kind, centre, heading, dur):
    centre = np.asarray(centre, dtype=np.float64)
    if kind == "wings":
        return centre.copy(), float(heading) + math.pi
    if kind in ("allemande", "star", "swing"):
        arc = _orbit_plan(kind, dur)
        return centre.copy(), float(heading) + arc.direction * arc.angle
    if kind == "courtesy":
        arc = _courtesy_plan(centre, heading, dur)
        return _courtesy_end(centre, heading, arc.angle)
    return centre.copy(), float(heading)


def plan(analysis, clips=None) -> list[Figure]:
    phrases = analysis["phrases"]
    if not phrases:
        dur = float(analysis.get("duration") or 8)
        phrases = [dict(t0=0.0, t1=dur, energy=0.5)]
    meter = int(analysis.get("meter") or 3)
    beat = float(analysis.get("beat") or 60.0 / float(analysis.get("bpm") or 98))
    bar = meter * beat
    maze_dur = 9.98
    if clips is not None and "maze" in clips.clips:
        maze_dur = float(clips.clips["maze"]["dur"])
    kinds = _kinds(phrases, meter)
    centre = np.zeros(2)
    heading = math.pi
    out = []
    promenades = 0
    for i, (ph, kind) in enumerate(zip(phrases, kinds)):
        dur = float(ph["t1"]) - float(ph["t0"])
        if kind == "promenade":
            c1, h1 = _next_spot(centre, heading, promenades)
            promenades += 1
        else:
            c1, h1 = _ending(kind, centre, heading, dur)
        ca, cb = CLIPS[kind]
        rate = 1.0
        if kind == "box":
            rate = min(1.45, max(0.65, maze_dur / max(4 * bar, 1e-3)))
        delay = beat if ca == cb else 0.2
        shot = "entrance" if i == 0 else "close" if i == len(phrases) - 1 else SHOT[kind]
        out.append(Figure(
            t0=float(ph["t0"]), t1=float(ph["t1"]), kind=kind,
            c0=centre.copy(), h0=float(heading), c1=np.asarray(c1, dtype=np.float64), h1=float(h1),
            clip_a=ca, clip_b=cb, rate=rate, delay=delay, shot=shot,
        ))
        centre, heading = np.asarray(c1, dtype=np.float64), float(h1)
    return out


def at(fig: Figure, t: float, beat: float):
    """World poses of Moon and Candle at time t: (xy, yaw) each."""
    dur = max(1e-3, fig.t1 - fig.t0)
    u = min(1.0, max(0.0, (t - fig.t0) / dur))
    local = min(max(t - fig.t0, 0.0), dur)
    if fig.kind == "orbit":
        c, h, form, off = fig.c0, fig.h0 + 2 * math.pi * u, 0.0, np.zeros(2)
    elif fig.kind == "promenade":
        w = _smooth(u)
        c = (1 - w) * fig.c0 + w * fig.c1
        h = lerp_angle(fig.h0, fig.h1, w)
        form, off = math.sin(math.pi * u), np.zeros(2)
        moon, my, candle, cy = place(c, h, form)
        return moon + off, my, candle + off, cy
    elif fig.kind == "wings":
        c, h, form, off = fig.c0, fig.h0 + math.pi * _smooth(u), 0.0, np.zeros(2)
    elif fig.kind == "box":
        c, h, form = fig.c0, fig.h0, 0.0
        off = _box_shift(local, beat, h, dur)
        moon, my, candle, cy = place(c, h, form)
        return moon + off, my, candle + off, cy
    elif fig.kind == "pirouette":
        return _pirouette(fig.c0, fig.h0, local, dur)
    elif fig.kind in ("allemande", "star", "swing"):
        return _orbit_at(fig.c0, fig.h0, fig.kind, local, dur)
    elif fig.kind == "dosado":
        return _dosado_at(fig.c0, fig.h0, local, dur)
    elif fig.kind == "sashay":
        return _sashay_at(fig.c0, fig.h0, local, dur)
    elif fig.kind == "balance":
        return _balance_at(fig.c0, fig.h0, local, dur)
    elif fig.kind == "courtesy":
        return _courtesy_at(fig.c0, fig.h0, local, dur)
    else:
        c, h, form, off = fig.c0, fig.h0, 0.0, np.zeros(2)
    moon, my, candle, cy = place(c, h, form)
    return moon + off, my, candle + off, cy


def figure_at(figs: list[Figure], t: float) -> Figure:
    for f in figs:
        if t <= f.t1 + 1e-6:
            return f
    return figs[-1]


def _yaw_err(a: float, b: float) -> float:
    return abs(_wrap(a - b))


def check() -> str:
    """The pair never crowds, never leaves the floor, and one figure hands the next its ending spot."""
    beat = 60.0 / 98.0
    bar = 3 * beat
    phrases = []
    t = 0.0
    for i in range(16):
        phrases.append(dict(t0=t, t1=t + 8 * bar, energy=round(i / 15, 3)))
        t += 8 * bar
    figs = plan(dict(bpm=98, meter=3, beat=beat, phrases=phrases, duration=t))
    closest = 9.0
    for fig in figs:
        for k in range(48):
            u = k / 47
            tt = fig.t0 + (fig.t1 - fig.t0) * u
            m, my, c, cy = at(fig, tt, beat)
            d = float(np.hypot(*(m - c)))
            closest = min(closest, d)
            if d < 0.60:
                raise AssertionError(f"{fig.kind} brings them to {d:.3f} m")
            if max(float(np.linalg.norm(m)), float(np.linalg.norm(c))) > 2.15:
                raise AssertionError(f"{fig.kind} leaves the floor")
        m, my, c, cy = at(fig, fig.t1, beat)
        em, emy, ec, ecy = place(fig.c1, fig.h1, 0.0)
        if float(np.hypot(*(m - em))) > 0.03 or float(np.hypot(*(c - ec))) > 0.03:
            raise AssertionError(f"{fig.kind} does not finish where the next figure starts")
        if _yaw_err(my, emy) > 0.08 or _yaw_err(cy, ecy) > 0.08:
            raise AssertionError(f"{fig.kind} finishes facing the wrong way")
    for a, b in zip(figs, figs[1:]):
        am, amy, ac, acy = at(a, a.t1, beat)
        bm, bmy, bc, bcy = at(b, b.t0, beat)
        if float(np.hypot(*(am - bm))) > 0.03 or float(np.hypot(*(ac - bc))) > 0.03:
            raise AssertionError(f"{a.kind} does not hand off to {b.kind}")
        if _yaw_err(amy, bmy) > 0.08 or _yaw_err(acy, bcy) > 0.08:
            raise AssertionError(f"{a.kind} turns away before {b.kind}")
    # A short phrase still has to close, including every new figure.
    for kind in ("pirouette", "dosado", "allemande", "star", "sashay", "swing", "balance", "courtesy"):
        dur = 7.2
        c1, h1 = _ending(kind, np.zeros(2), math.pi, dur)
        fig = Figure(0, dur, kind, np.zeros(2), math.pi, np.asarray(c1, dtype=np.float64), float(h1),
                     *CLIPS[kind], 1.0, 0.0, "side")
        m, my, c, cy = at(fig, dur, beat)
        em, emy, ec, ecy = place(c1, h1, 0.0)
        if float(np.hypot(*(m - em))) > 0.03 or _yaw_err(my, emy) > 0.08:
            raise AssertionError(f"a short {kind} does not finish in place")
    return f"{len(figs)} figures, closest approach {closest:.2f} m"
