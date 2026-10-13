"""A floor pattern for two crabs, timed to the phrases the music analysis found.

Moon and Candle stay on opposite sides of their shared centre, so the distance
between them does not change when they turn from face-to-face to side-by-side.
A box step moves both of them by the same amount. Nothing here joins their claws.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

from jhs.ballroom.scene import HALF

STEP = 0.10
CLIPS = {
    "bow": ("bow", "bow"),
    "hello": ("hello", "salute"),
    "box": ("maze", "maze"),
    "promenade": ("wings", "waist"),
    "orbit": ("waist", "maze"),
    "wings": ("wings", "wings"),
    "brazilian": ("brazilian", "brazilian"),
    "waist": ("waist", "paw"),
}
SHOT = {
    "bow": "side", "hello": "side", "waist": "side", "box": "follow",
    "promenade": "follow", "orbit": "orbit", "wings": "crane", "brazilian": "crane",
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
}


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


def _kinds(phrases, meter: int) -> list[str]:
    n = len(phrases)
    kinds = ["bow"] * n
    if n >= 2:
        kinds[-1] = "bow"
    for i in range(1, max(0, n - 1)):
        e = float(phrases[i].get("energy", 0.5))
        if i % 4 == 0:
            k = "orbit"
        elif e >= 0.75:
            k = "wings" if i % 2 == 0 else "brazilian"
        elif e <= 0.28:
            k = "hello" if i % 2 else "waist"
        elif meter == 3 and i % 2 == 1:
            k = "box"
        else:
            k = "promenade"
        if k == kinds[i - 1]:
            k = "orbit" if k != "orbit" else "promenade"
        kinds[i] = k
    return kinds


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
        if kind == "promenade":
            c1, h1 = _next_spot(centre, heading, promenades)
            promenades += 1
        elif kind == "wings":
            c1, h1 = centre.copy(), heading + math.pi
        else:
            c1, h1 = centre.copy(), heading
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
    local = t - fig.t0
    if fig.kind == "orbit":
        c, h, form, off = fig.c0, fig.h0 + 2 * math.pi * u, 0.0, np.zeros(2)
    elif fig.kind == "promenade":
        w = _smooth(u)
        c = (1 - w) * fig.c0 + w * fig.c1
        h = lerp_angle(fig.h0, fig.h1, w)
        form, off = math.sin(math.pi * u), np.zeros(2)
    elif fig.kind == "wings":
        c, h, form, off = fig.c0, fig.h0 + math.pi * _smooth(u), 0.0, np.zeros(2)
    elif fig.kind == "box":
        c, h, form = fig.c0, fig.h0, 0.0
        off = _box_shift(local, beat, h, dur)
    else:
        c, h, form, off = fig.c0, fig.h0, 0.0, np.zeros(2)
    moon, my, candle, cy = place(c, h, form)
    return moon + off, my, candle + off, cy


def figure_at(figs: list[Figure], t: float) -> Figure:
    for f in figs:
        if t <= f.t1 + 1e-6:
            return f
    return figs[-1]


def check() -> str:
    """The pair never crowds, never leaves the floor, and one figure hands the next its ending spot."""
    beat = 60.0 / 98.0
    bar = 3 * beat
    phrases = []
    t = 0.0
    for i in range(8):
        phrases.append(dict(t0=t, t1=t + 8 * bar, energy=round(i / 7, 3)))
        t += 8 * bar
    figs = plan(dict(bpm=98, meter=3, beat=beat, phrases=phrases, duration=t))
    closest = 9.0
    for fig in figs:
        for k in range(48):
            u = k / 47
            tt = fig.t0 + (fig.t1 - fig.t0) * u
            m, _, c, _ = at(fig, tt, beat)
            d = float(np.hypot(*(m - c)))
            closest = min(closest, d)
            if d < 0.60:
                raise AssertionError(f"{fig.kind} brings them to {d:.3f} m")
            if max(np.linalg.norm(m), np.linalg.norm(c)) > 2.15:
                raise AssertionError(f"{fig.kind} leaves the floor")
    for a, b in zip(figs, figs[1:]):
        am, _, ac, _ = at(a, a.t1, beat)
        bm, _, bc, _ = at(b, b.t0, beat)
        if float(np.hypot(*(am - bm))) > 0.03 or float(np.hypot(*(ac - bc))) > 0.03:
            raise AssertionError(f"{a.kind} does not hand off to {b.kind}")
    return f"{len(figs)} figures, closest approach {closest:.2f} m"
