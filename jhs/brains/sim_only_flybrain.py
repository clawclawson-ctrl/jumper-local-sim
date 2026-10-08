"""SIM-ONLY VISION CONCEPT. Fly-inspired rules driving the official locomotion policy.

flybrain.app's network is the official Jumper walking policy. It sees joints and the
IMU only. This script runs beside play.py --app in the simulator, renders the onboard
camera and the dToF, and drives the same gamepad a person would.

  ESCAPE     looming (range shrinking, short time-to-contact) or a bump:
             freeze briefly, then retreat and turn away
  AVOID      close depth on one side, or very close ahead: turn away / back up
  FIXATE     a dark vertical bar or a saturated blob: turn to centre it, walk
             toward it while centred, stop at a standoff
  OPTOMOTOR  wide-field image motion, with the crab's own yaw taken out: turn
             with the residual flow (syn-directional, the stabilising response)
  SACCADE    a short discrete turn, on a random fly-like interval
  WALK       straight ahead between saccades

Priority, with hysteresis: escape > avoid > fixate > optomotor > saccade > walk.
Hand-written rules. Not a FlyWire or Drosophila connectome.

Floor returns (fix1, 2026-10-08): the dToF also sees the floor. Each return is projected into the world with the
crab's own pose (the dToF camera pose = base pose + mount) and the ray geometry; returns less than FLOOR_MARGIN above
the floor are dropped before looming, nearest-range/side-avoid and blob range. Ranges are the HORIZONTAL distance
to the return, so the crab's own pitch (e.g. sagging while frozen) does not change the range to a vertical surface.
Looming is measured per WORLD bearing (2 deg bins of the above-floor returns' nearest horizontal range), not per
pixel, so the crab's own pitch and turning do not make one pixel compare two different surfaces; it needs at least
LOOM_MIN_BINS bearings of an above-floor region closing in, and closing speeds above LOOM_MAX_RATE (nothing in the
room moves that fast; it is a surface swap at a depth edge) are ignored.

Env: FB_SEED, FB_TMAX (sim seconds), FB_OUT, FB_APP, FB_MAP (empty = no map),
FB_MODE=sim|walk, FB_SAVEVIS=0/1, FB_YAW (optional fixed start yaw, else from seed).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/workspace/jumper")
HERE = Path(__file__).resolve().parent
APP = Path(os.environ.get("FB_APP", HERE / "flybrain.app"))
MAP = os.environ.get("FB_MAP", "")
SEED = int(os.environ.get("FB_SEED", "1"))
TMAX = float(os.environ.get("FB_TMAX", "120"))
MODE = os.environ.get("FB_MODE", "sim")
SAVEVIS = os.environ.get("FB_SAVEVIS", "0") == "1"
OUTD = Path(os.environ.get("FB_OUT", HERE / "runs" / f"seed{SEED}"))
VW, VH = 640, 480
VIS_DT = 0.1
ROOM = (-2.55, -2.55, 2.55, 2.55)  # inside the v7 walls; scoring only
STANDOFF = 0.42
FLOOR_MARGIN = 0.03   # m: a dToF return lower than this above the floor is floor
AZ_BIN = math.radians(2.0)   # world-bearing bins for looming
NBIN = int(round(2 * math.pi / AZ_BIN))
LOOM_MIN_BINS = 3     # bearings (2 deg each) that must close in together
LOOM_MAX_RATE = 2.0   # m/s; faster "closing" is a surface swap, not an approach
# Right stick x in this walking policy (flybrain.json controller.devices.gamepad, toolkit operator.py _band):
#   ang_vel_z: travel [0.5, 1.0] -> silent below |Rx| 0.5, then LINEAR to the full 4.0 rad/s at |Rx| 1.0
#   twist:     travel [0.0, 0.5, 0.5, 0.75] -> body twist, full at 0.5, fading to zero at 0.75
# The rules think in a turn t (|t| <= TURN_FULL, the old stick values). stick_turn() puts every non-zero t past
# the yaw dead zone, linear in t (fix2, 2026-10-08). SACCADE and ESCAPE were authored as raw stick values that are
# already past 0.5 (0.62 and 0.55) and keep their strength unchanged.
TURN_EPS = 0.02
TURN_FULL = 0.65
LABELS = ("ESCAPE", "AVOID", "FIXATE", "OPTOMOTOR", "SACCADE", "WALK")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tof_geometry(tof, tof_pos, tof_R, fovy_deg):
    """Height above the floor (z=0) and horizontal distance of every dToF return, from the crab's own pose.

    tof: (H, W) ranges along each ray (row 0 = top, col 0 = left), 0 = no return.
    tof_pos, tof_R: dToF camera position and rotation in the world (x right, y up, -z forward).
    """
    h, w = tof.shape
    f = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
    u = (np.arange(w) + 0.5 - w / 2) / f
    v = -(np.arange(h) + 0.5 - h / 2) / f
    uu, vv = np.meshgrid(u, v)
    d = np.stack([uu, vv, -np.ones_like(uu)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    D = d @ np.asarray(tof_R, float).T
    r = np.asarray(tof, float)
    z = float(tof_pos[2]) + D[..., 2] * r
    horiz = np.hypot(D[..., 0], D[..., 1]) * r
    return z, horiz, np.arctan2(D[..., 1], D[..., 0])


def above_floor(tof, tof_pos, tof_R, fovy_deg, margin=FLOOR_MARGIN):
    """Horizontal range of the above-floor returns (floor returns and no-returns become 0), the floor mask, and
    each pixel's world bearing."""
    z, horiz, az = tof_geometry(tof, tof_pos, tof_R, fovy_deg)
    valid = np.asarray(tof) > 0.04
    floor = valid & (z < margin)
    return np.where(valid & ~floor, horiz, 0.0).astype(np.float32), floor, az


def az_bins(horiz, az):
    """Nearest above-floor horizontal range per world-bearing bin (inf = nothing seen there)."""
    out = np.full(NBIN, np.inf)
    m = horiz > 0.05
    if m.any():
        idx = (np.floor((az[m] + math.pi) / AZ_BIN).astype(int)) % NBIN
        np.minimum.at(out, idx, horiz[m])
    return out


def _loom_bins(cur, prev, dt, heading, half_front, tof_hfov):
    """Looming on world bearings: an above-floor region in front that comes closer, fast enough (short TTC)."""
    centres = (np.arange(NBIN) + 0.5) * AZ_BIN - math.pi
    rel = (centres - heading + math.pi) % (2 * math.pi) - math.pi
    ok = (np.abs(rel) < half_front) & np.isfinite(cur) & np.isfinite(prev) & (cur < 1.6) & (prev < 3.2)
    with np.errstate(invalid="ignore"):
        rate = np.where(ok, (prev - cur) / max(dt, 1e-3), 0.0)
    m = ok & (rate > 0.30) & (rate < LOOM_MAX_RATE)
    if int(m.sum()) < LOOM_MIN_BINS:
        return None
    ttc = cur[m] / np.maximum(rate[m], 1e-3)
    med = float(np.median(ttc))
    if med > 0.65:
        return None
    # side > 0 = threat on the right (bearing clockwise of the heading), same scale as an image-column fraction
    return {"ttc": round(med, 3), "side": round(float(-rel[m].mean() / max(tof_hfov, 1e-3)), 3), "n": int(m.sum())}


def stick_turn(t):
    """Rule turn t -> right-stick x: 0 for |t| <= TURN_EPS, else sign(t) * (0.5 + 0.5 * min(1, |t| / TURN_FULL)).
    Yaw rate is then linear in t: 4.0 rad/s * min(1, |t| / TURN_FULL)."""
    t = float(t)
    if abs(t) <= TURN_EPS:
        return 0.0
    return math.copysign(0.5 + 0.5 * min(1.0, abs(t) / TURN_FULL), t)


def _near(a):
    v = a[(a > 0.04) & (a < 2.8)]
    return float(v.min()) if v.size else 3.0


def _flow(gray, prev):
    """Whole-pixel block match on each hemifield. Positive = features moved right.

    Returns (coherent, left, right). Coherent is 0 unless both halves agree,
    which is the wide-field motion an optomotor response should answer.
    """
    def best(a, b):
        base = float(np.mean((a - b) ** 2))
        bs, be = 0, base
        for s in range(-5, 6):
            if s == 0:
                continue
            if s > 0:
                e = float(np.mean((a[:, s:] - b[:, :-s]) ** 2))
            else:
                e = float(np.mean((a[:, :s] - b[:, -s:]) ** 2))
            if e < be:
                bs, be = s, e
        if bs == 0 or be > base * 0.90:
            return 0
        return bs

    w = gray.shape[1] // 2
    left, right = best(gray[:, :w], prev[:, :w]), best(gray[:, w:], prev[:, w:])
    if left == 0 or right == 0 or (left > 0) != (right > 0):
        return 0.0, left, right
    return 0.5 * (left + right), left, right


def _loom(tof, prev, dt):
    """Expanding approach: ranges that are shrinking. Median time-to-contact."""
    both = (tof > 0.05) & (prev > 0.05) & (tof < 1.6) & (prev < 3.2)
    rate = (prev - tof) / max(dt, 1e-3)
    m = both & (rate > 0.30)
    # centre-weighted: looming in front matters more than the side walls sliding by
    h, w = tof.shape
    yy, xx = np.mgrid[0:h, 0:w]
    front = (xx > w * 0.28) & (xx < w * 0.72)
    m = m & front
    if int(m.sum()) < 18:
        return None
    ttc = tof[m] / np.maximum(rate[m], 1e-3)
    med = float(np.median(ttc))
    if med > 0.65:
        return None
    xs = np.nonzero(m)[1]
    return {"ttc": round(med, 3), "side": round(float(xs.mean() / w - 0.5), 3), "n": int(m.sum())}


def _salient(img, selfm):
    """Dark vertical structure or a saturated blob, in the upper part of the view."""
    g = img.astype(np.float32).mean(-1)
    x = img.astype(np.float32) * (1.0 / 255.0)
    mx = x.max(-1)
    sat = np.where(mx > 1e-3, (mx - x.min(-1)) / np.maximum(mx, 1e-3), 0.0)
    if selfm is not None:
        g = g.copy()
        g[selfm] = 150
        sat = sat.copy()
        sat[selfm] = 0
    H, W = g.shape
    cut = int(H * 0.78)
    # Darker than the view's own median, so a dim room is not one giant bar.
    med = float(np.median(g[:cut]))
    dark = (g[:cut] < min(70.0, med - 18.0)).mean(0)
    color = ((sat[:cut] > 0.42) & (mx[:cut] > 0.22)).mean(0)
    score = np.maximum(dark, color * 1.5)
    sm = np.convolve(score, np.ones(9) / 9.0, mode="same")
    sm[:10] = 0
    sm[W - 10 :] = 0
    i = int(sm.argmax())
    # a peak, not a flat field: it has to rise above the typical column
    if sm[i] < 0.16 or sm[i] < float(np.median(sm)) + 0.10:
        return None
    cols = np.nonzero(sm > max(0.12, sm[i] * 0.55))[0]
    if cols.size < 2:
        return None
    # centre of the bar, not the first column of a flat top
    i = int(round(float(cols.mean())))
    return {
        "cx": i,
        "bearing": (i - W / 2) / (W / 2),
        "score": round(float(sm[i]), 3),
        "bbox": [int(cols[0]), int(H * 0.08), int(cols[-1]), cut],
        "kind": "color" if color[i] >= dark[i] else "dark",
    }


def _blob_range(tof, bearing, cam_hfov, tof_hfov):
    """dToF range along a camera bearing, or None if it sits outside the ToF cone."""
    half_c, half_t = cam_hfov * 0.5, tof_hfov * 0.5
    ang = bearing * half_c
    if abs(ang) > half_t * 0.92:
        return None
    w = tof.shape[1]
    col = int(np.clip((0.5 + ang / tof_hfov) * w, 1, w - 2))
    patch = tof[:, col - 1 : col + 2]
    v = patch[(patch > 0.05) & (patch < 2.5)]
    return float(np.median(v)) if v.size >= 4 else None


class Fly:
    """Arbitration. Pad signs match the locomotion contract: ly<0 walks forward,
    rx>0 turns right (kernel stick right, yaw command negated)."""

    def __init__(self, rng):
        self.rng = rng
        self.beh = "WALK"
        self.sub = ""
        self.lock_t = 0.0
        self.lock = None  # (until, pad dict) for escape / saccade
        self.saccade_at = float(rng.uniform(2.2, 4.2))
        self.avoid_until = 0.0
        self.fix_until = 0.0
        self.approaching = False
        self.counts = {k: 0 for k in ("escapes", "saccades", "approaches", "fixations", "avoids", "bumps", "optomotor")}
        self.time = {k: 0.0 for k in LABELS}
        self.prev_g = None
        self.prev_tof = None
        self.prev_yaw = None
        self.prev_xy = None
        self.prev_v = None
        self.last = {"ly": 0.0, "lx": 0.0, "rx": 0.0, "lb": False}
        self.last_turn = 0.0  # the rule-level turn t behind last["rx"]
        self.overlay = {}
        self.floor = None  # last floor mask (display only)
        self.prev_bins = None

    def _pad(self, beh, t, dt, ly=0.0, lx=0.0, rx=0.0, lb=False, sub="", raw=False):
        """rx is a rule turn t, mapped past the yaw dead zone by stick_turn(); raw=True passes a stick value as is
        (SACCADE / ESCAPE, authored past 0.5 already)."""
        self.time[beh] = self.time.get(beh, 0.0) + dt
        self.beh, self.sub = beh, sub
        self.last_turn = float(rx)
        self.last = {"ly": ly, "lx": lx, "rx": float(rx) if raw else stick_turn(rx), "lb": lb}
        return self.last

    def _begin_escape(self, t, side, why):
        self.counts["escapes"] += 1
        # threat on the right (side>0) -> turn left, rx<0
        turn = -0.55 if side > 0 else 0.55
        self.lock = ("escape", t, [(0.40, {"lb": True}), (1.15, {"ly": 0.58, "rx": turn})], why)
        self.approaching = False

    def _begin_saccade(self, t):
        self.counts["saccades"] += 1
        side = float(self.rng.choice([-1.0, 1.0]))
        dur = float(self.rng.uniform(0.30, 0.48))
        self.lock = ("saccade", t, [(dur, {"rx": side * 0.62})], "explore")
        self.saccade_at = t + dur + float(self.rng.uniform(2.6, 5.0))

    def step(self, t, dt, img, selfm, tof, yaw, xy, f_small, cam_hfov, tof_hfov, tof_pose=None):
        """tof_pose = (tof_pos, tof_R, tof_fovy_deg) from the crab's own pose. With it, floor returns are dropped and
        ranges are horizontal (pitch-proof). Without it (synthetic tests only), the raw ranges are used."""
        bins = heading = None
        if tof_pose is not None:
            tof, self.floor, az = above_floor(tof, *tof_pose)
            bins = az_bins(tof, az)
            fwd = -np.asarray(tof_pose[1], float)[:, 2]          # dToF optical axis in the world
            heading = math.atan2(fwd[1], fwd[0])
        else:
            self.floor = None
        g = img.astype(np.float32).mean(-1)
        small = g[::4, ::4]
        flow = left = right = 0.0
        residual = 0.0
        if self.prev_g is not None:
            flow, left, right = _flow(small, self.prev_g)
            dyaw = wrap(yaw - self.prev_yaw) if self.prev_yaw is not None else 0.0
            expected = dyaw * f_small
            residual = flow - expected
        if bins is not None:
            loom = _loom_bins(bins, self.prev_bins, dt, heading, 0.22 * tof_hfov, tof_hfov) if self.prev_bins is not None else None
            self.prev_bins = bins
        else:
            loom = _loom(tof, self.prev_tof, dt) if self.prev_tof is not None else None
        blob = _salient(img, selfm)
        br = _blob_range(tof, blob["bearing"], cam_hfov, tof_hfov) if blob else None
        h, w = tof.shape
        nl, nc, nr = _near(tof[:, : w // 3]), _near(tof[:, w // 3 : 2 * w // 3]), _near(tof[:, 2 * w // 3 :])
        v = 0.0
        if self.prev_xy is not None:
            v = float(np.hypot(*(np.asarray(xy) - self.prev_xy)) / max(dt, 1e-3))
        bump = (
            self.prev_v is not None
            and self.last["ly"] < -0.2
            and not self.last["lb"]
            and (self.prev_v - v) > 0.40
            and v < 0.08
            and self.prev_v > 0.12
        )
        self.prev_g, self.prev_tof, self.prev_yaw, self.prev_xy, self.prev_v = small, tof.copy(), yaw, np.asarray(xy, float), v
        self.overlay = {
            "flow": round(float(flow), 2),
            "residual": round(float(residual), 2),
            "hemifield": [left, right],
            "near": [round(nl, 2), round(nc, 2), round(nr, 2)],
            "loom": loom,
            "blob": blob,
            "blob_range": None if br is None else round(br, 2),
            "bump": bool(bump),
            "floor_px": None if self.floor is None else int(self.floor.sum()),
        }

        # locked escape or saccade, unless a new loom/bump should take over a saccade
        if self.lock is not None:
            kind, t0, phases, why = self.lock
            if kind == "saccade" and (loom or bump or nc < 0.16):
                self.lock = None
            else:
                acc = 0.0
                for dur, pad in phases:
                    if t - t0 < acc + dur:
                        beh = "ESCAPE" if kind == "escape" else "SACCADE"
                        return self._pad(beh, t, dt, sub=why, raw=True, **{"ly": 0, "lx": 0, "rx": 0, "lb": False, **pad})
                    acc += dur
                self.lock = None

        if loom or bump or nc < 0.13:
            side = 0.0
            if loom:
                side = loom["side"]
            elif nr + 0.05 < nl:
                side = 0.4
            elif nl + 0.05 < nr:
                side = -0.4
            why = "bump" if bump and not loom and nc >= 0.13 else ("loom" if loom else "close")
            if bump and not loom:
                self.counts["bumps"] += 1
            self._begin_escape(t, side, why)
            return self._pad("ESCAPE", t, dt, lb=True, sub=why)

        avoid = None
        if nl < 0.30 and nl < nr - 0.05:
            avoid = ("left", -0.12, 0.50)  # turn right, away
        elif nr < 0.30 and nr < nl - 0.05:
            avoid = ("right", -0.12, -0.50)
        elif nc < 0.20:
            away = 0.40 if nr >= nl else -0.40
            avoid = ("front", 0.45, away)
        if avoid or t < self.avoid_until:
            if avoid:
                self.avoid_until = t + 0.35
                if self.beh != "AVOID":
                    self.counts["avoids"] += 1
                tag, ly, rx = avoid
            else:
                tag, ly, rx = "hold", self.last["ly"], self.last_turn
            self.approaching = False
            return self._pad("AVOID", t, dt, ly=ly, rx=rx, sub=tag)

        if blob and (blob["score"] >= 0.18 or t < self.fix_until):
            self.fix_until = t + 0.45
            if self.beh != "FIXATE":
                self.counts["fixations"] += 1
            err = blob["bearing"]
            rng = br if br is not None else 9.0
            if abs(err) > 0.10:
                self.approaching = False
                return self._pad("FIXATE", t, dt, ly=-0.10, rx=float(np.clip(err * 1.4, -0.65, 0.65)), sub="centre")
            if rng > STANDOFF + 0.06:
                if not self.approaching:
                    self.counts["approaches"] += 1
                    self.approaching = True
                return self._pad("FIXATE", t, dt, ly=-0.40, rx=float(np.clip(err * 0.8, -0.3, 0.3)), sub="approach")
            self.approaching = False
            return self._pad("FIXATE", t, dt, ly=0.0, rx=float(np.clip(err * 0.5, -0.25, 0.25)), sub="standoff")
        self.approaching = False

        if abs(residual) >= 1.8 and flow != 0.0:
            self.counts["optomotor"] += 1
            rx = float(np.clip(residual / 7.0, -0.40, 0.40))
            return self._pad("OPTOMOTOR", t, dt, ly=-0.22, rx=rx, sub="stabilise")

        if t >= self.saccade_at:
            self._begin_saccade(t)
            kind, t0, phases, why = self.lock
            pad = phases[0][1]
            return self._pad("SACCADE", t, dt, sub=why, raw=True, **{"ly": 0, "lx": 0, "rx": 0, "lb": False, **pad})

        return self._pad("WALK", t, dt, ly=-0.40, sub="straight")


def _truth_fixations(model, live, beh, sub, yaw, xy):
    """Sim truth, for the score only. Never read by Fly.step."""
    import mujoco
    hits = []
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        low = nm.lower()
        if not any(k in low for k in ("duck", "ball", "football", "toy")):
            continue
        if "visual" in low:
            continue
        p = live.xpos[b]
        if p[2] > 0.4:
            continue
        d = p[:2] - xy
        dist = float(np.hypot(*d))
        bearing = wrap(math.atan2(d[1], d[0]) - yaw)
        if dist < 1.3 and abs(bearing) < 0.5:
            hits.append(nm.split("/")[-1])
    return hits if beh == "FIXATE" else []


def _run():
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("play", REPO / "scripts" / "play.py")
    play = importlib.util.module_from_spec(spec)
    sys.modules["play"] = play
    spec.loader.exec_module(play)

    import controller
    import torch

    torch.set_num_threads(int(os.environ.get("FB_THREADS", "2")))
    rng = np.random.default_rng(SEED)
    yaw0 = float(os.environ["FB_YAW"]) if os.environ.get("FB_YAW") else float(rng.uniform(-math.pi, math.pi))
    if MODE == "walk":
        yaw0 = 0.0

    import tasks
    _orig = tasks.load_env_cfg

    def _pinned(*a, **k):
        cfg = _orig(*a, **k)
        ev = cfg.events.get("reset_base") if getattr(cfg, "events", None) else None
        if ev is not None and "pose_range" in ev.params:
            ev.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.01, 0.01), "yaw": (yaw0, yaw0)}
        cfg.seed = SEED
        return cfg

    tasks.load_env_cfg = _pinned

    class VirtualPad:
        connected = True
        name = "virtual pad (flybrain sim-only)"
        path = "virtual"

        def __init__(self):
            self.s = controller.GamepadState()

        def state(self):
            return self.s

        def close(self):
            pass

    pad = VirtualPad()
    controller.open = lambda *a, **k: pad
    OUTD.mkdir(parents=True, exist_ok=True)

    def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
        import mujoco

        un = getattr(env, "unwrapped", env)
        sim = un.sim
        model, live = sim.env_mjdata(0)
        model.vis.global_.offwidth = max(VW, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(VH, model.vis.global_.offheight)
        rend = mujoco.Renderer(model, VH, VW)
        cid = model.camera("robot/onboard").id
        tid = model.camera("robot/tof").id
        fovy = float(model.cam_fovy[cid])
        f_px = (VH / 2) / math.tan(math.radians(fovy) / 2)
        # square pixels: horizontal focal equals vertical focal
        cam_hfov = 2 * math.atan((VW / 2) / f_px)
        tof_w = float(model.cam_sensorsize[tid][0])
        tof_f = float(model.cam_intrinsic[tid][0]) if hasattr(model, "cam_intrinsic") else 1.0
        # MuJoCo sensorsize is the full sensor; focal is in the same length units.
        tof_hfov = 2 * math.atan((tof_w / 2) / max(tof_f, 1e-6))
        tw, th = (int(v) for v in model.cam_resolution[tid])
        tof = un.scene["tof"]
        base_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")

        def _root(b):
            while model.body_parentid[b] != 0:
                b = model.body_parentid[b]
            return b

        robot_geom = np.array([
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, _root(model.geom_bodyid[g])) or "").startswith("robot")
            for g in range(model.ngeom)
        ])
        fly = Fly(rng)
        dt = float(un.step_dt)
        vis_every = max(1, round(VIS_DT / dt))
        q_every = max(1, round(0.04 / dt))
        obs = env.get_observations()
        step = 0
        falls = 0
        fallen = False
        wall_eps = 0
        in_wall = False
        fix_toys = set()
        logf = open(OUTD / "behaviours.jsonl", "w")
        qbuf, qt = [], []
        vis_meta, vis_tof = [], []
        t_wall = time.time()
        f_small = f_px / 4.0  # flow image is decimated by 4
        try:
            while step * dt < (4.5 if MODE == "walk" else TMAX):
                t = step * dt
                model, live = sim.env_mjdata(0)
                yaw = yaw_of(live.qpos[3:7])
                xy = np.array(live.qpos[:2], float)
                if MODE == "sim" and step % vis_every == 0 and t > 0.4:
                    rend.update_scene(live, camera="robot/onboard")
                    img = rend.render().copy()
                    rend.enable_segmentation_rendering()
                    rend.update_scene(live, camera="robot/onboard")
                    seg = rend.render()[..., 0].copy()
                    rend.disable_segmentation_rendering()
                    selfm = robot_geom[np.clip(seg, 0, robot_geom.size - 1)] & (seg >= 0)
                    rngs = tof.data.range[0].detach().cpu().numpy().reshape(th, tw)
                    tpose = (live.cam_xpos[tid].copy(), live.cam_xmat[tid].reshape(3, 3).copy(), float(model.cam_fovy[tid]))
                    cmd = fly.step(t, VIS_DT, img, selfm, rngs, yaw, xy, f_small, cam_hfov, tof_hfov, tof_pose=tpose)
                    if fly.beh == "FIXATE":
                        fix_toys.update(_truth_fixations(model, live, fly.beh, fly.sub, yaw, xy))
                    rec = {"t": round(t, 3), "beh": fly.beh, "sub": fly.sub, **cmd, **fly.overlay,
                           "xy": [round(float(xy[0]), 3), round(float(xy[1]), 3)], "yaw": round(yaw, 3)}
                    logf.write(json.dumps(rec) + "\n")
                    if SAVEVIS:
                        import imageio
                        (OUTD / "vis").mkdir(exist_ok=True)
                        imageio.imwrite(OUTD / "vis" / f"cam_{len(vis_meta):05d}.jpg", img, quality=85)
                        vis_tof.append(rngs.astype(np.float16))
                        vis_meta.append({
                            "t": round(t, 3), "beh": fly.beh, "sub": fly.sub,
                            "bbox": None if not fly.overlay.get("blob") else fly.overlay["blob"]["bbox"],
                            "flow": fly.overlay["flow"], "residual": fly.overlay["residual"],
                            "loom": fly.overlay["loom"], "near": fly.overlay["near"],
                            "pose": [round(float(xy[0]), 3), round(float(xy[1]), 3), round(yaw, 3)],
                        })
                elif MODE == "walk":
                    # scripted person: walk forward 2.2 s, then turn right 1.6 s
                    if t < 0.4:
                        cmd = {"ly": 0.0, "lx": 0.0, "rx": 0.0, "lb": False}
                    elif t < 2.6:
                        cmd = {"ly": -0.55, "lx": 0.0, "rx": 0.0, "lb": False}
                    else:
                        cmd = {"ly": 0.0, "lx": 0.0, "rx": 0.6, "lb": False}
                    fly.last = cmd
                else:
                    cmd = fly.last
                buttons = frozenset(["LB"]) if cmd.get("lb") else frozenset()
                pad.s = controller.GamepadState(lx=cmd.get("lx", 0), ly=cmd.get("ly", 0), rx=cmd.get("rx", 0), ry=0.0, buttons=buttons)
                with torch.inference_mode():
                    obs, _, _, _ = env.step(policy(obs))
                if step % 10 == 0:
                    model, live = sim.env_mjdata(0)
                    z = float(live.xpos[base_b][2])
                    upz = float(live.xmat[base_b].reshape(3, 3)[2, 2])
                    isf = z < 0.04 or upz < 0.5
                    if isf and not fallen:
                        falls += 1
                        print(f"[fall] t={t:.2f} z={z:.3f} up={upz:.2f}", flush=True)
                    fallen = isf
                    touching = False
                    for i in range(live.ncon):
                        g1, g2 = int(live.contact[i].geom1), int(live.contact[i].geom2)
                        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
                        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
                        if "wall" in n1.lower() or "wall" in n2.lower():
                            touching = True
                            break
                    if touching and not in_wall:
                        wall_eps += 1
                    in_wall = touching
                if (SAVEVIS or MODE == "walk") and step % q_every == 0:
                    qbuf.append(np.array(live.qpos, dtype=np.float64))
                    qt.append(t)
                step += 1
                if step % 4000 == 0:
                    print(f"[run] t={t:.0f}s wall={time.time()-t_wall:.0f}s beh={fly.beh} xy={np.round(xy,2)} falls={falls}", flush=True)
        finally:
            logf.close()
        model, live = sim.env_mjdata(0)
        end = np.array(live.qpos[:2], float)
        dist = float(np.hypot(*end))
        yaw1 = yaw_of(live.qpos[3:7])
        result = {
            "seed": SEED, "mode": MODE, "tmax": TMAX, "steps": step, "dt": dt,
            "yaw0": round(yaw0, 3), "end_xy": [round(float(end[0]), 3), round(float(end[1]), 3)],
            "path_m": round(dist, 3), "dyaw": round(wrap(yaw1 - yaw0), 3),
            "falls": falls, "wall_episodes": wall_eps,
            "time_s": {k: round(fly.time[k], 2) for k in LABELS},
            "counts": fly.counts, "fixated_toys_truth": sorted(fix_toys),
            "label": "SIM-ONLY VISION CONCEPT",
            "wall_s": round(time.time() - t_wall, 1),
        }
        if MODE == "walk":
            result["walk_ok"] = dist > 0.25 and abs(wrap(yaw1 - yaw0)) > 0.25
        json.dump(result, open(OUTD / "result.json", "w"), indent=1)
        if qbuf:
            np.savez_compressed(OUTD / "qpos.npz", qpos=np.stack(qbuf), t=np.array(qt))
        if SAVEVIS and vis_tof:
            np.savez_compressed(OUTD / "vision_tof.npz", tof=np.stack(vis_tof))
            json.dump(vis_meta, open(OUTD / "vision_meta.json", "w"))
        print("[result] " + json.dumps(result), flush=True)
        if falls:
            raise SystemExit(f"falls={falls}")
        if MODE == "walk" and not result["walk_ok"]:
            raise SystemExit(f"walk test failed: {result}")

    argv = [str(REPO / "scripts" / "play.py"), "--app", str(APP), "--headless",
            "--steps", "999999", "--backend", "native", "--device", "cpu"]
    if MAP:
        argv += ["--scene", MAP]
    sys.argv = argv
    play._loop = _loop
    play.main()


def unit():
    """No simulator. Synthetic images and dToF grids, one behaviour each."""
    rng = np.random.default_rng(0)
    H, W = 160, 240
    far = np.full((24, 32), 2.0, np.float32)
    black = np.zeros((H, W, 3), np.uint8)
    kw = dict(f_small=30.0, cam_hfov=1.9, tof_hfov=0.95)

    def call(fly, t, img, tof, yaw=0.0):
        fly.step(t, 0.1, img, None, tof, yaw, np.zeros(2), kw["f_small"], kw["cam_hfov"], kw["tof_hfov"])
        return fly.beh

    # straight walk: empty view, before the first saccade
    fly = Fly(rng)
    got = call(fly, 0.5, black, far)
    assert got == "WALK", got

    # looming: centre ranges collapse between two frames
    fly = Fly(np.random.default_rng(1))
    call(fly, 0.5, black, far)
    near = far.copy()
    near[:, 12:20] = 0.15
    got = call(fly, 0.6, black, near)
    assert got == "ESCAPE", (got, fly.overlay.get("loom"), fly.sub)

    # one-sided obstacle, nothing in front
    fly = Fly(np.random.default_rng(2))
    side = far.copy()
    side[:, :8] = 0.18
    got = call(fly, 0.5, black, side)
    assert got == "AVOID", (got, fly.overlay["near"])

    # saturated blob dead centre, depth beyond the standoff -> approach
    fly = Fly(np.random.default_rng(3))
    blob = black.copy()
    blob[40:110, 100:140] = (220, 40, 40)
    got = call(fly, 0.5, blob, far)
    assert got == "FIXATE" and fly.sub == "approach", (got, fly.sub, fly.overlay.get("blob"))

    # wide-field rightward motion, yaw held still -> optomotor turn to the right
    fly = Fly(np.random.default_rng(4))
    tex = np.random.default_rng(5).integers(90, 160, (H, W), np.uint8)
    a = np.repeat(tex[:, :, None], 3, axis=2)
    btex = np.zeros_like(tex)
    btex[:, 8:] = tex[:, :-8]
    b = np.repeat(btex[:, :, None], 3, axis=2)
    call(fly, 0.5, a, far, yaw=0.0)
    got = call(fly, 0.6, b, far, yaw=0.0)
    assert got == "OPTOMOTOR" and fly.last["rx"] > 0, (got, fly.last, fly.overlay.get("flow"), fly.overlay.get("residual"))

    # nothing salient and the saccade clock has elapsed
    fly = Fly(np.random.default_rng(6))
    fly.saccade_at = 0.2
    got = call(fly, 0.8, black, far)
    assert got == "SACCADE", got

    # --- floor (fix1): dToF pitched down at an empty floor, then pitched further down (the sag while frozen):
    # the floor gets closer in range, but it is floor -> WALK, never ESCAPE/AVOID
    def cam_R(pitch_down):
        # camera looks along +x (world), tilted down by pitch_down; columns = camera x (right), y (up), z (backward)
        c, s_ = math.cos(pitch_down), math.sin(pitch_down)
        fwd = np.array([c, 0.0, -s_]); right = np.array([0.0, -1.0, 0.0]); up = np.cross(right, fwd)
        return np.stack([right, up, -fwd], 1)

    def render_scene(pos, R, fovy, wall_x=None, box=None, shape=(42, 54)):
        h, w = shape
        f = (h / 2) / math.tan(math.radians(fovy) / 2)
        u = (np.arange(w) + 0.5 - w / 2) / f
        v = -(np.arange(h) + 0.5 - h / 2) / f
        uu, vv = np.meshgrid(u, v)
        d = np.stack([uu, vv, -np.ones_like(uu)], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        D = d @ R.T
        r = np.zeros((h, w))
        tf = np.where(D[..., 2] < -1e-6, -pos[2] / np.minimum(D[..., 2], -1e-6), np.inf)   # floor z=0
        r = np.where(tf < 3.0, tf, 0.0)
        if wall_x is not None:      # vertical wall at x = wall_x, up to 0.5 m high
            tw_ = np.where(D[..., 0] > 1e-6, (wall_x - pos[0]) / np.maximum(D[..., 0], 1e-6), np.inf)
            hit = (tw_ < np.where(r > 0, r, np.inf)) & (pos[2] + D[..., 2] * tw_ < 0.5) & (pos[2] + D[..., 2] * tw_ > 0)
            r = np.where(hit, tw_, r)
        if box is not None:         # vertical face at x = box[0], spanning y in [box[1], box[2]], 0.3 m high
            tb = np.where(D[..., 0] > 1e-6, (box[0] - pos[0]) / np.maximum(D[..., 0], 1e-6), np.inf)
            yb = pos[1] + D[..., 1] * tb; zb = pos[2] + D[..., 2] * tb
            hit = (tb < np.where(r > 0, r, np.inf)) & (yb > box[1]) & (yb < box[2]) & (zb > 0) & (zb < 0.3)
            r = np.where(hit, tb, r)
        return r.astype(np.float32)

    FOVY = 42.0
    pos = np.array([0.0, 0.0, 0.10])
    fly = Fly(np.random.default_rng(7))
    trace = []
    for i, pd in enumerate((0.35, 0.35, 0.55, 0.75, 0.85, 0.60)):        # level-ish, then sagging nose-down, then back up
        R = cam_R(pd)
        fly.step(0.5 + 0.1 * i, 0.1, black, None, render_scene(pos, R, FOVY), 0.0, np.zeros(2), kw["f_small"], kw["cam_hfov"], kw["tof_hfov"],
                 tof_pose=(pos, R, FOVY))
        trace.append(fly.beh)
    assert all(b == "WALK" for b in trace), (trace, fly.overlay)
    assert fly.overlay["floor_px"] > 100, fly.overlay
    # the same floor WITHOUT the pose (the old code path) is what used to escape: shows the test exercises the bug
    old = Fly(np.random.default_rng(7))
    old_trace = []
    for i, pd in enumerate((0.35, 0.55, 0.75, 0.85)):
        old.step(0.5 + 0.1 * i, 0.1, black, None, render_scene(pos, cam_R(pd), FOVY), 0.0, np.zeros(2), kw["f_small"], kw["cam_hfov"], kw["tof_hfov"])
        old_trace.append(old.beh)
    assert any(b in ("ESCAPE", "AVOID") for b in old_trace), old_trace

    # own pitch with a wall and a prop in view, no translation (the sag while frozen): no loom
    fly = Fly(np.random.default_rng(10))
    fly.saccade_at = 99.0
    tr = []
    for i, pd in enumerate((0.30, 0.45, 0.62, 0.80, 0.50)):
        R = cam_R(pd)
        fly.step(0.5 + 0.1 * i, 0.1, black, None, render_scene(pos, R, FOVY, wall_x=1.0, box=(0.55, -0.05, 0.12)), 0.0, np.zeros(2),
                 kw["f_small"], kw["cam_hfov"], kw["tof_hfov"], tof_pose=(pos, R, FOVY))
        tr.append((fly.beh, fly.overlay["loom"]))
    assert all(b[1] is None for b in tr) and all(b[0] != "ESCAPE" for b in tr), tr

    # own turning past a prop edge, no translation: no loom
    def cam_R_yaw(pitch_down, yaw):
        c, s_ = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        return Rz @ cam_R(pitch_down)
    fly = Fly(np.random.default_rng(11))
    fly.saccade_at = 99.0
    tr = []
    for i, yw in enumerate((0.0, 0.12, 0.24, 0.36, 0.48)):
        R = cam_R_yaw(0.35, yw)
        fly.step(0.5 + 0.1 * i, 0.1, black, None, render_scene(pos, R, FOVY, wall_x=1.4, box=(0.6, 0.05, 0.5)), yw, np.zeros(2),
                 kw["f_small"], kw["cam_hfov"], kw["tof_hfov"], tof_pose=(pos, R, FOVY))
        tr.append((fly.beh, fly.overlay["loom"]))
    assert all(b[1] is None for b in tr), tr

    # a real wall coming closer, with the pose: still ESCAPE (looming above the floor)
    fly = Fly(np.random.default_rng(8))
    R = cam_R(0.35)
    fly.step(0.5, 0.1, black, None, render_scene(pos, R, FOVY, wall_x=0.70), 0.0, np.zeros(2), kw["f_small"], kw["cam_hfov"], kw["tof_hfov"], tof_pose=(pos, R, FOVY))
    p2 = pos + np.array([0.15, 0.0, 0.0])        # 1.5 m/s closing over one 0.1 s tick
    fly.step(0.6, 0.1, black, None, render_scene(p2, R, FOVY, wall_x=0.70), 0.0, np.zeros(2), kw["f_small"], kw["cam_hfov"], kw["tof_hfov"], tof_pose=(p2, R, FOVY))
    assert fly.beh == "ESCAPE" and fly.sub == "loom", (fly.beh, fly.sub, fly.overlay)

    # a prop close on the left side, with the pose: AVOID (turn right, away)
    fly = Fly(np.random.default_rng(9))
    R = cam_R(0.35)
    fly.step(0.5, 0.1, black, None, render_scene(pos, R, FOVY, box=(0.22, 0.03, 0.4)), 0.0, np.zeros(2), kw["f_small"], kw["cam_hfov"], kw["tof_hfov"],
             tof_pose=(pos, R, FOVY))
    assert fly.beh == "AVOID" and fly.last["rx"] > 0, (fly.beh, fly.sub, fly.overlay)

    # --- fix2: every non-zero rule turn reaches the yaw band (|rx| > 0.5), linear in t; zero stays zero
    for tt in np.linspace(-0.65, 0.65, 131):
        r = stick_turn(tt)
        if abs(tt) <= TURN_EPS:
            assert r == 0.0, (tt, r)
        else:
            assert 0.5 < abs(r) <= 1.0 and math.copysign(1, r) == math.copysign(1, tt), (tt, r)
            assert abs((abs(r) - 0.5) / 0.5 - min(1.0, abs(tt) / TURN_FULL)) < 1e-9, (tt, r)
    assert stick_turn(2.0) == 1.0 and stick_turn(-2.0) == -1.0
    # and through the behaviours: AVOID (0.50 -> 0.885), OPTOMOTOR, FIXATE centre / approach / standoff, SACCADE, ESCAPE
    seen = {}
    def rec(f):
        if f.last["rx"] != 0.0:
            assert abs(f.last["rx"]) > 0.5, (f.beh, f.sub, f.last)
            seen.setdefault(f.beh + "/" + f.sub, round(f.last["rx"], 3))
    f = Fly(np.random.default_rng(12)); call(f, 0.5, black, side); rec(f)                     # AVOID left
    assert f.beh == "AVOID" and abs(f.last["rx"] - stick_turn(0.50)) < 1e-9, f.last
    f.step(0.6, 0.1, black, None, far, 0.0, np.zeros(2), kw["f_small"], kw["cam_hfov"], kw["tof_hfov"]); rec(f)  # AVOID hold
    assert f.sub == "hold" and abs(f.last["rx"] - stick_turn(0.50)) < 1e-9, (f.sub, f.last)
    f = Fly(np.random.default_rng(4)); call(f, 0.5, a, far); call(f, 0.6, b, far); rec(f)     # OPTOMOTOR
    for x0 in (60, 112, 118, 150, 180):                                                      # FIXATE at several bearings
        f = Fly(np.random.default_rng(13)); im = black.copy(); im[40:110, x0:x0 + 16] = (220, 40, 40)
        for nearr in (2.0, 0.40):
            tofz = far.copy(); tofz[:, :] = nearr
            call(f, 0.5, im, tofz); rec(f)
    f = Fly(np.random.default_rng(6)); f.saccade_at = 0.2; call(f, 0.8, black, far); rec(f)  # SACCADE (raw 0.62)
    assert abs(abs(f.last["rx"]) - 0.62) < 1e-9, f.last
    f = Fly(np.random.default_rng(1)); call(f, 0.5, black, far); call(f, 0.6, black, near)   # ESCAPE: freeze, then turn
    for k in range(16):
        f.step(0.7 + 0.1 * k, 0.1, black, None, far, 0.0, np.zeros(2), kw["f_small"], kw["cam_hfov"], kw["tof_hfov"]); rec(f)
    assert "ESCAPE/loom" in seen and abs(abs(seen["ESCAPE/loom"]) - 0.55) < 1e-9, seen
    assert any(k.startswith("FIXATE/") for k in seen) and "AVOID/left" in seen and "OPTOMOTOR/stabilise" in seen, seen

    print("[unit] WALK ESCAPE AVOID FIXATE/approach OPTOMOTOR SACCADE ok; floor-only (pitching) -> WALK ok; "
          "own pitch/turn with wall+prop in view -> no loom ok; wall loom -> ESCAPE ok; side prop -> AVOID ok; "
          f"every non-zero turn -> |rx|>0.5 ok {json.dumps(seen)}", flush=True)


if __name__ == "__main__":
    if os.environ.get("FB_UNIT") == "1":
        unit()
    else:
        _run()
