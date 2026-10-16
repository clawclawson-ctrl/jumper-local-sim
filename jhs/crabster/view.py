"""Crabster picture: cinematic camera (Follow / Fixed), overlay, and one function that draws a frame -- shared by the live view
and the MP4. DISPLAY ONLY. SCRIPTED STUNT, SIM-ONLY THRUSTER + SUCTION CONCEPT."""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from jhs import PKG
from jhs.crabster import skin as SK, flight as F

TITLE = "CRABSTER"; LABEL = "SCRIPTED STUNT, SIM-ONLY THRUSTER + SUCTION CONCEPT"
ACT = {"ramp": "ARCH: climb, hang inverted, release, roll, land", "kickflip": "KICKFLIP over five cars", "sprint": "0-1.5 m/s SPRINT (sim)"}


class Cam:
    """'follow': a slow cinematic follow -- eased look-at (critically damped), a gentle orbit, pulls back on the arch and in flight.
    'fixed': the grandstand view of the whole stage."""
    W0 = 1.6
    def __init__(self, mode="follow"):
        import mujoco
        self.mode = (mode or "follow").lower(); self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.s = None; self.v = None; self.t = None
    def update(self, t, p, phase=""):
        c = self.cam
        if self.mode == "fixed":
            c.lookat[:] = (0.1, 0.0, 0.55); c.distance = 5.6; c.azimuth = 90.0; c.elevation = -13.0; return c
        ph = phase.split(":")[-1]; air = ph in ("release", "land", "lift", "flip")
        tgt = np.array([p[0], p[1] * 0.6, max(0.15, p[2] + (0.12 if air else 0.0)), 3.4 if ph in ("climb", "hang", "release") else (2.6 if air else 2.1)])
        if self.s is None: self.s = tgt.copy(); self.v = np.zeros(4); self.t = t
        dt = float(np.clip(t - self.t, 0.0, 0.1)); self.t = t; w = self.W0 * (1.8 if air else 1.0)
        for _ in range(4):
            h = dt / 4; a = w * w * (tgt - self.s) - 2 * w * self.v; self.v += a * h; self.s += self.v * h
        c.lookat[:] = self.s[:3]; c.distance = float(self.s[3]); c.azimuth = 70.0 + 18.0 * math.sin(t * 0.07); c.elevation = -10.0
        return c


class Overlay:
    def __init__(self, W=1920, H=1080):
        self.W, self.H = W, H; s = H / 1080.0; self.s = s
        Fn = lambda n, z: ImageFont.truetype(str(PKG / "fonts" / n), max(8, int(z * s)))
        self.ft = Fn("DejaVuSans-Bold.ttf", 54); self.fl = Fn("DejaVuSans-Bold.ttf", 19); self.fv = Fn("DejaVuSans-Bold.ttf", 24)
        self.fs = Fn("DejaVuSans.ttf", 18); self.fm = Fn("DejaVuSans-Bold.ttf", 30); self.fh = Fn("DejaVuSans-Bold.ttf", 110)

    def draw(self, img, t, gas, gas0, suction, g, phase, msg=None, results=(), speed_label="1x speed", title_card=0.0):
        s = self.s; S = lambda v: int(round(v * s)); d = ImageDraw.Draw(img, "RGBA")
        if title_card > 0:                                                        # MP4 title card (fades out)
            a = int(255 * min(1.0, title_card))
            d.rectangle([0, 0, self.W, self.H], fill=(0, 0, 0, int(0.75 * a)))
            tw = d.textlength(TITLE, font=self.fh); d.text(((self.W - tw) / 2, self.H * 0.36), TITLE, font=self.fh, fill=(255, 255, 255, a))
            lw = d.textlength(LABEL, font=self.fm); d.text(((self.W - lw) / 2, self.H * 0.36 + S(140)), LABEL, font=self.fm, fill=(255, 70, 60, a))
        x0, y0 = S(28), S(24)
        d.rectangle([x0 - S(12), y0 - S(10), x0 + S(520), y0 + S(272)], fill=(0, 0, 0, 150))
        d.text((x0, y0), TITLE, font=self.ft, fill=(255, 255, 255)); d.rectangle([x0, y0 + S(66), x0 + S(500), y0 + S(69)], fill=(220, 30, 30))
        d.text((x0, y0 + S(76)), LABEL, font=self.fl, fill=(255, 110, 100))
        y = y0 + S(110)
        d.text((x0, y), "SIM CLOCK", font=self.fs, fill=(170, 170, 180)); d.text((x0 + S(150), y - S(3)), f"{t:6.2f} s", font=self.fv, fill=(255, 255, 255))
        y += S(36); fr = max(0.0, gas / gas0) if gas0 else 0.0
        d.text((x0, y), "GAS (N2)", font=self.fs, fill=(170, 170, 180))
        bx = x0 + S(150); d.rectangle([bx, y + S(2), bx + S(250), y + S(22)], outline=(200, 200, 210), width=max(1, S(2)))
        col = (80, 200, 255) if fr > 0.25 else (255, 170, 40) if fr > 0.08 else (255, 60, 50)
        d.rectangle([bx + S(3), y + S(5), bx + S(3) + int((S(250) - S(6)) * fr), y + S(19)], fill=col)
        d.text((bx + S(262), y - S(1)), f"{gas * 1000:.0f} g", font=self.fs, fill=(255, 255, 255))
        y += S(36); on = suction > 0.5
        d.text((x0, y), "SUCTION", font=self.fs, fill=(170, 170, 180))
        d.text((x0 + S(150), y - S(3)), f"ON  {suction:.0f} N" if on else "off", font=self.fv, fill=(90, 255, 140) if on else (150, 150, 155))
        y += S(36)
        d.text((x0, y), "G-FORCE", font=self.fs, fill=(170, 170, 180)); d.text((x0 + S(150), y - S(3)), f"{g:4.1f} g", font=self.fv, fill=(255, 230, 120) if g > 2 else (255, 255, 255))
        act = phase.split(":")[0] if phase else ""
        if act in ACT:
            d.text((x0, y + S(38)), ACT[act] + ("  -  " + phase.split(":")[1] if ":" in phase else ""), font=self.fs, fill=(220, 220, 230))
        if msg:                                                                   # the stunt caller, bottom centre
            mw = d.textlength(msg, font=self.fm); mx = (self.W - mw) / 2; my = self.H - S(120)
            d.rectangle([mx - S(18), my - S(10), mx + mw + S(18), my + S(44)], fill=(0, 0, 0, 160)); d.text((mx, my), msg, font=self.fm, fill=(255, 255, 255))
        ry = self.H - S(60)
        for r in results:
            txt = ("SUCCESS " if r["success"] else "FAILED ") + f"{r['act']}: {r['why']}"
            d.text((x0, ry), txt, font=self.fs, fill=(90, 255, 140) if r["success"] else (255, 80, 70)); ry -= S(26)
        lw = d.textlength(speed_label, font=self.fs); d.text((self.W - lw - S(24), self.H - S(36)), speed_label, font=self.fs, fill=(200, 200, 205))
        return img


class Painter:
    """renders the stage + crab with the Roadster skin and decor from a qpos, then the overlay."""
    def __init__(self, rmodel, W, H, cam_mode="follow"):
        import mujoco
        self.mj = mujoco; self.m = rmodel; SK.skin(rmodel)
        rmodel.vis.global_.offwidth = max(W, rmodel.vis.global_.offwidth); rmodel.vis.global_.offheight = max(H, rmodel.vis.global_.offheight)
        self.d = mujoco.MjData(rmodel); self.rend = mujoco.Renderer(rmodel, H, W, max_geom=12000)
        self.b = mujoco.mj_name2id(rmodel, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link"); self.ext = None
        self.cam = Cam(cam_mode); self.ov = Overlay(W, H)
    def frame(self, qpos, t, rec, msg=None, results=(), speed_label="1x speed", title_card=0.0):
        mj = self.mj; self.d.qpos[:] = qpos[: self.m.nq]; mj.mj_forward(self.m, self.d)
        if self.ext is None: self.ext = SK.base_extent(self.m, self.d, self.b)
        self.rend.update_scene(self.d, self.cam.update(t, self.d.xpos[self.b].copy(), rec.get("phase", "")))
        SK.decor(self.rend.scene, self.d, self.b, *self.ext, lift=rec.get("lift", 0.0), tq=rec.get("tq", (0, 0, 0)), push=rec.get("push", (0, 0, 0)))
        img = Image.fromarray(self.rend.render())
        return self.ov.draw(img, t, rec.get("gas", F.GAS0), F.GAS0, rec.get("suction", 0.0), rec.get("g", 1.0), rec.get("phase", ""), msg, results, speed_label, title_card)


def caption(events, t, hold=3.0):
    """the latest stunt call within `hold` seconds."""
    m = None
    for e in events:
        if e["t"] <= t and t - e["t"] < hold: m = e["msg"]
    return m


def unit():
    out = []
    ov = Overlay(640, 360); img = Image.new("RGB", (640, 360))
    ov.draw(img, 12.3, 0.3, 0.6, 45.0, 2.4, "kickflip:flip", "side thrusters -- 360 kickflip", [dict(act="ramp", success=True, why="landed")], title_card=0.5)
    assert np.asarray(img).mean() > 1
    c = Cam("follow"); p = np.array([1.0, 0.0, 0.5])
    for k in range(200): c.update(k * 0.04, p, "kickflip:approach")
    assert np.linalg.norm(np.array(c.cam.lookat) - np.array([1.0, 0.0, 0.5])) < 0.02
    for k in range(200): c.update(8 + k * 0.04, p, "kickflip:flip")
    assert abs(c.cam.lookat[2] - 0.62) < 0.02 and c.cam.distance > 2.5
    f = Cam("fixed"); f.update(0, p); assert f.cam.distance > 5
    assert caption([dict(t=1.0, msg="a"), dict(t=2.0, msg="b")], 2.5) == "b" and caption([dict(t=1.0, msg="a")], 9.0) is None
    out.append("overlay draws (title card, gas, suction, g, caller); Follow cam settles on the crab, Fixed is the grandstand")
    return out
