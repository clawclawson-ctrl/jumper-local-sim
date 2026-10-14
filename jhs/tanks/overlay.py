"""Crab Tanks overlay (live + MP4). SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from jhs import PKG
from jhs.overlay import tof_image
from jhs.soccer import field as F
from . import arena as A

TC = {"A": (235, 60, 40), "B": (60, 110, 240)}
DC = {"opponent": (255, 120, 255), "shot": (60, 240, 255)}
STATE = {"kickoff": "kickoff", "search": "looking for the opponent", "explore": "EXPLORE", "hunt": "HUNT",
         "cover": "COVER", "dodge": "DODGE", "backoff": "backing off (stuck)", "stunned": "STUNNED",
         "celebrate": "HIT! celebrating", "paused": "paused (hit)", "return": "RETURNING TO START",
         "push_obst": "PUSHING OBSTACLE", "referee": "referee reset"}


class TanksOverlay:
    def __init__(self, W=1920, H=1080):
        self.W, self.H = W, H; s = H / 1080.0; self.s = s
        Fn = lambda name, size: ImageFont.truetype(str(PKG / "fonts" / name), max(8, int(size * s)))
        self.fb = Fn("DejaVuSans-Bold.ttf", 46); self.f2 = Fn("DejaVuSans-Bold.ttf", 22); self.f3 = Fn("DejaVuSans.ttf", 18)
        self.f4 = Fn("DejaVuSans-Bold.ttf", 18); self.f5 = Fn("DejaVuSans-Bold.ttf", 26); self.f6 = Fn("DejaVuSans.ttf", 15)
        self.fdet = ImageFont.truetype(str(PKG / "fonts" / "DejaVuSans-Bold.ttf"), 13)

    def _cam(self, img, k, m, cam_img, tof, x0, y0):
        s = self.s; S = lambda v: int(round(v * s)); d = ImageDraw.Draw(img, "RGBA")
        PW, PH, TW, TH = S(400), S(300), S(54 * 3), S(42 * 3)
        d.rectangle([x0 - S(8), y0 - S(34), x0 + PW + S(8), y0 + PH + TH + S(64)], fill=(0, 0, 0, 155))
        d.rectangle([x0 - S(8), y0 - S(34), x0 + PW + S(8), y0 - S(30)], fill=TC[k] + (255,))
        d.text((x0, y0 - S(28)), f"{F.TEAM[k]} crab's own camera", font=self.f4, fill=TC[k])
        if cam_img is not None:
            ci = (cam_img if isinstance(cam_img, Image.Image) else Image.fromarray(cam_img)).convert("RGB").resize((640, 480), Image.BILINEAR)
            cd = ImageDraw.Draw(ci)
            for det in (m or {}).get("dets", []):
                a, b, c_, e = det["bbox"]; col = DC.get(det["cls"], (255, 255, 255))
                cd.rectangle([a - 3, b - 3, c_ + 3, e + 3], outline=col, width=3)
                cd.text((a - 3, max(0, b - 17)), f'{det["cls"]} {det["rng"]:.1f} m', fill=col, font=self.fdet)
            img.paste(ci.resize((PW, PH), Image.BILINEAR), (x0, y0))
        if tof is not None:
            img.paste(Image.fromarray(tof_image(np.asarray(tof, np.float32))).resize((TW, TH), Image.NEAREST), (x0, y0 + PH + S(6)))
        st = (m or {}).get("state", ""); sub = (m or {}).get("sub", "")
        d.text((x0 + TW + S(10), y0 + PH + S(8)), STATE.get(st, st), font=self.f5, fill=(255, 235, 120))
        d.text((x0 + TW + S(10), y0 + PH + S(42)), sub[:36], font=self.f6, fill=(235, 235, 235))
        oa = (m or {}).get("opp_age")
        d.text((x0 + TW + S(10), y0 + PH + S(64)), "opp: " + ("in view" if oa is not None and oa < 0.25 else f"last seen {oa:.0f} s ago" if oa is not None and oa < 90 else "not seen yet"),
               font=self.f6, fill=(235, 235, 235))
        d.text((x0, y0 + PH + TH + S(10)), "dToF 54x42 · boxes: opponent / shot (own vision)", font=self.f6, fill=(220, 220, 220))
        tur = (m or {}).get("turret")
        if tur is not None: d.text((x0, y0 + PH + TH + S(32)), f"turret {tur:+.0f} deg (within FOV)", font=self.f6, fill=(220, 220, 220))

    def draw(self, img, meta, camA, camB, tofA, tofB, t, tmax, speed_label="1x speed", props=None, shots=None):
        W, H, s = self.W, self.H, self.s; S = lambda v: int(round(v * s)); d = ImageDraw.Draw(img, "RGBA")
        meta = meta or {}
        self._cam(img, "A", meta.get("A"), camA, tofA, S(22), S(150))
        self._cam(img, "B", meta.get("B"), camB, tofB, W - S(22) - S(400), S(150))
        sc = meta.get("score", {"A": 0, "B": 0})
        tl = max(0.0, tmax - meta.get("clock", t)); clock = f"{int(tl // 60)}:{int(tl % 60):02d}"
        txt_r, txt_mid, txt_b = "RED", f"{sc.get('A', 0)} - {sc.get('B', 0)}", "BLUE"
        wr, wm, wb = (int(d.textlength(x, font=self.fb)) for x in (txt_r, txt_mid, txt_b))
        tot = wr + wm + wb + S(80); x0 = W // 2 - tot // 2
        d.rectangle([x0 - S(20), S(14), x0 + tot + S(20), S(118)], fill=(0, 0, 0, 185))
        d.text((x0, S(20)), txt_r, font=self.fb, fill=TC["A"]); d.text((x0 + wr + S(40), S(20)), txt_mid, font=self.fb, fill=(255, 255, 255))
        d.text((x0 + wr + wm + S(80), S(20)), txt_b, font=self.fb, fill=TC["B"])
        cl = f"match clock {clock}  ·  first to {meta.get('hits_to_win', 3)} hits"; d.text((W // 2 - int(d.textlength(cl, font=self.f2)) // 2, S(80)), cl, font=self.f2, fill=(230, 230, 230))
        d.rectangle([S(22), S(14), S(470), S(56)], fill=(150, 20, 20, 210)); d.text((S(34), S(20)), "SIM-ONLY VISION CONCEPT", font=self.f5, fill=(255, 255, 255))
        d.text((S(26), S(64)), "Jumper Crab Tanks 1v1 -- each crab sees only", font=self.f3, fill=(255, 255, 255))
        d.text((S(26), S(88)), "with its own sim camera + dToF + pose · slow Combat shots", font=self.f3, fill=(255, 255, 255))
        MW = S(520); MH = int(MW * (2 * F.FY + 0.3) / (2 * F.FX + 1.0)); mx0 = W // 2 - MW // 2; my0 = H - MH - S(150)
        d.rectangle([mx0 - S(10), my0 - S(34), mx0 + MW + S(10), my0 + MH + S(54)], fill=(0, 0, 0, 160))
        d.text((mx0, my0 - S(30)), "top view: each crab's OWN belief · cyan = shots (SIM TRUTH)", font=self.f6, fill=(255, 255, 255))
        def P(x, y): return mx0 + (x + F.FX + 0.5) / (2 * F.FX + 1.0) * MW, my0 + (F.FY + 0.15 - y) / (2 * F.FY + 0.3) * MH
        d.rounded_rectangle([*P(-F.FX, F.FY), *P(F.FX, -F.FY)], radius=F.CORNER_R / (2 * F.FX + 1.0) * MW, fill=(48, 100, 50), outline=(230, 230, 210), width=max(1, S(2)))
        d.line([P(0, F.FY), P(0, -F.FY)], fill=(230, 230, 210), width=1)
        for _k, Pp in (props or []):
            if max(abs(x) for x, _ in Pp) > F.FX + 0.3 or max(abs(y) for _, y in Pp) > F.FY + 0.3: continue
            d.polygon([P(x, y) for x, y in Pp], outline=(235, 235, 235), width=max(1, S(1.5)))
        for sh in (shots or meta.get("shots") or []):
            sx, sy = P(sh["x"], sh["y"]); k = S(5)
            d.ellipse([sx - k, sy - k, sx + k, sy + k], fill=(60, 240, 255), outline=(40, 30, 0))
            a = math.atan2(sh.get("vy", 0), sh.get("vx", 1)); d.line([sx, sy, sx + S(14) * math.cos(-a), sy + S(14) * math.sin(-a)], fill=(160, 250, 255), width=max(1, S(2)))
        for k in ("A", "B"):
            m = meta.get(k) or {}
            if m.get("opp"):
                ox, oy = P(*m["opp"]); q = S(8); d.rectangle([ox - q, oy - q, ox + q, oy + q], outline=TC[k], width=max(1, S(2)))
            if m.get("pose"):
                x, y, yw = m["pose"]; qx, qy = P(x, y); a, b = S(15), S(10)
                d.polygon([(qx + a * math.cos(-yw), qy + a * math.sin(-yw)), (qx + b * math.cos(-yw + 2.4), qy + b * math.sin(-yw + 2.4)),
                           (qx + b * math.cos(-yw - 2.4), qy + b * math.sin(-yw - 2.4))], fill=TC[k], outline=(255, 255, 255))
                tw = m.get("turret_rad")
                if tw is not None:
                    d.line([qx, qy, qx + S(22) * math.cos(-tw), qy + S(22) * math.sin(-tw)], fill=(255, 255, 255), width=max(1, S(2)))
        d.text((mx0, my0 + MH + S(6)), "square = where that crab thinks the other is; white line = its turret", font=self.f6, fill=(230, 230, 230))
        d.text((mx0, my0 + MH + S(26)), "cyan dots = projectiles (SIM TRUTH, display only); outlines = obstacles", font=self.f6, fill=(230, 230, 230))
        ref = meta.get("ref", "")
        d.rectangle([S(22), H - S(66), S(1100), H - S(20)], fill=(0, 0, 0, 175))
        d.text((S(34), H - S(58)), f"REFEREE (SIM TRUTH): {ref}", font=self.f2, fill=(200, 255, 200))
        clock = f"sim time t = {t:5.1f} s   ({speed_label})"; cw = int(d.textlength(clock, font=self.f2)) + S(28)
        d.rectangle([W - cw - S(14), H - S(66), W - S(14), H - S(20)], fill=(0, 0, 0, 175))
        d.text((W - cw, H - S(58)), clock, font=self.f2, fill=(240, 240, 240))
        g = meta.get("final")
        if not g and any((meta.get(k) or {}).get("state") == "return" for k in ("A", "B")): g = "RETURNING TO START"
        elif ref.startswith("HIT") and any((meta.get(k) or {}).get("state") in ("celebrate", "paused") for k in ("A", "B")): g = ref.split(" at ")[0] + "!"
        if g:
            gw = int(d.textlength(g, font=self.fb))
            d.rectangle([W // 2 - gw // 2 - S(30), H // 2 - S(60), W // 2 + gw // 2 + S(30), H // 2 + S(10)], fill=(20, 20, 20, 200))
            d.text((W // 2 - gw // 2, H // 2 - S(54)), g, font=self.fb, fill=(255, 230, 90))
        return img


class BroadcastCam:
    def __init__(self):
        import mujoco
        self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE; self.x = 0.0

    def update(self, look_xy=None):
        if look_xy is not None: self.x += (0.35 * float(look_xy[0]) - self.x) * 0.05
        self.cam.lookat[:] = [self.x, -0.35, 0.0]; self.cam.distance = 5.4; self.cam.azimuth = 90.0; self.cam.elevation = -48.0
        return self.cam
