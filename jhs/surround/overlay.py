"""Crab Surround overlay (live + MP4), synthwave style. Compact insets so the arena dominates the frame. SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from jhs import PKG
from jhs.overlay import tof_image
from . import game as G

NC = {"A": (255, 40, 210), "B": (0, 240, 255)}          # neon magenta / cyan (trail colours)
TN = {"A": "RED", "B": "BLUE"}
DC = {"opponent": (255, 255, 90)}
STATE = {"start": "ready", "opener": "OPENER", "cutoff": "CUT OFF", "race": "RACE", "hunt": "HUNT", "escape": "ESCAPE!", "fill": "FILL POCKET", "trapped": "TRAPPED",
         "out": "OUT", "return": "WALKING BACK", "celebrate": "WINS ROUND", "paused": "round over", "referee": "referee", "over": "over"}


def _ag(a):
    if isinstance(a, dict): return f" · aggression R {a.get('A', 60):.0f}% / B {a.get('B', 60):.0f}%"
    return f" · aggression {float(a):.0f}%"


class SurroundOverlay:
    def __init__(self, W=1920, H=1080):
        self.W, self.H = W, H; s = H / 1080.0; self.s = s
        Fn = lambda name, size: ImageFont.truetype(str(PKG / "fonts" / name), max(8, int(size * s)))
        self.fb = Fn("DejaVuSans-Bold.ttf", 40); self.f2 = Fn("DejaVuSans-Bold.ttf", 19); self.f3 = Fn("DejaVuSans.ttf", 14)
        self.f4 = Fn("DejaVuSans-Bold.ttf", 15); self.f5 = Fn("DejaVuSans-Bold.ttf", 18); self.f6 = Fn("DejaVuSans.ttf", 12)
        self.fdet = ImageFont.truetype(str(PKG / "fonts" / "DejaVuSans-Bold.ttf"), 13)

    def _panel(self, d, box, col):
        S = lambda v: int(round(v * self.s))
        d.rectangle(box, fill=(8, 0, 20, 170), outline=col + (230,), width=max(1, S(2)))

    def _glow(self, d, xy, txt, font, col):
        x, y = xy
        o = max(1, int(round(1.5 * self.s)))
        for dx, dy in ((-o, 0), (o, 0), (0, -o), (0, o)): d.text((x + dx, y + dy), txt, font=font, fill=tuple(c // 3 for c in col) + (90,))
        d.text((x, y), txt, font=font, fill=col)

    def _cam(self, img, k, m, cam_img, tof, x0, y0):
        s = self.s; S = lambda v: int(round(v * s)); d = ImageDraw.Draw(img, "RGBA")
        PW, PH, TW, TH = S(240), S(180), S(108), S(84)
        self._panel(d, [x0 - S(6), y0 - S(24), x0 + PW + TW + S(14), y0 + PH + S(44)], NC[k])
        d.text((x0, y0 - S(21)), f"{TN[k]} crab cam ({G.NEON[k].lower()} trail)", font=self.f4, fill=NC[k])
        if cam_img is not None:
            ci = (cam_img if isinstance(cam_img, Image.Image) else Image.fromarray(cam_img)).convert("RGB").resize((640, 480), Image.BILINEAR)
            cd = ImageDraw.Draw(ci)
            for det in (m or {}).get("dets", []):
                a, b, c_, e = det["bbox"]; col = DC.get(det["cls"], (255, 255, 255))
                cd.rectangle([a - 3, b - 3, c_ + 3, e + 3], outline=col, width=4)
                cd.text((a - 3, max(0, b - 17)), f'{det["cls"]} {det["rng"]:.1f} m', fill=col, font=self.fdet)
            img.paste(ci.resize((PW, PH), Image.BILINEAR), (x0, y0))
        if tof is not None:
            img.paste(Image.fromarray(tof_image(np.asarray(tof, np.float32))).resize((TW, TH), Image.NEAREST), (x0 + PW + S(6), y0))
            d.text((x0 + PW + S(6), y0 + TH + S(2)), "depth (dToF)", font=self.f6, fill=(200, 200, 230))
        st = (m or {}).get("state", ""); sub = (m or {}).get("sub", "")
        d.text((x0 + PW + S(6), y0 + TH + S(20)), STATE.get(st, st)[:13], font=self.f5, fill=(255, 230, 120))
        oa = (m or {}).get("opp_age")
        d.text((x0 + PW + S(6), y0 + TH + S(44)), "opp " + ("in view" if oa is not None and oa < 0.25 else f"{oa:.0f} s ago" if oa is not None and oa < 90 else "unseen"), font=self.f6, fill=(220, 220, 240))
        if (m or {}).get("opener"): d.text((x0 + PW + S(6), y0 + TH + S(60)), m["opener"][:14] + ("" if m.get("opener_on") else " ✓"), font=self.f6, fill=(150, 220, 255))
        d.text((x0, y0 + PH + S(4)), sub[:40], font=self.f6, fill=(235, 235, 245))
        d.text((x0, y0 + PH + S(22)), "box = opponent (own vision)", font=self.f6, fill=(190, 190, 210))

    def draw(self, img, meta, camA, camB, tofA, tofB, t, tmax, speed_label="1x speed", trails=None):
        W, H, s = self.W, self.H, self.s; S = lambda v: int(round(v * s)); d = ImageDraw.Draw(img, "RGBA")
        meta = meta or {}
        # title + label (top-left)
        d.rectangle([S(16), S(12), S(330), S(44)], fill=(150, 20, 60, 215)); d.text((S(26), S(16)), "SIM-ONLY VISION CONCEPT", font=self.f5, fill=(255, 255, 255))
        self._glow(d, (S(18), S(50)), "CRAB SURROUND", self.f2, (255, 90, 220))
        d.text((S(18), S(74)), "own cam + dToF + pose only · trails = sim add-on", font=self.f6, fill=(220, 220, 255))
        # score (top-centre)
        sc = meta.get("score", {"A": 0, "B": 0})
        tl = max(0.0, tmax - meta.get("clock", t)); clock = f"{int(tl // 60)}:{int(tl % 60):02d}"
        a_, m_, b_ = "RED", f"{sc.get('A', 0)} : {sc.get('B', 0)}", "BLUE"
        wa, wm, wb = (int(d.textlength(x, font=self.fb)) for x in (a_, m_, b_)); tot = wa + wm + wb + S(60); x0 = W // 2 - tot // 2
        d.rectangle([x0 - S(16), S(10), x0 + tot + S(16), S(92)], fill=(8, 0, 20, 185), outline=(255, 60, 200, 200), width=max(1, S(2)))
        self._glow(d, (x0, S(12)), a_, self.fb, NC["A"]); d.text((x0 + wa + S(30), S(12)), m_, font=self.fb, fill=(255, 255, 255))
        self._glow(d, (x0 + wa + wm + S(60), S(12)), b_, self.fb, NC["B"])
        cl = f"round {meta.get('round', 1)} · {meta.get('round_t', 0):.0f} s · clock {clock} · first to {meta.get('rounds_to_win', 3)}" + _ag(meta.get('aggression', 60))
        d.text((W // 2 - int(d.textlength(cl, font=self.f3)) // 2, S(66)), cl, font=self.f3, fill=(230, 220, 255))
        # compact crab insets: RED bottom-left, BLUE top-right
        self._cam(img, "A", meta.get("A"), camA, tofA, S(22), H - S(320))
        self._cam(img, "B", meta.get("B"), camB, tofB, W - S(22) - S(360), S(130))
        # small minimap (bottom-right): trails (SIM TRUTH) + each crab's own belief markers
        MW = S(250); MH = int(MW * (2 * G.FY) / (2 * G.FX)); mx0 = W - MW - S(26); my0 = H - MH - S(90)
        self._panel(d, [mx0 - S(6), my0 - S(22), mx0 + MW + S(6), my0 + MH + S(20)], (170, 90, 255))
        d.text((mx0, my0 - S(19)), "top view · trails SIM TRUTH", font=self.f6, fill=(240, 240, 255))
        def P(x, y): return mx0 + (x + G.FX) / (2 * G.FX) * MW, my0 + (G.FY - y) / (2 * G.FY) * MH
        d.rounded_rectangle([*P(-G.FX, G.FY), *P(G.FX, -G.FY)], radius=G.CORNER_R / (2 * G.FX) * MW, fill=(10, 4, 26), outline=(110, 50, 255), width=max(1, S(2)))
        for gx in np.arange(-G.FX + 0.5, G.FX, 0.5): d.line([P(gx, G.FY), P(gx, -G.FY)], fill=(60, 30, 110, 140))
        for gy in np.arange(-G.FY + 0.5, G.FY, 0.5): d.line([P(-G.FX, gy), P(G.FX, gy)], fill=(60, 30, 110, 140))
        for k, pts in (trails or {}).items():
            if len(pts) >= 2: d.line([P(*p) for p in pts[::2] + [pts[-1]]], fill=NC[k], width=max(1, S(3)))
        for k in ("A", "B"):
            m = meta.get(k) or {}
            if m.get("opp"):
                ox, oy = P(*m["opp"]); q = S(6); d.rectangle([ox - q, oy - q, ox + q, oy + q], outline=NC[k], width=max(1, S(2)))
            if m.get("pose"):
                x, y, yw = m["pose"]; qx, qy = P(x, y); a, b = S(10), S(7)
                d.polygon([(qx + a * math.cos(-yw), qy + a * math.sin(-yw)), (qx + b * math.cos(-yw + 2.4), qy + b * math.sin(-yw + 2.4)),
                           (qx + b * math.cos(-yw - 2.4), qy + b * math.sin(-yw - 2.4))], fill=NC[k], outline=(255, 255, 255))
        d.text((mx0, my0 + MH + S(4)), "square = where it thinks the other is", font=self.f6, fill=(220, 220, 240))
        # referee (SIM TRUTH) bottom strip + sim clock
        ref = meta.get("ref", "")
        d.rectangle([S(400), H - S(54), S(1300), H - S(18)], fill=(8, 0, 20, 185), outline=(0, 240, 160, 180))
        d.text((S(412), H - S(48)), f"SCORER (SIM TRUTH): {ref}"[:80], font=self.f2, fill=(160, 255, 200))
        ck = f"sim t = {t:5.1f} s ({speed_label})"; cw = int(d.textlength(ck, font=self.f2)) + S(24)
        d.rectangle([W - cw - S(14), H - S(54), W - S(14), H - S(18)], fill=(8, 0, 20, 185), outline=(170, 90, 255, 180))
        d.text((W - cw, H - S(48)), ck, font=self.f2, fill=(240, 240, 255))
        g = meta.get("final") or meta.get("banner")
        if g:
            gw = int(d.textlength(g, font=self.fb))
            d.rectangle([W // 2 - gw // 2 - S(30), H // 2 - S(150), W // 2 + gw // 2 + S(30), H // 2 - S(84)], fill=(8, 0, 20, 205), outline=(255, 60, 200, 230), width=max(1, S(3)))
            self._glow(d, (W // 2 - gw // 2, H // 2 - S(144)), g, self.fb, (255, 230, 90))
        return img
