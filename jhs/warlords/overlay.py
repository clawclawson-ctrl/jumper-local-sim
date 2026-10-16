"""Crab Warlords overlay (live + stills): castle style, compact insets so the arena dominates. SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from jhs import PKG
from jhs.overlay import tof_image
from jhs.tanks.overlay import BroadcastCam
from . import game as G

NC = {"A": (240, 70, 50), "B": (80, 130, 255), "C": (70, 220, 90)}
DC = {"ball": (255, 255, 255), "crab": (255, 220, 80)}
STATE = {"opener": "OPENER", "defend": "DEFEND", "attack": "ATTACK", "hold": "HOLD", "out": "OUT", "celebrate": "WINS ROUND",
         "paused": "round over", "over": "over", "return": "RESET"}
GOLD = (240, 200, 110)


class WarlordsCam(BroadcastCam):
    """Tanks-style follow camera: frames all crabs still in + the ball, eased; 'fixed' = whole triangle."""
    OVER = (0.0, 0.25, 10.5)

    def target(self, crabs, shots=None):
        pts = [np.asarray(c[:2], float) for c in (crabs or {}).values() if c is not None]
        if not pts: return np.array(self.OVER)
        P = np.array(pts); lo, hi = P.min(0) - 0.8, P.max(0) + 0.8; c = (lo + hi) / 2
        ext = max(hi[0] - lo[0], (hi[1] - lo[1]) * 1.9)
        dist = float(np.clip(2.0 + 0.95 * ext, 4.5, self.OVER[2]))
        k = (dist - 4.5) / (self.OVER[2] - 4.5)
        cx = float(np.clip(c[0], -3.5, 3.5)) * (1 - k) + self.OVER[0] * k
        cy = (float(np.clip(c[1], -2.0, 4.0)) * (1 - k) + self.OVER[1] * k) - 0.5 * dist / self.OVER[2]
        return np.array([cx, cy, dist])

    def update(self, look_xy=None, t=None, crabs=None, shots=None):
        cam = super().update(look_xy, t, crabs, shots); cam.elevation = -62.0; return cam


class WarlordsOverlay:
    def __init__(self, W=1920, H=1080):
        self.W, self.H = W, H; s = H / 1080.0; self.s = s
        Fn = lambda name, size: ImageFont.truetype(str(PKG / "fonts" / name), max(8, int(size * s)))
        self.fb = Fn("DejaVuSans-Bold.ttf", 34); self.f2 = Fn("DejaVuSans-Bold.ttf", 19); self.f3 = Fn("DejaVuSans.ttf", 14)
        self.f4 = Fn("DejaVuSans-Bold.ttf", 14); self.f5 = Fn("DejaVuSans-Bold.ttf", 16); self.f6 = Fn("DejaVuSans.ttf", 12)
        self.fdet = ImageFont.truetype(str(PKG / "fonts" / "DejaVuSans-Bold.ttf"), 13)

    def _panel(self, d, box, col):
        d.rectangle(box, fill=(28, 20, 14, 185), outline=col + (230,), width=max(1, int(round(2 * self.s))))

    def _cam(self, img, k, m, cam_img, tof, x0, y0):
        s = self.s; S = lambda v: int(round(v * s)); d = ImageDraw.Draw(img, "RGBA"); m = m or {}
        PW, PH, TW, TH = S(200), S(150), S(90), S(70)
        self._panel(d, [x0 - S(5), y0 - S(22), x0 + PW + TW + S(12), y0 + PH + S(26)], NC[k])
        d.text((x0, y0 - S(19)), f"{G.TEAM[k]} cam · bricks {m.get('bricks', '?')}/{m.get('bricks_total', '?')} (scoreboard)", font=self.f4, fill=NC[k])
        if cam_img is not None:
            ci = (cam_img if isinstance(cam_img, Image.Image) else Image.fromarray(cam_img)).convert("RGB").resize((640, 480), Image.BILINEAR)
            cd = ImageDraw.Draw(ci)
            for det in m.get("dets", []):
                a, b, c_, e = det["bbox"]; col = DC.get(det["cls"], (255, 255, 255))
                cd.rectangle([a - 3, b - 3, c_ + 3, e + 3], outline=col, width=4); cd.text((a - 3, max(0, b - 17)), f'{det["cls"]} {det["rng"]:.1f} m', fill=col, font=self.fdet)
            img.paste(ci.resize((PW, PH), Image.BILINEAR), (x0, y0))
        if tof is not None:
            img.paste(Image.fromarray(tof_image(np.asarray(tof, np.float32))).resize((TW, TH), Image.NEAREST), (x0 + PW + S(6), y0))
            d.text((x0 + PW + S(6), y0 + TH + S(2)), "depth", font=self.f6, fill=(230, 220, 200))
        st = m.get("state", ""); d.text((x0 + PW + S(6), y0 + TH + S(18)), STATE.get(st, st)[:12], font=self.f5, fill=GOLD)
        if m.get("opener"): d.text((x0 + PW + S(6), y0 + TH + S(40)), m["opener"][:10] + ("" if m.get("opener_on") else " ✓"), font=self.f6, fill=(220, 210, 180))
        d.text((x0, y0 + PH + S(4)), (m.get("sub") or "")[:36], font=self.f6, fill=(245, 235, 220))

    def draw(self, img, meta, cams, tofs, t, tmax, speed_label="1x", ball=None):
        W, H, s = self.W, self.H, self.s; S = lambda v: int(round(v * s)); d = ImageDraw.Draw(img, "RGBA"); meta = meta or {}
        d.rectangle([S(16), S(12), S(330), S(44)], fill=(150, 20, 20, 215)); d.text((S(26), S(16)), "SIM-ONLY VISION CONCEPT", font=self.f5, fill=(255, 255, 255))
        d.text((S(18), S(50)), "CRAB WARLORDS", font=self.f2, fill=GOLD)
        d.text((S(18), S(74)), "own cam + dToF + pose · brick counts = scoreboard read", font=self.f6, fill=(240, 230, 210))
        sc = meta.get("score", {}); tl = max(0.0, tmax - meta.get("clock", t)); clock = f"{int(tl // 60)}:{int(tl % 60):02d}"
        parts = [(G.TEAM[k] + f" {sc.get(k, 0)}", NC[k]) for k in G.KEYS]; gap = S(36)
        ws = [int(d.textlength(p, font=self.fb)) for p, _ in parts]; tot = sum(ws) + gap * 2; x0 = W // 2 - tot // 2
        d.rectangle([x0 - S(16), S(10), x0 + tot + S(16), S(84)], fill=(28, 20, 14, 195), outline=GOLD + (200,), width=max(1, S(2)))
        x = x0
        for (p, c), w in zip(parts, ws): d.text((x, S(12)), p, font=self.fb, fill=c); x += w + gap
        ag = meta.get("aggression", {})
        rt = int(meta.get('round_t', 0)); cl = f"round {meta.get('round', 1)} · {rt // 60}:{rt % 60:02d} · clock {clock} · first to {meta.get('rounds_to_win', 3)} · aggr " + "/".join(f"{ag.get(k, 60):.0f}" for k in G.KEYS)
        d.text((W // 2 - int(d.textlength(cl, font=self.f3)) // 2, S(58)), cl, font=self.f3, fill=(240, 230, 210))
        pos = {"A": (S(20), H - S(250)), "B": (W - S(20) - S(310), H - S(250)), "C": (W - S(20) - S(310), S(118))}
        for k in G.KEYS: self._cam(img, k, meta.get(k), (cams or {}).get(k), (tofs or {}).get(k), *pos[k])
        # minimap (top-left under the title): castles (SIM TRUTH bricks), crabs, ball
        MW = S(200); mx0 = S(22); my0 = S(110); sc_ = MW / (2 * 4.2)
        def P(x, y): return mx0 + (x + 4.2) * sc_, my0 + (4.8 - y) * sc_
        MH = int(7.3 * sc_)
        self._panel(d, [mx0 - S(5), my0 - S(4), mx0 + MW + S(5), my0 + MH + S(20)], GOLD)
        d.polygon([P(*G.VERT[k]) for k in G.KEYS], fill=(60, 52, 44), outline=(200, 180, 150))
        from .belief import X0, Y0, CELL                       # each crab's OWN ball belief (where the ball may be), in its colour
        for k in G.KEYS:
            Bm = (meta.get(k) or {}).get("belief")
            if not Bm or (meta.get(k) or {}).get("state") == "out": continue
            for i, col in enumerate(Bm):
                for j, v in enumerate(col):
                    if v < 0.08: continue
                    x0_, y0_ = X0 + i * CELL, Y0 + j * CELL
                    d.rectangle([*P(x0_, y0_ + CELL), *P(x0_ + CELL, y0_)], fill=tuple(NC[k][:3]) + (int(50 + 130 * v),))
        for k in G.KEYS:
            al = (meta.get("bricks_alive") or {}).get(k)
            for i, b in enumerate(G.bricks(k)):
                if al is None or al[i]: q = P(*b[:2]); d.rectangle([q[0] - S(2), q[1] - S(2), q[0] + S(2), q[1] + S(2)], fill=NC[k])
            K = P(*G.king(k)); d.rectangle([K[0] - S(4), K[1] - S(4), K[0] + S(4), K[1] + S(4)], fill=GOLD, outline=NC[k])
            m = meta.get(k) or {}
            if m.get("pose") and m.get("state") != "out":
                x, y, yw = m["pose"]; qx, qy = P(x, y); a, b = S(8), S(6)
                d.polygon([(qx + a * math.cos(-yw), qy + a * math.sin(-yw)), (qx + b * math.cos(-yw + 2.4), qy + b * math.sin(-yw + 2.4)),
                           (qx + b * math.cos(-yw - 2.4), qy + b * math.sin(-yw - 2.4))], fill=NC[k], outline=(255, 255, 255))
        if meta.get("ball"):
            q = P(*meta["ball"][:2]); d.ellipse([q[0] - S(3), q[1] - S(3), q[0] + S(3), q[1] + S(3)], fill=(255, 255, 255))
        d.text((mx0, my0 + MH + S(4)), "top view · SIM TRUTH · heat = OWN ball beliefs", font=self.f6, fill=(240, 230, 210))
        ref = meta.get("ref", "")
        d.rectangle([S(360), H - S(52), S(1340), H - S(16)], fill=(28, 20, 14, 195), outline=GOLD + (180,))
        d.text((S(372), H - S(46)), f"SCORER (SIM TRUTH): {ref}"[:88], font=self.f2, fill=(255, 235, 170))
        ck = f"sim t = {t:5.1f} s ({speed_label})"; cw = int(d.textlength(ck, font=self.f3)) + S(20)
        d.rectangle([W // 2 - cw // 2, H - S(80), W // 2 + cw // 2, H - S(56)], fill=(28, 20, 14, 195))
        d.text((W // 2 - cw // 2 + S(10), H - S(78)), ck, font=self.f3, fill=(240, 230, 210))
        g = meta.get("final") or meta.get("banner")
        if g:
            gw = int(d.textlength(g, font=self.fb))
            d.rectangle([W // 2 - gw // 2 - S(30), H // 2 - S(150), W // 2 + gw // 2 + S(30), H // 2 - S(88)], fill=(28, 20, 14, 215), outline=GOLD + (230,), width=max(1, S(3)))
            d.text((W // 2 - gw // 2, H // 2 - S(146)), g, font=self.fb, fill=GOLD)
        return img
