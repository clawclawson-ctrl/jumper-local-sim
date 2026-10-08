"""Demo overlay for flybrain runs (live view and the 1x MP4). SIM-ONLY VISION CONCEPT: fly-inspired rules, not a connectome.
Camera inset with optic-flow arrows and the fixated/salient blob box, dToF inset with the looming highlight, the big
behaviour state, behaviour counts, the crab's path on a minimap (props' current outlines = sim truth), sim clock."""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from . import PKG
from .overlay import tof_image

ROOM = (-2.65, -2.65, 2.65, 2.65)
STATE_COL = {"ESCAPE": (255, 80, 80), "FREEZE": (120, 200, 255), "AVOID": (255, 170, 60), "FIXATE": (255, 230, 90),
             "APPROACH": (150, 240, 120), "OPTOMOTOR": (200, 140, 255), "SACCADE": (90, 220, 230), "WALK": (235, 235, 235)}
STATE_TXT = {"ESCAPE": "looming / bump: back off and turn away", "FREEZE": "freeze (LB hold) before escaping",
             "AVOID": "something close on one side: turn away", "FIXATE": "salient bar / blob: turn to centre it",
             "APPROACH": "walking toward the fixated blob", "OPTOMOTOR": "wide-field motion: turn with it",
             "SACCADE": "quick exploratory turn", "WALK": "walking straight"}
COUNT_KEYS = [("escapes", "escapes"), ("bumps", "bumps"), ("avoids", "avoids"), ("fixations", "fixations"),
              ("approaches", "approaches"), ("optomotor", "optomotor"), ("saccades", "saccades")]


def display_state(beh, sub, cmd):
    if cmd and cmd.get("lb"): return "FREEZE"
    if beh == "FIXATE" and sub == "approach": return "APPROACH"
    return beh or "WALK"


class FlyOverlay:
    def __init__(self, W=1920, H=1080):
        self.W, self.H = W, H; s = H / 1080.0; self.s = s
        F = lambda name, size: ImageFont.truetype(str(PKG / "fonts" / name), max(8, int(size * s)))
        self.fbig = F("DejaVuSans-Bold.ttf", 66); self.f1 = F("DejaVuSans-Bold.ttf", 30); self.f2 = F("DejaVuSans-Bold.ttf", 24)
        self.f3 = F("DejaVuSans.ttf", 20); self.f6 = F("DejaVuSans.ttf", 17); self.f5 = F("DejaVuSans-Bold.ttf", 28)
        self.fdet = ImageFont.truetype(str(PKG / "fonts" / "DejaVuSans-Bold.ttf"), 18)

    def draw(self, img, meta, cam_img, tof, t, speed_label="1x speed", props=None, path=None, banner=None, floor=None):
        """img: PIL RGB 3D view. meta: the fly loop's per-tick record (state, sub, cmd, flow, hemifield, residual, near,
        loom, blob, blob_range, pose, counts). path: [(x, y)] poses so far. props: [(id, corners)] sim truth outlines."""
        W, H, s = self.W, self.H, self.s; S = lambda v: int(round(v * s))
        d = ImageDraw.Draw(img, "RGBA")
        meta = meta or {}
        st = meta.get("state", "WALK"); col = STATE_COL.get(st, (255, 255, 255))
        # --- camera inset with flow arrows + salient blob ---------------------------------------------------------
        PW, PH, TW, TH = S(600), S(450), S(54 * 7), S(42 * 7)
        px, py = W - PW - S(24), S(24)
        d.rectangle([px - S(10), py - S(10), W - S(14), py + PH + S(40) + TH + S(70)], fill=(0, 0, 0, 150))
        if cam_img is not None:
            ci = (cam_img if isinstance(cam_img, Image.Image) else Image.fromarray(cam_img)).convert("RGB").copy()
            cw, chh = ci.size; cd = ImageDraw.Draw(ci)
            b = meta.get("blob")
            if b and b.get("bbox"):
                x0, y0, x1, y1 = b["bbox"]; bc = (255, 230, 90) if st in ("FIXATE", "APPROACH") else (200, 200, 200)
                cd.rectangle([x0, y0, x1, y1], outline=bc, width=3)
                rng = meta.get("blob_range")
                cd.text((x0 + 3, max(0, y0 - 20)), f'salient {b.get("kind", "")}' + (f" {rng:.1f} m" if rng else ""), fill=bc, font=self.fdet)
            hf = meta.get("hemifield") or [0, 0]
            for i, v in enumerate(hf):           # flow on each half, decimated px/tick -> full px, exaggerated x6 for visibility
                if not v: continue
                cx0 = cw * (0.25 + 0.5 * i); cy0 = chh * 0.88; L = float(v) * 4 * 6
                cd.line([cx0, cy0, cx0 + L, cy0], fill=(120, 255, 255), width=4)
                hx = cx0 + L; sgn = 1 if L > 0 else -1
                cd.polygon([(hx + sgn * 12, cy0), (hx, cy0 - 8), (hx, cy0 + 8)], fill=(120, 255, 255))
            cd.text((8, chh - 26), f"optic flow L {hf[0]:+d} / R {hf[1]:+d} px   residual {meta.get('residual', 0):+.1f}",
                    fill=(120, 255, 255), font=self.fdet)
            img.paste(ci.resize((PW, PH), Image.BILINEAR), (px, py))
        d.text((px + S(8), py + PH + S(6)), "Jumper's onboard camera: flow arrows, salient blob box", font=self.f3, fill=(255, 255, 255))
        ty = py + PH + S(36); tx = px + (PW - TW) // 2
        if tof is not None:
            ti = tof_image(np.asarray(tof, np.float32))
            if floor is not None and np.shape(floor) == np.shape(tof):      # floor returns: ignored by the rules, drawn grey
                fm = np.asarray(floor, bool); ti[fm] = (ti[fm].astype(np.float32).mean(-1, keepdims=True) * 0.35 + 40).astype(ti.dtype)
            img.paste(Image.fromarray(ti).resize((TW, TH), Image.NEAREST), (tx, ty))
            lo = meta.get("loom")
            if lo:
                d.rectangle([tx + int(TW * 0.28), ty, tx + int(TW * 0.72), ty + TH], outline=(255, 40, 40), width=max(2, S(5)))
                d.text((tx + S(8), ty + S(6)), f"LOOMING  ttc {lo['ttc']:.2f} s", font=self.f2, fill=(255, 60, 60))
        nr = meta.get("near") or [None, None, None]
        d.text((px + S(20), ty + TH + S(6)), "dToF: red = near, grey = floor (ignored), box = looming", font=self.f3, fill=(255, 255, 255))
        if nr[0] is not None:
            d.text((px + S(20), ty + TH + S(32)), "nearest above floor  " + "   ".join(f"{n} {v:.2f} m" if v < 2.9 else f"{n} clear" for n, v in zip(("left", "centre", "right"), nr)), font=self.f3, fill=(230, 230, 230))
        # --- big state ---------------------------------------------------------------------------------------------
        d.rectangle([S(20), S(20), S(900), S(196)], fill=(0, 0, 0, 160))
        d.text((S(36), S(26)), "Jumper flybrain", font=self.f1, fill=(255, 255, 255))
        d.text((S(36), S(64)), st, font=self.fbig, fill=col)
        sw = int(d.textlength(st, font=self.fbig))
        sub = meta.get("sub") or ""
        d.text((S(36) + sw + S(20), S(92)), f"({sub})" if sub else "", font=self.f2, fill=col)
        cmd = meta.get("cmd") or {}
        d.text((S(36), S(150)), STATE_TXT.get(st, "") + f"   pad: fwd {-cmd.get('ly', 0):+.2f} turn {cmd.get('rx', 0):+.2f}",
               font=self.f3, fill=(230, 230, 230))
        # --- behaviour counts ----------------------------------------------------------------------------------------
        cnt = meta.get("counts") or {}
        bx0, by0 = S(20), S(206)
        d.rectangle([bx0, by0, bx0 + S(330), by0 + S(34) + S(24) * len(COUNT_KEYS)], fill=(0, 0, 0, 150))
        d.text((bx0 + S(14), by0 + S(6)), "behaviour counts", font=self.f2, fill=(255, 255, 255))
        for i, (k, lab) in enumerate(COUNT_KEYS):
            yy = by0 + S(36) + S(24) * i; v = str(cnt.get(k, 0))
            d.text((bx0 + S(14), yy), lab, font=self.f3, fill=(225, 225, 225))
            d.text((bx0 + S(300) - int(d.textlength(v, font=self.f3)), yy), v, font=self.f3, fill=(225, 225, 225))
        # --- minimap: path -------------------------------------------------------------------------------------------
        MS = S(300); mx0, my0 = S(24), H - S(110) - MS - S(40)
        d.rectangle([mx0 - S(10), my0 - S(40), mx0 + MS + S(10), my0 + MS + S(50)], fill=(0, 0, 0, 160))
        d.text((mx0, my0 - S(34)), "crab's path (odometry)", font=self.f6, fill=(255, 255, 255))
        d.rectangle([mx0, my0, mx0 + MS, my0 + MS], fill=(38, 38, 42), outline=(200, 90, 70), width=max(1, S(2)))
        def to_px(x, y): return mx0 + (x - ROOM[0]) / (ROOM[2] - ROOM[0]) * MS, my0 + (ROOM[3] - y) / (ROOM[3] - ROOM[1]) * MS
        d.polygon([to_px(x, y) for x, y in ((0.87, 0.64), (-0.87, 0.64), (-0.87, -0.64), (0.87, -0.64))], outline=(225, 185, 120), width=max(1, S(1.5)))
        for _k, P in (props or []):
            d.polygon([to_px(x, y) for x, y in P], outline=(235, 235, 235), width=max(1, S(1.5)))
        if path and len(path) > 1:
            d.line([to_px(x, y) for x, y in path], fill=(90, 200, 255), width=max(1, S(2)))
        pose = meta.get("pose")
        if pose:
            bx, by, byaw = pose; qx, qy = to_px(bx, by); a, b = S(17), S(11)
            d.polygon([(qx + a * math.cos(-byaw), qy + a * math.sin(-byaw)), (qx + b * math.cos(-byaw + 2.4), qy + b * math.sin(-byaw + 2.4)),
                       (qx + b * math.cos(-byaw - 2.4), qy + b * math.sin(-byaw - 2.4))], fill=(255, 40, 40), outline=(255, 255, 255))
        d.text((mx0, my0 + MS + S(6)), "blue path, tan rug, red walls", font=self.f6, fill=(230, 230, 230))
        d.text((mx0, my0 + MS + S(27)), "white outlines: props now (SIM TRUTH)", font=self.f6, fill=(230, 230, 230))
        # --- banner + clock -------------------------------------------------------------------------------------------
        d.rectangle([S(20), H - S(100), S(1270), H - S(20)], fill=(0, 0, 0, 165))
        d.text((S(34), H - S(96)), "SIM-ONLY VISION CONCEPT: fly-inspired rules, not a connectome", font=self.f5, fill=(255, 120, 120))
        d.text((S(34), H - S(56)), "Official Jumper walking policy, steered by hand-written rules from the sim camera + dToF. Not real-robot capable.",
               font=self.f6, fill=(255, 255, 255))
        clock = f"sim time t = {t:5.1f} s   ({speed_label})"; cw_ = int(d.textlength(clock, font=self.f2)) + S(28)
        d.rectangle([W - cw_ - S(14), H - S(64), W - S(14), H - S(20)], fill=(0, 0, 0, 165))
        d.text((W - cw_, H - S(56)), clock, font=self.f2, fill=(240, 240, 240))
        if banner:
            d.rectangle([W // 2 - S(380), S(210), W // 2 + S(280), S(270)], fill=(150, 20, 20, 210)); d.text((W // 2 - S(362), S(222)), banner, font=self.f2, fill=(255, 255, 255))
        return img
