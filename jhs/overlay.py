"""The demo layout, drawn over a frame of the 3D view: camera inset with the vision's boxes, dToF, the crab's own
map, the status line, a scorer line marked SIM TRUTH, the sim clock and the SIM-ONLY VISION CONCEPT label.
Used both by the live window and by the MP4 renderer, so the saved video looks like the live view."""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from . import PKG

COL = {"ball": (60, 140, 255), "duck": (255, 205, 40), "duck2": (255, 205, 40), "minifb": (240, 240, 240), "football": (200, 200, 255),
       "basket": (196, 154, 92), "crate": (72, 128, 214)}
NAME = {"ball": "blue ball", "duck": "yellow duck", "duck2": "2nd yellow duck", "minifb": "mini football", "football": "football",
        "basket": "basket", "crate": "blue crate"}
ROOM = (-2.65, -2.65, 2.65, 2.65)
# Tidy-up floor plan, and the same "in the bin" test tidy_vision.py uses to score (BINR + toy radius + 0.20).
TIDY_ROOM = (-1.95, -2.75, 4.35, 1.80)
TIDY_DEST = {"ball": "basket", "minifb": "basket", "duck": "crate"}
TIDY_REACH = {"ball": 0.18 + 0.055 + 0.20, "minifb": 0.18 + 0.02 + 0.20, "duck": 0.16 + 0.045 + 0.20}


def tidy_in_bin(xy, cls, bins) -> bool:
    """True when toy xy is inside the correct bin. bins: {name: (x, y)}. Display scoring only."""
    dest = TIDY_DEST.get(cls)
    if not dest or dest not in bins: return False
    bx, by = bins[dest][0], bins[dest][1]
    return math.hypot(float(xy[0]) - float(bx), float(xy[1]) - float(by)) < TIDY_REACH[cls]


def turbo(x):
    """Google's Turbo colormap (polynomial approximation), x in [0,1] -> uint8 RGB. No matplotlib needed."""
    x = np.clip(x, 0.0, 1.0)[..., None]
    kr = (0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943)
    kg = (0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604)
    kb = (0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973)
    def poly(k): return k[0] + x * (k[1] + x * (k[2] + x * (k[3] + x * (k[4] + x * k[5]))))
    return (np.clip(np.concatenate([poly(kr), poly(kg), poly(kb)], -1), 0, 1) * 255).astype(np.uint8)


def tof_image(r):
    lo, hi = math.log(0.05), math.log(8.8)
    f = (np.log(np.clip(r, 0.05, 8.8)) - lo) / (hi - lo)
    img = turbo(1.0 - 0.9 * f); img[r <= 0] = 0
    return img


def status_text(m, ntoys=None, task="hide_seek"):
    st = m.get("state"); foc = m.get("focus")
    if task == "tidy":
        if st == "finish" and ntoys is not None and len(m.get("gathered", [])) < ntoys: return "time's up"
        if st == "push" and foc: return f"pushing the {NAME.get(foc, foc)} to its bin"
        if st == "approach" and foc: return f"going to the {NAME.get(foc, foc)}"
        if st == "approach": return "going to look closer"
        if st == "scan": return "looking around"
        if st == "explore": return "searching the room"
        if st == "start": return "getting ready"
        if st == "finish": return "all sorted!"
        return str(st)
    if st == "finish" and ntoys is not None and len(m.get("gathered", [])) < ntoys: return "time's up -- bow and dance"
    if st == "push" and foc: return f"pushing the {NAME.get(foc, foc)} onto the rug"
    if st == "approach" and foc: return f"going round to the {NAME.get(foc, foc)}"
    if st == "approach": return "going to look closer"
    if st == "scan": return "looking around"
    if st == "explore": return "searching behind things"
    if st == "finish": return "all gathered!"
    return str(st)


class Overlay:
    def __init__(self, W=1920, H=1080):
        self.W, self.H = W, H; s = H / 1080.0; self.s = s
        F = lambda name, size: ImageFont.truetype(str(PKG / "fonts" / name), max(8, int(size * s)))
        self.f1 = F("DejaVuSans-Bold.ttf", 34); self.f2 = F("DejaVuSans-Bold.ttf", 24); self.f3 = F("DejaVuSans.ttf", 20)
        self.fdet = ImageFont.truetype(str(PKG / "fonts" / "DejaVuSans-Bold.ttf"), 18)
        self.f4 = F("DejaVuSans-Bold.ttf", 18); self.f5 = F("DejaVuSans-Bold.ttf", 30); self.f6 = F("DejaVuSans.ttf", 17)

    def draw(self, img, meta, cam_img, tof, mp, t, n_truth, ntoys, speed_label="1x speed", banner=None, extra=None, props=None, task="hide_seek", room=None):
        """img: PIL RGB WxH of the 3D view. meta: the brain's per-frame record. cam_img: PIL/ndarray onboard frame (640x480).
        tof: 42x54 ranges. mp: coverage map (NX x NY, 0/1/2) or None. n_truth: toys on the rug by SIM TRUTH (display only).
        props: [(id, [4 world-xy corners])] where the obstacles are NOW (sim truth, display only), drawn as outlines on the map."""
        W, H, s = self.W, self.H, self.s; S = lambda v: int(round(v * s))
        d = ImageDraw.Draw(img, "RGBA")
        PW, PH, TW, TH, MS = S(600), S(450), S(54 * 7), S(42 * 7), S(300)
        px, py = W - PW - S(24), S(24)
        d.rectangle([px - S(10), py - S(10), W - S(14), py + PH + S(40) + TH + S(40)], fill=(0, 0, 0, 150))
        if cam_img is not None:
            ci = (cam_img if isinstance(cam_img, Image.Image) else Image.fromarray(cam_img)).convert("RGB").copy(); cd = ImageDraw.Draw(ci)
            for det in meta.get("dets", []):
                x0, y0, x1, y1 = det["bbox"]; c = COL.get(det["cls"], (255, 255, 255))
                cd.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=c, width=2)
                cd.text((x0 - 3, max(0, y0 - 20)), f'{NAME.get(det["cls"], det["cls"])} {det["rng"]:.1f} m', fill=c, font=self.fdet)
            img.paste(ci.resize((PW, PH), Image.BILINEAR), (px, py))
        d.text((px + S(8), py + PH + S(6)), "Jumper's onboard camera (boxes = what its vision found)", font=self.f3, fill=(255, 255, 255))
        if tof is not None:
            img.paste(Image.fromarray(tof_image(np.asarray(tof, np.float32))).resize((TW, TH), Image.NEAREST), (px + (PW - TW) // 2, py + PH + S(36)))
        d.text((px + S(40), py + PH + S(40) + TH + S(6)), "dToF depth, 54x42 zones (red = near, blue = far)", font=self.f3, fill=(255, 255, 255))
        mx0, my0 = S(24), H - S(110) - MS - S(40)
        d.rectangle([mx0 - S(10), my0 - S(40), mx0 + MS + S(10), my0 + MS + (S(50) if props else S(32))], fill=(0, 0, 0, 160))
        d.text((mx0, my0 - S(34)), "room map: crab, seen toys, bins" if task == "tidy" else "Jumper's own map: searched / not yet", font=self.f6, fill=(255, 255, 255))
        if mp is not None:
            colm = np.array([[38, 38, 42], [70, 150, 95], [215, 90, 70]], np.uint8)
            img.paste(Image.fromarray(colm[np.asarray(mp).T[::-1]]).resize((MS, MS), Image.NEAREST), (mx0, my0))
        bounds = room if room is not None else ROOM
        def to_px(x, y): return mx0 + (x - bounds[0]) / (bounds[2] - bounds[0]) * MS, my0 + (bounds[3] - y) / (bounds[3] - bounds[1]) * MS
        for _k, P in (props or []):
            d.polygon([to_px(x, y) for x, y in P], outline=(235, 235, 235), width=max(1, S(1.5)))
        if meta.get("rug"):
            cx, cy, ux, uy, hu, hv = meta["rug"]; u = np.array([ux, uy]); v = np.array([-uy, ux]); c0 = np.array([cx, cy])
            d.polygon([to_px(*(c0 + su * hu * u + sv * hv * v)) for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))], outline=(225, 185, 120), width=max(1, S(2)))
        if meta.get("goal"):
            gx, gy = to_px(*meta["goal"]); k = S(6)
            d.line([gx - k, gy - k, gx + k, gy + k], fill=(255, 255, 255), width=2); d.line([gx - k, gy + k, gx + k, gy - k], fill=(255, 255, 255), width=2)
        for cls, x, y, _srt in meta.get("tracks", []):
            qx, qy = to_px(x, y); k = S(5); d.ellipse([qx - k, qy - k, qx + k, qy + k], fill=COL.get(cls, (255, 255, 255)), outline=(0, 0, 0))
        if task == "tidy":
            for name, x, y in meta.get("bins") or []:
                qx, qy = to_px(x, y); k = S(9)
                d.rectangle([qx - k, qy - k, qx + k, qy + k], outline=COL.get(name, (255, 255, 255)), width=max(2, S(2)))
        if meta.get("pose"):
            bx, by, byaw = meta["pose"]; qx, qy = to_px(bx, by); a, b = S(17), S(11)
            d.polygon([(qx + a * math.cos(-byaw), qy + a * math.sin(-byaw)), (qx + b * math.cos(-byaw + 2.4), qy + b * math.sin(-byaw + 2.4)),
                       (qx + b * math.cos(-byaw - 2.4), qy + b * math.sin(-byaw - 2.4))], fill=(255, 40, 40), outline=(255, 255, 255))
        d.text((mx0, my0 + MS + S(6)), "dots: what the crab has seen; squares: bins now (SIM TRUTH)" if task == "tidy" else "green searched, red obstacle, tan rug, dots toys", font=self.f6, fill=(230, 230, 230))
        if props: d.text((mx0, my0 + MS + S(27)), "white outlines: props now (SIM TRUTH)", font=self.f6, fill=(230, 230, 230))
        ng = len(meta.get("gathered", []))
        tidy = task == "tidy"
        d.rectangle([S(20), S(20), S(840), S(140)], fill=(0, 0, 0, 150))
        d.text((S(36), S(30)), "Jumper Tidy-Up" if tidy else "Jumper Hide & Seek", font=self.f1, fill=(255, 255, 255))
        d.text((S(36), S(78)), f"{status_text(meta, ntoys, task)}   (crab's own count: {ng}/{ntoys} {'sorted' if tidy else 'on the rug'})", font=self.f2, fill=(255, 230, 120))
        d.rectangle([S(20), S(148), S(840), S(188)], fill=(0, 0, 0, 150))
        d.text((S(36), S(155)), f"scorer (SIM TRUTH, not used by the crab): {n_truth}/{ntoys} {'in the right bin' if tidy else 'on the rug'}", font=self.f3, fill=(200, 255, 200))
        d.rectangle([S(20), H - S(100), S(1270), H - S(20)], fill=(0, 0, 0, 165))
        d.text((S(34), H - S(96)), "SIM-ONLY VISION CONCEPT", font=self.f5, fill=(255, 120, 120))
        foot = (f"{ntoys} toys to sort by camera + dToF into the basket or the crate. Simulation only, not real-robot capable." if tidy
                else f"{ntoys} toys to find by camera + dToF and push onto the rug. Simulation only, not real-robot capable.")
        d.text((S(34), H - S(56)), foot, font=self.f3, fill=(255, 255, 255))
        clock = f"sim time t = {t:5.1f} s   ({speed_label})"; cw = int(d.textlength(clock, font=self.f2)) + S(28)
        d.rectangle([W - cw - S(14), H - S(64), W - S(14), H - S(20)], fill=(0, 0, 0, 165))
        d.text((W - cw, H - S(56)), clock, font=self.f2, fill=(240, 240, 240))
        if extra:
            d.rectangle([S(860), S(20), S(860) + S(14) * len(extra) + S(30), S(64)], fill=(0, 0, 0, 150)); d.text((S(876), S(28)), extra, font=self.f3, fill=(255, 255, 255))
        if banner:
            d.rectangle([W // 2 - S(380), S(210), W // 2 + S(280), S(270)], fill=(150, 20, 20, 210)); d.text((W // 2 - S(362), S(222)), banner, font=self.f2, fill=(255, 255, 255))
        return img


class FollowCam:
    """The demo's follow camera: from the arena's centre side, looking outward past the crab."""
    def __init__(self):
        import mujoco
        self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE; self.look = None; self.az = None

    def update(self, qpos, rate=1.0):
        rp = np.array(qpos[:3]); q = qpos[3:7]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        tgt = np.array([rp[0], rp[1], 0.08])
        if self.look is None:
            self.look = tgt.copy(); self.az = math.degrees(math.atan2(-rp[1], -rp[0])) + 180 if np.hypot(*rp[:2]) > 0.3 else math.degrees(yaw) + 150
        self.look += (tgt - self.look) * min(1.0, 0.05 * rate)
        want = math.degrees(math.atan2(rp[1], rp[0])) if np.hypot(*rp[:2]) > 0.4 else math.degrees(yaw)
        self.az += math.degrees((math.radians(want - self.az) + math.pi) % (2 * math.pi) - math.pi) * min(1.0, 0.01 * rate)
        self.cam.lookat[:] = self.look; self.cam.distance = 2.6; self.cam.azimuth = self.az; self.cam.elevation = -42
        return self.cam
