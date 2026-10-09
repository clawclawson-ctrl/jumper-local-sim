"""[jhs/soccer copy of the hide & seek app's vision/vision.py, unchanged except this line.]
Perception for the hide-and-seek gather concept (forked from vision_sort) (SIM-ONLY: the official toolkit gives apps no camera/ToF access).

Input per frame: the onboard RGB camera image (rendered from the robot's own `robot/onboard` camera),
the camera's pose from the robot's OWN kinematics (base pose + joints), and the dToF range grid
(env.scene["tof"], 42x54, metres). No object/bin position from the simulator ever enters here.

Pipeline: HSV colour masks -> connected components -> per-blob ground-plane geometry (bottom pixel ray
meets the floor -> range/bearing; top pixel -> height; column span -> width) -> class by colour + size
+ height -> optional dToF range check -> world-frame xy of the object's centre.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import ndimage

# class: colour gate (hue deg range, s min/max, v min/max), size gate (width m, height m), radius (m),
# needs_below_horizon (toys are lower than the 12 cm camera, so the whole blob is under the horizon)
CLASSES = {
    "ball":     dict(off=0.0, top=0.11, h=(205, 235), s=(0.55, 1.0), v=(0.18, 1.0), w=(0.06, 0.20), ht=(0.06, 0.16), r=0.055, below=True, kind="toy"),
    "duck":     dict(off=0.025, top=0.095, h=(35, 52),   s=(0.57, 1.0), v=(0.40, 1.0), w=(0.04, 0.20), ht=(0.04, 0.14), r=0.045, below=True, kind="toy"),
    "minifb":   dict(off=0.0, top=0.04, h=(0, 360),   s=(0.0, 0.10), v=(0.62, 1.0), w=(0.015, 0.075), ht=(0.015, 0.075), r=0.02, below=True, kind="toy"),
    "football": dict(off=0.0, top=None, h=(0, 360),   s=(0.0, 0.10), v=(0.62, 1.0), w=(0.09, 0.19), ht=(0.095, 0.165), r=0.065, below=False, kind="toy"),
}
RADK = 1.0
MFB_DARK = (0.05, 0.45)   # min fraction of black (pentagon) pixels in a close mini-football blob
TOY_CLASSES = [k for k, v in CLASSES.items() if v["kind"] == "toy"]


def hsv(img):
    x = img.astype(np.float32) * (1 / 255.0)
    mx = x.max(-1); mn = x.min(-1); d = mx - mn + 1e-6
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = np.where(mx > 1e-4, d / (mx + 1e-6), 0)
    return h, s, mx


class Camera:
    def __init__(self, W, H, fovy_deg):
        self.W, self.H = W, H
        self.f = (H / 2) / math.tan(math.radians(fovy_deg) / 2)
        u = (np.arange(W) + 0.5 - W / 2) / self.f
        v = -(np.arange(H) + 0.5 - H / 2) / self.f
        uu, vv = np.meshgrid(u, v)
        self.dcam = np.stack([uu, vv, -np.ones_like(uu)], -1)  # camera frame: x right, y up, -z forward

    def rays(self, R):
        """world-frame (unnormalised) ray directions for every pixel."""
        return self.dcam @ R.T


def detect(img, cam_pos, cam_R, cam: Camera, tof=None, tof_pos=None, tof_R=None, tof_cam: Camera | None = None, dbg=None, selfmask=None):
    """Return a list of detections: dict(cls, xy (world), rng, bearing_world, w, ht, bbox, npx, tof)."""
    H, W = img.shape[:2]
    h, s, v = hsv(img)
    D = cam.rays(cam_R)
    Dn = D / np.linalg.norm(D, axis=-1, keepdims=True)
    elev = np.degrees(np.arcsin(np.clip(Dn[..., 2], -1, 1)))
    below = elev < -0.3
    out = []
    for cls, c in CLASSES.items():
        h0, h1 = c["h"]
        m_full = (h >= h0) & (h <= h1) & (s >= c["s"][0]) & (s <= c["s"][1]) & (v >= c["v"][0]) & (v <= c["v"][1])
        if selfmask is not None: m_full &= ~selfmask        # the robot's own legs/claws (from its body model)
        rows = np.nonzero(m_full.any(1))[0]
        if rows.size == 0: continue
        cols = np.nonzero(m_full.any(0))[0]
        r0, r1 = max(0, rows[0] - 4), min(H, rows[-1] + 5); c0, c1 = max(0, cols[0] - 4), min(W, cols[-1] + 5)
        m = m_full[r0:r1, c0:c1]
        bl = below[r0:r1, c0:c1]; vv_ = v[r0:r1, c0:c1]; ss_ = s[r0:r1, c0:c1]
        if cls in ("minifb", "football"):
            # white things that rise above the horizon (walls, bed) are not a 4 cm ball: drop those components
            # first, then close the black pentagons so one ball is one blob
            lab0, n0 = ndimage.label(m)
            if n0 and cls == "minifb":
                tall = np.unique(lab0[(~bl) & (lab0 > 0)])
                m = m & ~np.isin(lab0, tall)
            elif n0:
                # the football's top is level with the camera (13 cm), so it may graze the horizon; white/cream things
                # that clearly rise above it (stair steps, crate labels high up, walls) are dropped
                hi = elev[r0:r1, c0:c1] > 0.25
                tall = np.unique(lab0[hi & (lab0 > 0)])
                m = m & ~np.isin(lab0, tall)
            dark = (vv_ < 0.22) & (ss_ < 0.25)
            m = ndimage.binary_closing(m | (dark & ndimage.binary_dilation(m, iterations=2)), iterations=1)
        else:
            m = ndimage.binary_opening(m, iterations=1) if c["w"][0] > 0.1 else m
            m = ndimage.binary_closing(m, iterations=2)
        lab, n = ndimage.label(m)
        if n == 0: continue
        objs = ndimage.find_objects(lab)
        for i, sl in enumerate(objs):
            if sl is None: continue
            blob = lab[sl] == (i + 1)
            npx = int(blob.sum())
            if npx < (3 if cls == "minifb" else 6): continue
            ys, xs = np.nonzero(blob); ys = ys + sl[0].start + r0; xs = xs + sl[1].start + c0
            if c["below"] and not below[ys, xs].all():
                if dbg is not None: dbg.append((cls, "horizon", (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), npx))
                continue
            cut_b = ys.max() >= H - 2; cut_s = xs.min() <= 1 or xs.max() >= W - 2
            if not cut_b:
                # bottom pixel ray -> floor (the floor point just in front of the object)
                vb = ys.max(); cols = xs[ys >= vb - 1]; ub = int(np.median(cols))
                d = cam_R @ np.array([(ub + 0.5 - W / 2) / cam.f, -(vb + 1.0 - H / 2) / cam.f, -1.0])   # lower edge of the bottom pixel
                if d[2] >= -1e-4: continue
                t = -cam_pos[2] / d[2]; g = cam_pos + t * d
                rngh = float(np.hypot(g[0] - cam_pos[0], g[1] - cam_pos[1]))
                vt = ys.min(); ut = int(np.median(xs[ys <= vt + 1]))
                dt_ = Dn[vt, ut]; ht = cam_pos[2] + rngh * dt_[2] / max(1e-6, math.hypot(dt_[0], dt_[1]))
            else:
                # cut by the image bottom (very close): range from the top pixel and the class's known top height
                if c["top"] is None:
                    if dbg is not None: dbg.append((cls, "bottom", (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), npx))
                    continue
                vt = ys.min(); ut = int(np.median(xs[ys <= vt + 1])); dt_ = D[vt, ut]
                if dt_[2] >= -1e-4: continue
                t = (c["top"] - cam_pos[2]) / dt_[2]; g = cam_pos + t * dt_
                rngh = float(np.hypot(g[0] - cam_pos[0], g[1] - cam_pos[1])); ht = c["top"]
                ub = ut
            if rngh > 4.5: continue
            # width from the azimuth span of the blob's columns (exact for a wide-angle pinhole)
            rowc = int(np.clip(np.median(ys), 0, H - 1))
            dd2 = D[rowc, [int(xs.min()), int(xs.max())]]; azs = np.arctan2(dd2[:, 1], dd2[:, 0])
            daz = abs((azs[1] - azs[0] + math.pi) % (2 * math.pi) - math.pi) + 1.0 / cam.f
            wv = daz * rngh
            wok = (c["w"][0] <= wv or cut_s or cut_b) and wv <= c["w"][1] * (1.6 if (cut_s or cut_b) else 1.0)
            if not (wok and c["ht"][0] <= ht <= c["ht"][1]):
                if dbg is not None: dbg.append((cls, "size", (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), npx, round(float(wv), 3), round(float(ht), 3), round(rngh, 2)))
                continue
            dirh = np.array([g[0] - cam_pos[0], g[1] - cam_pos[1]]) / max(rngh, 1e-6)
            if cut_b:
                if npx < 40 or wv < 0.6 * c["w"][0]:      # slivers at the bottom edge are the robot's own claws
                    continue
                ctr = g[:2] + dirh * (c["r"] * 0.3)
            else:
                ctr = g[:2] + dirh * c["off"]
            # dToF check: zones on the blob's centre direction (if in the ToF field of view)
            tr = None
            if tof is not None and tof_R is not None:
                uc, vc = int(xs.mean()), int(ys.mean())
                dw = Dn[vc, ub if cls in ("minifb", "football") else uc]
                dc = tof_R.T @ dw                                        # into the ToF camera frame
                if dc[2] < -1e-3:
                    tu = dc[0] / -dc[2] * tof_cam.f + tof_cam.W / 2; tv = -dc[1] / -dc[2] * tof_cam.f + tof_cam.H / 2
                    if 0 <= tu < tof_cam.W and 0 <= tv < tof_cam.H:
                        a, b = int(tv), int(tu)
                        patch = tof[max(0, a - 1):a + 2, max(0, b - 1):b + 2]; pv = patch[patch > 0]
                        if pv.size: tr = float(np.min(pv))
            dark = None
            if cls in ("minifb", "football"):
                bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                vb_ = v[by0:by1, bx0:bx1]; dark = float((vb_ < 0.2).mean())
                if ((cls == "minifb" and rngh < 1.3 and npx >= 25) or (cls == "football" and npx >= 25)) and not (MFB_DARK[0] <= dark <= MFB_DARK[1]):
                    if dbg is not None: dbg.append((cls, "nodark", (bx0, by0, bx1, by1), npx, round(dark, 3), round(rngh, 2)))
                    continue
            out.append(dict(cls=cls, xy=ctr, cut=bool(cut_b), dark=dark, rng=rngh, w=round(float(wv), 3), ht=round(float(ht), 3),
                            bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), npx=npx, tof=tr))
    return out


def tof_obstacles(tof, tof_pos, tof_R, tof_cam: Camera, zmin=0.045, zmax=0.4, rmax=0.6):
    """3D points from the dToF that stick out of the floor (obstacles), world frame, Nx3."""
    D = tof_cam.rays(tof_R); Dn = D / np.linalg.norm(D, axis=-1, keepdims=True)
    ok = tof > 0
    P = tof_pos + Dn * tof[..., None]
    sel = ok & (P[..., 2] > zmin) & (P[..., 2] < zmax) & (tof < rmax)
    return P[sel]


# ---- the gathering rug: colour votes on the floor plane --------------------------------------------------------
RUG_C = dict(h=(22, 48), s=(0.17, 0.50), v=(0.16, 0.75))
FLOOR_C = dict(h=(60, 200), s=(0.0, 0.13), v=(0.06, 0.45))


def rug_votes(img, cam_pos, cam_R, cam: Camera, selfmask=None, step=3, rmax=3.2):
    """World-xy floor points of rug-coloured and floor-coloured pixels (below the horizon). Rug-coloured blobs that
    reach up to the horizon are upright things (tan crates), not the flat rug, and are dropped."""
    H, W = img.shape[:2]
    sub = img[::step, ::step]
    h, s, v = hsv(sub)
    D = cam.rays(cam_R)[::step, ::step]
    Dn = D / np.linalg.norm(D, axis=-1, keepdims=True)
    elev = np.degrees(np.arcsin(np.clip(Dn[..., 2], -1, 1)))
    below = elev < -1.0
    sm = selfmask[::step, ::step] if selfmask is not None else np.zeros(h.shape, bool)
    def gate(c):
        return (h >= c["h"][0]) & (h <= c["h"][1]) & (s >= c["s"][0]) & (s <= c["s"][1]) & (v >= c["v"][0]) & (v <= c["v"][1]) & ~sm
    rug = gate(RUG_C)
    lab, n = ndimage.label(rug)
    if n:
        tall = np.unique(lab[(elev > -0.6) & (lab > 0)])
        rug &= ~np.isin(lab, tall)
    rug &= below
    flo = gate(FLOOR_C) & below
    out = []
    for m in (rug, flo):
        d = D[m]
        if d.shape[0] == 0: out.append(np.zeros((0, 2))); continue
        t = -(cam_pos[2] - 0.006) / d[:, 2]
        P = cam_pos[None, :2] + t[:, None] * d[:, :2]
        r = np.hypot(P[:, 0] - cam_pos[0], P[:, 1] - cam_pos[1])
        out.append(P[(r < rmax) & (r > 0.12)])
    return out[0], out[1]
