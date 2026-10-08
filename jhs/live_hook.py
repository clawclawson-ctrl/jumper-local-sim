"""Loaded by the vision brain (VS_LIVE_HOOK) inside the sim process: writes the live demo-layout frame for the
browser window, caps the run at real time, and honours the Stop button. DISPLAY ONLY: nothing here reaches the
brain. The scorer count uses sim truth and is labelled so on screen. SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jhs.overlay import Overlay, FollowCam  # noqa: E402


def _atomic_write(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_bytes(data); os.replace(tmp, path)


class Hook:
    def __init__(self, sim, toys, rug, room, outd, dt, kind="hide_seek"):
        import mujoco
        self.mj = mujoco; self.sim = sim; self.toys = dict(toys); self.rug = rug; self.dt = dt
        self.dir = Path(os.environ.get("JHS_LIVE_DIR", str(Path(outd) / "live"))); self.dir.mkdir(parents=True, exist_ok=True)
        self.stopf = Path(os.environ.get("JHS_STOP", str(self.dir / "STOP")))
        self.every = float(os.environ.get("JHS_LIVE_EVERY", 0.25))          # sim seconds between live frames
        self.W, self.H = int(os.environ.get("JHS_LIVE_W", 1280)), int(os.environ.get("JHS_LIVE_H", 720))
        self.speed = float(os.environ.get("JHS_SPEED", 1.0))                # cap: at most this x real time (0 = uncapped)
        self.enabled = os.environ.get("JHS_LIVE", "1") == "1"
        model, live = sim.env_mjdata(0)
        self.rmodel = getattr(sim, "render_model", None) or model
        self.rmodel.vis.global_.offwidth = max(self.W, self.rmodel.vis.global_.offwidth)
        self.rmodel.vis.global_.offheight = max(self.H, self.rmodel.vis.global_.offheight)
        self.rd = mujoco.MjData(self.rmodel)
        self.rend = mujoco.Renderer(self.rmodel, self.H, self.W) if self.enabled else None
        self.kind = kind; self.path = []
        if kind == "fly":
            from jhs.fly_overlay import FlyOverlay
            self.ov = FlyOverlay(self.W, self.H)
        else:
            self.ov = Overlay(self.W, self.H)
        self.cam = FollowCam()
        self.last = -1e9; self.t0 = None; self.sim0 = 0.0
        self.meta = {}; self.img = None; self.tof = None; self.mp = None
        self.ntoys = len(self.toys); self.nframes = 0; self.last_status = 0.0
        from jhs.pushable import prop_geoms
        self.pg = prop_geoms(model)          # obstacles' boxes: outlines of where they are now (sim truth, display only)

    def vision(self, t, img, rngs, meta, mp):
        self.img = img; self.tof = rngs; self.meta = meta
        if meta.get("pose"): self.path.append((meta["pose"][0], meta["pose"][1]))
        if mp is not None: self.mp = mp

    def _n_truth(self, live):
        cx, cy, hx, hy = self.rug
        return sum(abs(live.xpos[b][0] - cx) < hx and abs(live.xpos[b][1] - cy) < hy for b in self.toys.values())

    def step(self, t):
        now = time.time()
        if self.t0 is None: self.t0 = now; self.sim0 = t
        if self.speed > 0:                                    # never faster than JHS_SPEED x real time
            ahead = (t - self.sim0) / self.speed - (now - self.t0)
            if ahead > 0.002: time.sleep(min(ahead, 0.5))
        if self.enabled and t - self.last >= self.every and self.meta:
            self.last = t
            from PIL import Image
            model, live = self.sim.env_mjdata(0)
            self.rd.qpos[:] = live.qpos[: self.rmodel.nq]; self.mj.mj_forward(self.rmodel, self.rd)
            self.rend.update_scene(self.rd, self.cam.update(self.rd.qpos, rate=self.every / 0.04))
            frame = Image.fromarray(self.rend.render())
            wall = time.time() - self.t0; rate = (t - self.sim0) / wall if wall > 1 else 0.0
            from jhs.pushable import footprints
            lab = f"live, {rate:.2f}x" if rate else "live"
            if self.kind == "fly":
                self.ov.draw(frame, self.meta, self.img, self.tof, t, speed_label=lab, props=footprints(model, live, self.pg), path=self.path,
                             floor=self.mp)
            else:
                self.ov.draw(frame, self.meta, self.img, self.tof, self.mp, t, self._n_truth(live), self.ntoys,
                             speed_label=lab, props=footprints(model, live, self.pg))
            import io
            buf = io.BytesIO(); frame.save(buf, "JPEG", quality=82); _atomic_write(self.dir / "live.jpg", buf.getvalue())
            self.nframes += 1
        if now - self.last_status > 1.0:
            self.last_status = now
            model, live = self.sim.env_mjdata(0)
            wall = now - self.t0
            st = dict(t=round(t, 2), wall=round(wall, 1), rate=round((t - self.sim0) / wall, 3) if wall > 1 else None,
                      state=self.meta.get("state"), gathered=len(self.meta.get("gathered", [])), n_truth=int(self._n_truth(live)),
                      ntoys=self.ntoys, frames=self.nframes, kind=self.kind)
            if self.kind == "fly": st.update(sub=self.meta.get("sub"), counts=self.meta.get("counts"))
            _atomic_write(self.dir / "status.json", json.dumps(st).encode())
        if self.stopf.exists():
            return "stop"
        return None
