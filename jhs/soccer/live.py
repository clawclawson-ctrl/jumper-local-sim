"""Live view / real-time cap / Stop for a soccer match (inside the sim process). DISPLAY ONLY: nothing here reaches a
brain. SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import io, json, os, time
from pathlib import Path


def _atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_bytes(data); os.replace(tmp, path)


class SoccerHook:
    def __init__(self, sim, outd, dt):
        import mujoco
        from .overlay import SoccerOverlay, BroadcastCam
        self.mj = mujoco; self.sim = sim; self.dt = dt
        self.dir = Path(os.environ.get("JHS_LIVE_DIR", str(Path(outd) / "live"))); self.dir.mkdir(parents=True, exist_ok=True)
        self.stopf = Path(os.environ.get("JHS_STOP", str(self.dir / "STOP")))
        self.every = float(os.environ.get("JHS_LIVE_EVERY", 0.25)); self.speed = float(os.environ.get("JHS_SPEED", 1.0))
        self.W, self.H = int(os.environ.get("JHS_LIVE_W", 1280)), int(os.environ.get("JHS_LIVE_H", 720))
        self.enabled = os.environ.get("JHS_LIVE", "1") == "1"; self.tmax = float(os.environ.get("VS_TMAX", 300))
        model, live = sim.env_mjdata(0)
        self.rmodel = getattr(sim, "render_model", None) or model
        self.rmodel.vis.global_.offwidth = max(self.W, self.rmodel.vis.global_.offwidth); self.rmodel.vis.global_.offheight = max(self.H, self.rmodel.vis.global_.offheight)
        self.rd = mujoco.MjData(self.rmodel)
        self.rend = mujoco.Renderer(self.rmodel, self.H, self.W) if self.enabled else None
        self.ov = SoccerOverlay(self.W, self.H); self.cam = BroadcastCam()
        self.img = {"A": None, "B": None}; self.tof = {"A": None, "B": None}; self.m = {}
        self.last = -1e9; self.t0 = None; self.sim0 = 0.0; self.nframes = 0; self.last_status = 0.0
        from jhs.pushable import prop_geoms
        self.pg = prop_geoms(model)

    def vision(self, k, img, rngs): self.img[k] = img; self.tof[k] = rngs
    def meta(self, m): self.m = m

    def step(self, t):
        now = time.time()
        if self.t0 is None: self.t0 = now; self.sim0 = t
        if self.speed > 0:
            ahead = (t - self.sim0) / self.speed - (now - self.t0)
            if ahead > 0.002: time.sleep(min(ahead, 0.5))
        if self.enabled and t - self.last >= self.every and self.m:
            self.last = t
            from PIL import Image
            from jhs.pushable import footprints
            model, live = self.sim.env_mjdata(0)
            self.rd.qpos[:] = live.qpos[: self.rmodel.nq]; self.mj.mj_forward(self.rmodel, self.rd)
            self.rend.update_scene(self.rd, self.cam.update(self.m.get("ball_truth")))
            frame = Image.fromarray(self.rend.render())
            wall = time.time() - self.t0; rate = (t - self.sim0) / wall if wall > 1 else 0.0
            self.ov.draw(frame, self.m, self.img["A"], self.img["B"], self.tof["A"], self.tof["B"], t, self.tmax,
                         speed_label=f"live, {rate:.2f}x" if rate else "live", props=footprints(model, live, self.pg), ball_truth=self.m.get("ball_truth"))
            buf = io.BytesIO(); frame.save(buf, "JPEG", quality=82); _atomic(self.dir / "live.jpg", buf.getvalue()); self.nframes += 1
        if now - self.last_status > 1.0:
            self.last_status = now; wall = now - self.t0
            sc = self.m.get("score", {"A": 0, "B": 0})
            st = dict(t=round(t, 2), wall=round(wall, 1), rate=round((t - self.sim0) / wall, 3) if wall > 1 else None, kind="soccer",
                      state=f"RED {sc.get('A', 0)} - {sc.get('B', 0)} BLUE", score=sc, ref=self.m.get("ref"), frames=self.nframes,
                      stateA=(self.m.get("A") or {}).get("state"), stateB=(self.m.get("B") or {}).get("state"), final=self.m.get("final"))
            _atomic(self.dir / "status.json", json.dumps(st).encode())
        return "stop" if self.stopf.exists() else None
