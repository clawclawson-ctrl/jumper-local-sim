"""Live view / real-time cap / Stop for Crab Warlords. DISPLAY ONLY. SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import io, json, os, time
from pathlib import Path


def _atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_bytes(data); os.replace(tmp, path)


class WarlordsHook:
    def __init__(self, sim, outd, dt):
        import mujoco
        from .overlay import WarlordsOverlay, WarlordsCam
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
        self.rend = mujoco.Renderer(self.rmodel, self.H, self.W, max_geom=12000) if self.enabled else None
        cm = (json.load(open(os.environ["WL_SETUP"])).get("camera") if os.environ.get("WL_SETUP") else None) or os.environ.get("WL_CAMERA")
        self.ov = WarlordsOverlay(self.W, self.H); self.cam = WarlordsCam(cm)
        self.img = {}; self.tof = {}; self.m = {}
        self.last = -1e9; self.t0 = None; self.sim0 = 0.0; self.nframes = 0; self.last_status = 0.0

    def vision(self, k, img, rngs): self.img[k] = img; self.tof[k] = rngs
    def meta(self, m): self.m = m

    def frame(self, t, rate=0.0):
        from PIL import Image
        model, live = self.sim.env_mjdata(0)
        self.rd.qpos[:] = live.qpos[: self.rmodel.nq]; self.mj.mj_forward(self.rmodel, self.rd)
        pts = {k: tuple(self.m[k]["pose"][:2]) for k in ("A", "B", "C") if (self.m.get(k) or {}).get("pose") and self.m[k].get("state") != "out"}
        if self.m.get("ball"): pts["ball"] = tuple(self.m["ball"][:2])
        self.rend.update_scene(self.rd, self.cam.update(t=t, crabs=pts or None))
        fr = Image.fromarray(self.rend.render())
        self.ov.draw(fr, self.m, self.img, self.tof, t, self.tmax, speed_label=f"live, {rate:.2f}x" if rate else "live")
        return fr

    def step(self, t):
        now = time.time()
        if self.t0 is None: self.t0 = now; self.sim0 = t
        if self.speed > 0:
            ahead = (t - self.sim0) / self.speed - (now - self.t0)
            if ahead > 0.002: time.sleep(min(ahead, 0.5))
        if self.enabled and t - self.last >= self.every and self.m:
            self.last = t; wall = time.time() - self.t0; rate = (t - self.sim0) / wall if wall > 1 else 0.0
            fr = self.frame(t, rate); buf = io.BytesIO(); fr.save(buf, "JPEG", quality=82); _atomic(self.dir / "live.jpg", buf.getvalue()); self.nframes += 1
        if now - self.last_status > 1.0:
            self.last_status = now; wall = now - self.t0; sc = self.m.get("score", {})
            st = dict(t=round(t, 2), wall=round(wall, 1), rate=round((t - self.sim0) / wall, 3) if wall > 1 else None, kind="warlords",
                      state=" - ".join(f"{n} {sc.get(k, 0)}" for k, n in (("A", "RED"), ("B", "BLUE"), ("C", "GREEN"))), score=sc, ref=self.m.get("ref"),
                      frames=self.nframes, final=self.m.get("final"))
            _atomic(self.dir / "status.json", json.dumps(st).encode())
        return "stop" if self.stopf.exists() else None
