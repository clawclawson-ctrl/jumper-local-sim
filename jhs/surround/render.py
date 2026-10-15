"""Save a finished (or stopped) Crab Surround match as an MP4 with the full overlay at 1x. Render-only: replays the recorded qpos and
trails into a copy of the two-crab world; nothing is simulated. Called by jhs.render_run for kind=surround runs. SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np


def main(A, RUN, INFO, OUT, W, H, progress):
    os.environ["TK_ARENA"] = "open"
    from jhs import TOOLKIT
    from jhs.soccer import world as WD
    REPO = Path(os.environ.get("JUMPER_REPO", str(TOOLKIT)))
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("play", REPO / "scripts" / "play.py")
    play = importlib.util.module_from_spec(spec); sys.modules["play"] = play; spec.loader.exec_module(play)
    import controller, tasks
    class NoPad:
        connected = False; name = "none"; path = "none"
        def state(self): return controller.GamepadState()
        def close(self): pass
    controller.open = lambda *a, **k: NoPad()
    from jhs.surround import game as G
    WD.install(tasks, G.START["A"], G.START["B"], int(INFO.get("seed", 1)))

    def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
        import mujoco, imageio
        from PIL import Image
        from jhs.surround.overlay import SurroundOverlay
        from jhs.tanks.overlay import BroadcastCam
        from jhs.surround import fx as FXD
        sim = env.unwrapped.sim; model, live = sim.env_mjdata(0)
        rmodel = getattr(sim, "render_model", None) or model; WD.skin(rmodel); FXD.theme(rmodel)
        rmodel.vis.global_.offwidth = max(W, rmodel.vis.global_.offwidth); rmodel.vis.global_.offheight = max(H, rmodel.vis.global_.offheight)
        rd = mujoco.MjData(rmodel); rend = mujoco.Renderer(rmodel, H, W, max_geom=12000)
        Q = np.load(RUN / "qpos.npz"); qpos, qt = Q["qpos"], Q["t"]
        meta = json.load(open(RUN / "vision_meta.json")); tz = np.load(RUN / "vision_tof.npz")
        tA, tB = tz["tofA"].astype(np.float32), tz["tofB"].astype(np.float32)
        mt = np.array([m["t"] for m in meta])
        FXB = json.load(open(RUN / "surround_fx.json")); TR = json.load(open(RUN / "trails.json"))
        st = json.load(open(RUN / "setup.json")) if (RUN / "setup.json").exists() else {}
        ov = SurroundOverlay(W, H); bc = BroadcastCam(st.get("camera") or os.environ.get("SR_CAMERA")); tmax = float(INFO.get("tmax", 300))
        sel = [i for i in range(len(qt)) if A.t0 <= qt[i] <= A.t1]
        w = imageio.get_writer(str(OUT), fps=25, codec="libx264", quality=8, macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
        t_start = time.time(); n = 0; last_k = -1; cams = (None, None)
        for j, i in enumerate(sel):
            t = float(qt[i])
            rd.qpos[:] = qpos[i][: rmodel.nq]; mujoco.mj_forward(rmodel, rd)
            k = int(np.clip(np.searchsorted(mt, t, side="right") - 1, 0, len(meta) - 1))
            if k != last_k:
                def ld(tag):
                    p = RUN / "vis" / f"{tag}_{k:05d}.jpg"; return Image.open(p).convert("RGB") if p.exists() else None
                cams = (ld("A"), ld("B")); last_k = k
            m = meta[k] if mt[k] <= t + 1e-6 else {}
            fx = FXB[i] if i < len(FXB) else {"r": len(TR) - 1, "n": {}}
            rr = TR[min(fx["r"], len(TR) - 1)]
            trails = {kk: [tuple(p) for p in rr.get(kk, [])[: fx["n"].get(kk, 0)]] for kk in ("A", "B")}
            crabs = {kk: tuple(float(v) for v in rd.xpos[mujoco.mj_name2id(rmodel, mujoco.mjtObj.mjOBJ_BODY, pf + "base_link")][:2]) for kk, pf in (("A", "robot/"), ("B", "robot2/"))}
            rend.update_scene(rd, bc.update(t=t, crabs=crabs)); FXD.draw(mujoco, rend.scene, trails); img = Image.fromarray(rend.render())
            ov.draw(img, m, cams[0], cams[1], tA[k], tB[k], t, tmax, speed_label="1x speed", trails=trails)
            w.append_data(np.asarray(img)); n += 1
            if j % 25 == 0:
                el = time.time() - t_start
                progress(done=j, total=len(sel), fps=round(n / el, 2) if el > 0 else None, eta_s=round((len(sel) - j) / (n / el)) if el > 0 and n else None, out=str(OUT))
        w.close()
        audio_note = None
        if A.audio:
            progress(done=len(sel), total=len(sel), muxing=True, out=str(OUT))
            from jhs.audio import mux
            try:
                mux(OUT, A.audio, None, A.audio_start, A.audio_delay, A.audio_volume, A.audio_fade, A.audio_loop, log=str(OUT) + ".audio.log")
                audio_note = f"audio added: {Path(A.audio).name}"; Path(str(OUT) + ".audio.log").unlink(missing_ok=True)
            except Exception as e:  # noqa: BLE001
                audio_note = f"AUDIO NOT ADDED ({e}); the video was saved without sound"
            print(f"[render] {audio_note}", flush=True)
        progress(done=len(sel), total=len(sel), finished=True, out=str(OUT), frames=n, seconds=round(n / 25, 2), audio=audio_note)
        print(f"[render] wrote {OUT} frames={n} ({n / 25:.1f} s at 1x)", flush=True)

    play._loop = _loop
    sys.argv = [str(REPO / "scripts" / "play.py"), "--app", INFO["app"], "--scene", INFO["scene"], "--backend", "native", "--device", "cpu", "--headless"]
    progress(done=0, total=0, starting=True, out=str(OUT))
    play.main()
