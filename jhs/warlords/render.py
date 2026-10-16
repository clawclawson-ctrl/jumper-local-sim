"""Save a finished (or stopped) Crab Warlords match as an MP4 with the full overlay at 1x real speed. Render-only: replays the
recorded qpos (25 fps) into a copy of the three-crab world, with each crab's recorded camera / dToF / meta (beliefs, bricks);
nothing is simulated. Called by jhs.render_run for kind=warlords runs (optional user audio: trim / delay / volume / fade, as the
other apps). Streams frame by frame, so 25-minute matches render with flat memory. SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np


def _grid(sp):
    from .belief import NX, NY
    g = [[0.0] * NY for _ in range(NX)]
    for i, j, v in sp: g[i][j] = v
    return g


def main(A, RUN, INFO, OUT, W, H, progress):
    from jhs import TOOLKIT
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
    from jhs.warlords import game as G, world as WW
    WW.install(tasks, G.home("A"), G.home("B"), G.home("C"), int(INFO.get("seed", 1)))

    def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
        import mujoco, imageio
        from PIL import Image
        from jhs.warlords.overlay import WarlordsOverlay, WarlordsCam
        from jhs.warlords import fx as FXW
        sim = env.unwrapped.sim; model, live = sim.env_mjdata(0)
        rmodel = getattr(sim, "render_model", None) or model; WW.skin(rmodel); FXW.theme(rmodel)
        rmodel.vis.global_.offwidth = max(W, rmodel.vis.global_.offwidth); rmodel.vis.global_.offheight = max(H, rmodel.vis.global_.offheight)
        rd = mujoco.MjData(rmodel); rend = mujoco.Renderer(rmodel, H, W, max_geom=12000)
        bricks = {}
        for g in range(rmodel.ngeom):
            n = mujoco.mj_id2name(rmodel, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "wl_brick_" in n: k, i = n[n.index("wl_brick_") + 9:].split("_"); bricks[(k, int(i))] = g
        Q = np.load(RUN / "qpos.npz"); qpos, qt = Q["qpos"], Q["t"]
        meta = json.load(open(RUN / "vision_meta.json")); tz = np.load(RUN / "vision_tof.npz")
        tofs = {k: tz[f"tof{k}"].astype(np.float32) for k in G.KEYS}
        mt = np.array([m["t"] for m in meta])
        st = json.load(open(RUN / "setup.json")) if (RUN / "setup.json").exists() else {}
        ov = WarlordsOverlay(W, H); cam = WarlordsCam(st.get("camera") or os.environ.get("WL_CAMERA")); tmax = float(INFO.get("tmax", 300))
        sel = [i for i in range(len(qt)) if A.t0 <= qt[i] <= A.t1]
        w = imageio.get_writer(str(OUT), fps=25, codec="libx264", quality=8, macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
        t_start = time.time(); n = 0; last_k = -1; imgs = {k: None for k in G.KEYS}
        for j, i in enumerate(sel):
            t = float(qt[i])
            rd.qpos[:] = qpos[i][: rmodel.nq]; mujoco.mj_forward(rmodel, rd)
            k = int(np.clip(np.searchsorted(mt, t, side="right") - 1, 0, len(meta) - 1))
            m = meta[k] if mt[k] <= t + 1e-6 else {}
            if k != last_k:
                for c in G.KEYS:
                    p = RUN / "vis" / f"{c}_{k:05d}.jpg"
                    if p.exists(): imgs[c] = np.asarray(Image.open(p).convert("RGB").resize((640, 480)))
                    if (m.get(c) or {}).get("state") == "out": imgs[c] = None
                for (c, bi), g in bricks.items():
                    al = (m.get("bricks_alive") or {}).get(c); rmodel.geom_rgba[g][3] = 1.0 if (al is None or al[bi]) else 0.0
                last_k = k
            mm = dict(m)
            for c in G.KEYS:
                if mm.get(c) and "belief_sp" in mm[c]: mm[c] = dict(mm[c], belief=_grid(mm[c]["belief_sp"]))
            pts = {c: tuple(mm[c]["pose"][:2]) for c in G.KEYS if (mm.get(c) or {}).get("pose") and mm[c].get("state") != "out"}
            if mm.get("ball"): pts["ball"] = tuple(mm["ball"][:2])
            rend.update_scene(rd, cam.update(t=t, crabs=pts or None)); img = Image.fromarray(rend.render())
            ov.draw(img, mm, {c: v for c, v in imgs.items() if v is not None}, {c: tofs[c][k] for c in G.KEYS}, t, tmax, speed_label="1x speed")
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
