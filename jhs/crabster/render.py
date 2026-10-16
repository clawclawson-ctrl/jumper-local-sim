"""Save a Crabster show as an MP4 at 1x with the full overlay and a title card. Render-only: replays the recorded qpos (and the
recorded thrust / suction / gas / g per frame) into a copy of the stage; nothing is simulated. Called by jhs.render_run.
SCRIPTED STUNT, SIM-ONLY THRUSTER + SUCTION CONCEPT."""
from __future__ import annotations
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np


def main(A, RUN, INFO, OUT, W, H, progress):
    from jhs import TOOLKIT
    REPO = Path(os.environ.get("JUMPER_REPO", str(TOOLKIT)))
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("play", REPO / "scripts" / "play.py")
    play = importlib.util.module_from_spec(spec); sys.modules["play"] = play; spec.loader.exec_module(play)
    import controller
    class NoPad:
        connected = False; name = "none"; path = "none"
        def state(self): return controller.GamepadState()
        def close(self): pass
    controller.open = lambda *a, **k: NoPad()

    def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
        import imageio
        from jhs.crabster.view import Painter, caption
        sim = env.unwrapped.sim; model, live = sim.env_mjdata(0)
        rmodel = getattr(sim, "render_model", None) or model
        Q = np.load(RUN / "qpos.npz", allow_pickle=True); qt = Q["t"]
        res = json.loads((RUN / "result.json").read_text()); ev = [json.loads(l) for l in open(RUN / "events.jsonl")] if (RUN / "events.jsonl").exists() else []
        cam = INFO.get("camera") or res.get("camera") or "follow"
        P = Painter(rmodel, W, H, cam)
        sel = [i for i in range(len(qt)) if A.t0 <= qt[i] <= A.t1]
        w = imageio.get_writer(str(OUT), fps=25, codec="libx264", quality=8, macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
        t_start = time.time(); n = 0
        for j, i in enumerate(sel):
            t = float(qt[i])
            rec = dict(lift=float(Q["lift"][i]), tq=Q["tq"][i], push=Q["push"][i], suction=float(Q["suction"][i]), gas=float(Q["gas"][i]), g=float(Q["g"][i]), phase=str(Q["phase"][i]))
            done = [r for r in res.get("results", []) if r["t"] <= t]
            tc = 1.0 if t < 1.6 else max(0.0, 1.0 - (t - 1.6) / 0.6)
            img = P.frame(Q["qpos"][i], t, rec, caption(ev, t), done, "1x speed", title_card=tc)
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
