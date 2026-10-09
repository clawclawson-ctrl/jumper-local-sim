"""Save a finished (or stopped) local run as an MP4 with the full demo overlay, at 1x real speed.
Render-only: replays the run's recorded qpos into a copy of the world; nothing is simulated.
    python -m jhs.render_run RUN_DIR [OUT.mp4] [--size 1080|720|480] [--from S] [--to S]
SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jhs import TOOLKIT  # noqa: E402
from jhs.runner import gl_env  # noqa: E402
for _k, _v in gl_env().items(): os.environ.setdefault(_k, _v)

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("out", nargs="?"); ap.add_argument("--size", default="1080", choices=["1080", "720", "480"])
ap.add_argument("--from", dest="t0", type=float, default=0.0); ap.add_argument("--to", dest="t1", type=float, default=1e9)
ap.add_argument("--audio", help="optional music/audio file to add (mp3, wav, m4a, aac, ...); the video stays 1x and unchanged")
ap.add_argument("--audio-start", type=float, default=0.0, help="skip the first S seconds of the audio file")
ap.add_argument("--audio-delay", type=float, default=0.0, help="audio begins S seconds into the video")
ap.add_argument("--audio-volume", type=float, default=1.0, help="loudness multiplier (1.0 = unchanged)")
ap.add_argument("--audio-fade", type=float, default=3.0, help="fade-out over the last S seconds (0 = none)")
ap.add_argument("--audio-loop", action="store_true", help="repeat the audio if shorter than the video (default: once, then silence)")
A = ap.parse_args()
RUN = Path(A.run).resolve(); INFO = json.loads((RUN / "local_run.json").read_text())
OUT = Path(A.out or (RUN / "run_1x.mp4")).expanduser().resolve(); OUT.parent.mkdir(parents=True, exist_ok=True)
W, H = {"1080": (1920, 1080), "720": (1280, 720), "480": (848, 480)}[A.size]
PROG = Path(str(OUT) + ".progress.json")
def progress(**k): PROG.write_text(json.dumps(k))

if INFO.get("kind") == "soccer":     # two-crab world: its own replay (jhs/soccer/render.py)
    from jhs.soccer.render import main as _soccer_render
    _soccer_render(A, RUN, INFO, OUT, W, H, progress); sys.exit(0)

REPO = Path(os.environ.get("JUMPER_REPO", str(TOOLKIT)))
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("play", REPO / "scripts" / "play.py")
play = importlib.util.module_from_spec(spec); sys.modules["play"] = play; spec.loader.exec_module(play)
import controller  # noqa: E402
class NoPad:
    connected = False; name = "none"; path = "none"
    def state(self): return controller.GamepadState()
    def close(self): pass
controller.open = lambda *a, **k: NoPad()


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
    import mujoco, imageio
    from PIL import Image
    from jhs.overlay import Overlay, FollowCam
    sim = env.unwrapped.sim; model, live = sim.env_mjdata(0)
    rmodel = getattr(sim, "render_model", None) or model
    rmodel.vis.global_.offwidth = max(W, rmodel.vis.global_.offwidth); rmodel.vis.global_.offheight = max(H, rmodel.vis.global_.offheight)
    rd = mujoco.MjData(rmodel); rend = mujoco.Renderer(rmodel, H, W)
    Q = np.load(RUN / "qpos.npz"); qpos, qt = Q["qpos"], Q["t"]
    meta = json.load(open(RUN / "vision_meta.json")); _tz = np.load(RUN / "vision_tof.npz"); tofs = _tz["tof"].astype(np.float32)
    floors = _tz["floor"] if "floor" in _tz.files else None
    mt = np.array([m["t"] for m in meta])
    FLY = INFO.get("kind") == "flybrain"
    from jhs.pushable import prop_geoms, footprints
    pg = prop_geoms(rmodel)          # obstacles where they are in this frame (sim truth, display only)
    if FLY:
        from jhs.fly_overlay import FlyOverlay
        ov = FlyOverlay(W, H); fc = FollowCam()
        poses = [(m["pose"][0], m["pose"][1]) for m in meta]
    else:
        M = np.load(RUN / "maps.npz"); midx, maps = M["idx"], M["maps"]
        res = json.load(open(RUN / "result.json")); toys = list(res["placed"].keys()); NT = len(toys); RUG = INFO.get("rug", [0.0, 0.0, 0.87, 0.64])
        truth = [json.loads(l) for l in open(RUN / "truth.jsonl")]; tt = np.array([x["t"] for x in truth])
        def n_truth(t):   # display only (sim truth); the crab never sees this
            x = truth[int(np.clip(np.searchsorted(tt, t), 0, len(truth) - 1))]
            return sum(abs(x[c][0] - RUG[0]) < RUG[2] and abs(x[c][1] - RUG[1]) < RUG[3] for c in toys)
        ov = Overlay(W, H); fc = FollowCam()
    sel = [i for i in range(len(qt)) if A.t0 <= qt[i] <= A.t1]
    w = imageio.get_writer(str(OUT), fps=25, codec="libx264", quality=8, macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
    t_start = time.time(); n = 0; last_k = -1; cam_img = None
    for j, i in enumerate(sel):
        t = float(qt[i])
        rd.qpos[:] = qpos[i][: rmodel.nq]; mujoco.mj_forward(rmodel, rd)
        rend.update_scene(rd, fc.update(rd.qpos)); img = Image.fromarray(rend.render())
        k = int(np.clip(np.searchsorted(mt, t, side="right") - 1, 0, len(meta) - 1))
        if k != last_k:
            cp = RUN / "vis" / f"cam_{k:05d}.jpg"; cam_img = Image.open(cp).convert("RGB") if cp.exists() else None; last_k = k
        if FLY:
            ov.draw(img, meta[k] if mt[k] <= t + 1e-6 else {}, cam_img, tofs[k], t, speed_label="1x speed", props=footprints(rmodel, rd, pg), path=poses[: k + 1],
                    floor=floors[k] if floors is not None else None)
        else:
            jm = int(np.clip(np.searchsorted(midx, k, side="right") - 1, 0, len(midx) - 1)) if len(midx) else None
            ov.draw(img, meta[k], cam_img, tofs[k], maps[jm] if jm is not None else None, t, n_truth(t), NT, speed_label="1x speed",
                    props=footprints(rmodel, rd, pg))
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
            audio_note = f"audio added: {Path(A.audio).name}"
            Path(str(OUT) + ".audio.log").unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001  -- keep the silent video rather than lose the render
            audio_note = f"AUDIO NOT ADDED ({e}); the video was saved without sound"
        print(f"[render] {audio_note}", flush=True)
    progress(done=len(sel), total=len(sel), finished=True, out=str(OUT), frames=n, seconds=round(n / 25, 2), audio=audio_note)
    print(f"[render] wrote {OUT} frames={n} ({n / 25:.1f} s at 1x)", flush=True)


play._loop = _loop
sys.argv = [str(REPO / "scripts" / "play.py"), "--app", INFO["app"], "--scene", INFO["scene"], "--backend", "native", "--device", "cpu", "--headless"]
progress(done=0, total=0, starting=True, out=str(OUT))
play.main()
