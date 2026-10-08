"""Self-test of the local sim install.   python -m jhs.check [--full]
--full also builds the hide & seek world with the app's controller and steps it (about a minute)."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jhs import ROOT, TOOLKIT  # noqa: E402
from jhs.runner import gl_env  # noqa: E402

ok = True
def item(name, fn):
    global ok
    try:
        r = fn(); print(f"  [ok]   {name}{': ' + str(r) if r else ''}", flush=True)
    except Exception as e:  # noqa: BLE001
        ok = False; print(f"  [FAIL] {name}: {type(e).__name__}: {e}", flush=True)

print(f"Jumper Hide & Seek -- Local Sim self-test  (python {sys.version.split()[0]}, {sys.platform})")
os.environ.update(gl_env())
item("numpy / scipy / pillow", lambda: (__import__("numpy").__version__, __import__("scipy").__version__, __import__("PIL").__version__))
item("mujoco", lambda: __import__("mujoco").__version__)
item("torch", lambda: __import__("torch").__version__)
item("onnxruntime", lambda: __import__("onnxruntime").__version__)
item("robot controller (mjrl_fsm)", lambda: Path(__import__("mjrl_fsm").__file__).name)
item("ffmpeg for MP4", lambda: Path(__import__("imageio_ffmpeg").get_ffmpeg_exe()).name)
def aac():
    import subprocess, imageio_ffmpeg
    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    assert " aac " in out, "the bundled ffmpeg has no AAC encoder"
    return "AAC encoder present (for adding your own audio)"
item("ffmpeg audio", aac)
def render():
    import mujoco
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos='0 0 3'/><geom type='sphere' size='.2' rgba='1 0 0 1'/></worldbody></mujoco>")
    d = mujoco.MjData(m); r = mujoco.Renderer(m, 64, 64); mujoco.mj_forward(m, d); r.update_scene(d); img = r.render()
    assert img.shape == (64, 64, 3) and img.max() > 0, "blank image"
    return f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} renders"
item("off-screen rendering (camera, dToF picture, live view, MP4)", render)
item("toolkit", lambda: (TOOLKIT / "scripts" / "play.py").exists() or (_ for _ in ()).throw(FileNotFoundError("toolkit/scripts/play.py")))
def apps():
    from jhs.apps import inspect
    a = ROOT / "apps" / "jumper_hide_seek_vision.app"; i = inspect(a)
    assert i["ok"] and i["kind"] == "hide_seek", i
    return f"{a.name}: {i['kind']}, controller {i['controller']}, local hooks {i['local_hooks']}"
item("hide & seek app", apps)
def flybrain():
    from jhs.apps import inspect
    a = ROOT / "apps" / "flybrain.app"
    if not a.exists(): return "apps/flybrain.app not present (skipped)"
    i = inspect(a); assert i["ok"] and i["kind"] == "flybrain", i
    for f in ("flybrain_local.py", "sim_only_flybrain.py"):
        assert (ROOT / "jhs" / "brains" / f).exists(), f"jhs/brains/{f} missing"
    import contextlib, io
    sys.path.insert(0, str(ROOT / "jhs" / "brains")); import sim_only_flybrain as FB
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): FB.unit()          # FB_UNIT: synthetic images, one behaviour each
    return f"{a.name}: {i['kind']}, controller {i['controller']}; rules unit test: {buf.getvalue().strip().replace('[unit] ', '')}"
item("flybrain app + fly rules (SIM-ONLY)", flybrain)
item("room v7 map", lambda: (ROOT / "maps" / "jumper-hide-seek-v7.map").stat().st_size)
if "--full" in sys.argv:
    def world():
        from jhs.runner import start_hide_seek
        t0 = time.time()
        proc, rd = start_hide_seek(ROOT / "apps" / "jumper_hide_seek_vision.app", toys=None, start=None, seed=31, tmax=15, speed=0, live=True)
        rc = proc.wait(timeout=900)
        log = (rd / "log.txt").read_text(errors="replace")
        assert rc == 0 and (rd / "result.json").exists(), f"exit {rc}; see {rd / 'log.txt'}:\n" + "\n".join(log.splitlines()[-15:])
        ctrl = next((l.strip() for l in log.splitlines() if "controller" in l and ("from the app" in l or "installed" in l)), "?")
        assert (rd / "live" / "live.jpg").exists(), "no live frame written"
        return f"15 s of sim in {time.time() - t0:.0f} s wall; {ctrl}; run dir {rd}"
    item("hide & seek world: app + map + controller + brain + live frame (15 s sim)", world)
print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
