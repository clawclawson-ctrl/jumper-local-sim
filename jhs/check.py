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
    assert i["ok"] and i["kind"] == "hide_seek" and i.get("local_hooks"), i
    return f"{a.name}: {i['kind']}, controller {i['controller']}, local hooks {i['local_hooks']}"
item("hide & seek app", apps)
def tidy():
    from jhs.apps import inspect
    a = ROOT / "apps" / "jumper_tidy_vision.app"
    if not a.exists(): return "apps/jumper_tidy_vision.app not present (skipped)"
    i = inspect(a)
    assert i["ok"] and i["kind"] == "tidy_vision" and i.get("local_hooks"), i
    m = ROOT / "maps" / "jumper-tidy-room.map"; assert m.exists(), "maps/jumper-tidy-room.map missing"
    return f"{a.name}: {i['kind']}, local hooks {i['local_hooks']}, room map {m.stat().st_size} B"
item("tidy-up vision app", tidy)
def tidy_score():
    from PIL import Image
    import numpy as np
    from jhs.overlay import tidy_in_bin, Overlay, TIDY_ROOM
    basket, crate = (1.0, 0.0), (-1.0, 0.0); bins = {"basket": basket, "crate": crate}
    assert tidy_in_bin((1.2, 0.0), "ball", bins) and tidy_in_bin((1.2, 0.0), "minifb", bins)
    assert not tidy_in_bin((1.2, 0.0), "duck", bins) and tidy_in_bin((-1.1, 0.1), "duck", bins)
    assert not tidy_in_bin((0.0, 0.0), "ball", bins)
    ov = Overlay(1280, 720); tof = np.ones((42, 54), np.float32); cam = Image.new("RGB", (64, 48), (80, 80, 80))
    meta = dict(state="push", focus="duck", dets=[dict(cls="duck", bbox=[8, 8, 30, 30], rng=0.8)], gathered=["ball"],
                pose=[0.2, -0.4, 0.5], tracks=[("duck", 0.3, -0.2, False), ("ball", 1.1, 0.1, True)],
                bins=[("basket", 1.0, 0.2), ("crate", -0.6, 0.4)])
    ov.draw(Image.new("RGB", (1280, 720), (20, 20, 20)), meta, cam, tof, None, 1.5, 1, 3, task="tidy", room=TIDY_ROOM)
    ov.draw(Image.new("RGB", (1280, 720), (20, 20, 20)), dict(state="scan", dets=[], gathered=[]), None, None, None, 1.0, 0, 3)
    return "ball and mini football count for the basket, the duck for the crate; both overlays draw"
item("tidy scorer and live overlay", tidy_score)
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
def soccer():
    a = ROOT / "apps" / "jumper_soccer_vision.app"
    if not a.exists(): return "apps/jumper_soccer_vision.app not present (skipped)"
    from jhs.apps import inspect as _insp
    i = _insp(a); assert i["ok"] and i["kind"] == "soccer", i
    m = ROOT / "maps" / "jumper-soccer-field.map"; assert m.exists(), "maps/jumper-soccer-field.map missing"
    from jhs.soccer import brain as SB
    return f"{a.name}: soccer, controller {i['controller']}; field map {m.stat().st_size} B; brain unit test: " + "; ".join(SB.unit())
def ballroom():
    from jhs.ballroom.choreo import check as dance_check
    from jhs.ballroom.gait import check as gait_check
    msg = dance_check() + "; " + gait_check()
    import mujoco
    from jhs.ballroom.scene import build
    model, data, _moon, _candle = build()
    rend = mujoco.Renderer(model, 180, 320)
    mujoco.mj_kinematics(model, data)
    rend.update_scene(data)
    img = rend.render()
    rend.close()
    assert int(img.max()) > 8, "the ballroom rendered black"
    return msg
item("moonlit ballroom (two crabs, floor pattern)", ballroom)
item("crab soccer app + field + brains (SIM-ONLY)", soccer)
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
