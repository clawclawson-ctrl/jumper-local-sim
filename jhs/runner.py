"""Start a run of an app in the local sim (a separate process), and save runs as MP4. SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path
from . import ROOT, TOOLKIT, PKG
from .apps import inspect, extract_vision

RUNS = Path(os.environ.get("JHS_RUNS", str(ROOT / "runs")))
MAPS = {"hide_seek": ROOT / "maps" / "jumper-hide-seek-v7.map", "tidy_vision": ROOT / "maps" / "jumper-tidy-room.map",
        "soccer": ROOT / "maps" / "jumper-soccer-field.map"}
DEFAULT_MOVIES = Path.home() / "Movies" / "Jumper Hide & Seek"


def gl_env() -> dict:
    """Off-screen OpenGL for the camera/dToF/live picture: CGL on a Mac, EGL (else OSMesa) on Linux."""
    gl = os.environ.get("JHS_GL") or os.environ.get("MUJOCO_GL")
    if not gl: gl = "cgl" if sys.platform == "darwin" else "egl"
    env = {"MUJOCO_GL": gl}
    if gl in ("egl", "osmesa"): env["PYOPENGL_PLATFORM"] = gl
    return env


def base_env() -> dict:
    env = dict(os.environ); env.update(gl_env())
    env["JUMPER_REPO"] = str(TOOLKIT); env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "2"); env.setdefault("VS_THREADS", "2")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def new_run_dir(tag="run") -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    d = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"; d.mkdir(); return d


def start_hide_seek(app, toys=None, start=None, seed=1, tmax=600.0, speed=1.0, live=True, live_size=(1280, 720), toyset=("ball", "duck", "duck2"), physics_hz=None,
                    pushable=True, moves=None):
    """toys: {name: [x, y]} hand placement, or None for the app's random hidden placement (seed).
    pushable: stair steps / planters / crates are free bodies the crab can shove (jhs.pushable); False = the map as evaluated.
    moves: {obstacle id: [x, y, yaw_deg]} obstacles moved on the setup screen.
    start: [x, y, heading_deg] or None (rug centre, random heading from the seed). Returns (Popen, run_dir)."""
    info = inspect(app)
    if not info.get("ok"): raise ValueError(info.get("error"))
    if info["kind"] != "hide_seek": raise ValueError("not a hide & seek app")
    rd = new_run_dir("hideseek")
    vis = extract_vision(Path(info["path"]), rd / "app_brain")
    env = base_env()
    env.update(VS_SEED=str(int(seed)), VS_OUT=str(rd), VS_TMAX=str(float(tmax)), VS_SAVEVIS="1", VS_QPOS="1",
               VS_TOYS=",".join(toyset), VS_WALL_CL="0.55", JHS_SPEED=str(float(speed)), JHS_LIVE="1" if live else "0",
               JHS_LIVE_DIR=str(rd / "live"), JHS_STOP=str(rd / "STOP"), JHS_LIVE_W=str(live_size[0]), JHS_LIVE_H=str(live_size[1]))
    if info.get("local_hooks"):
        env["VS_LIVE_HOOK"] = str(PKG / "live_hook.py")
        if start is not None: env["VS_START"] = ",".join(str(float(v)) for v in start)
        if toys:
            (rd / "placement.json").write_text(json.dumps(toys, indent=1)); env["VS_PLACE"] = str(rd / "placement.json")
    scene = MAPS["hide_seek"]
    if pushable or moves:
        from .pushable import variant
        scene = variant(MAPS["hide_seek"], rd / "scene.map", pushable=bool(pushable), moves=moves or None)
    app_for_play = Path(info["path"])
    if os.environ.get("JHS_FORCE_LOCAL_CONTROLLER") == "1":
        app_for_play = strip_controller(app_for_play, rd / "app_local_controller.app")
    (rd / "local_run.json").write_text(json.dumps(dict(app=str(app_for_play), app_original=info["path"], scene=str(scene),
                                                       pushable=bool(pushable), moves=moves or {},
                                                       kind="hide_seek", seed=seed, tmax=tmax, physics_hz=physics_hz or 1000, toys=toys, start=start, speed=speed,
                                                       rug=[0.0, 0.0, 0.87, 0.64], started=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                       info=info), indent=1))
    cmd = [sys.executable, str(vis / "hide_seek.py"), "--app", str(app_for_play), "--scene", str(scene),
           "--backend", "native", "--device", "cpu", "--headless", "--steps", "999999999"]
    if physics_hz: cmd += ["--physics-hz", str(int(physics_hz))]
    log = open(rd / "log.txt", "w")
    proc = subprocess.Popen(cmd, cwd=str(rd), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (rd / "pid").write_text(str(proc.pid))
    return proc, rd


def start_flybrain(app, toys=None, start=None, seed=1, tmax=120.0, speed=1.0, live=True, live_size=(1280, 720), physics_hz=None,
                   pushable=True, moves=None):
    """flybrain.app driven by the fly-inspired rules (jhs/brains/flybrain_local.py) in the hide & seek room v7.
    toys {name: [x, y]} (default: none placed -> all parked outside), start [x, y, heading_deg], pushable/moves as for hide & seek."""
    info = inspect(app)
    if not info.get("ok"): raise ValueError(info.get("error"))
    if info["kind"] != "flybrain": raise ValueError("not a flybrain app")
    rd = new_run_dir("flybrain")
    env = base_env()
    env.update(VS_SEED=str(int(seed)), VS_OUT=str(rd), VS_TMAX=str(float(tmax)), VS_SAVEVIS="1", JHS_SPEED=str(float(speed)),
               JHS_LIVE="1" if live else "0", JHS_LIVE_DIR=str(rd / "live"), JHS_STOP=str(rd / "STOP"),
               JHS_LIVE_W=str(live_size[0]), JHS_LIVE_H=str(live_size[1]), VS_LIVE_HOOK=str(PKG / "live_hook.py"))
    if start is not None: env["VS_START"] = ",".join(str(float(v)) for v in start)
    if toys:
        (rd / "placement.json").write_text(json.dumps(toys, indent=1)); env["VS_PLACE"] = str(rd / "placement.json")
    scene = MAPS["hide_seek"]
    if pushable or moves:
        from .pushable import variant
        scene = variant(MAPS["hide_seek"], rd / "scene.map", pushable=bool(pushable), moves=moves or None)
    app_for_play = Path(info["path"])
    if os.environ.get("JHS_FORCE_LOCAL_CONTROLLER") == "1":
        app_for_play = strip_controller(app_for_play, rd / "app_local_controller.app")
    (rd / "local_run.json").write_text(json.dumps(dict(app=str(app_for_play), app_original=info["path"], scene=str(scene), kind="flybrain",
                                                       pushable=bool(pushable), moves=moves or {}, seed=seed, tmax=tmax, physics_hz=physics_hz or 1000,
                                                       toys=toys, start=start, speed=speed, started=time.strftime("%Y-%m-%d %H:%M:%S"), info=info), indent=1))
    cmd = [sys.executable, str(PKG / "brains" / "flybrain_local.py"), "--app", str(app_for_play), "--scene", str(scene),
           "--backend", "native", "--device", "cpu", "--headless", "--steps", "999999999"]
    if physics_hz: cmd += ["--physics-hz", str(int(physics_hz))]
    log = open(rd / "log.txt", "w")
    proc = subprocess.Popen(cmd, cwd=str(rd), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (rd / "pid").write_text(str(proc.pid))
    return proc, rd


def start_soccer(app, setup=None, seed=1, tmax=300.0, speed=1.0, live=True, live_size=(1280, 720), physics_hz=None, pushable=True, goals_to_win=3):
    """1v1 crab soccer (jhs/soccer/run.py) on the soccer field. setup = {ball: [x, y], A: [x, y, deg], B: [x, y, deg],
    props: {obstacle id: [x, y, deg]}} (validated by jhs.soccer.field.validate); obstacles not listed stay parked outside."""
    from .soccer import field as SF
    info = inspect(app)
    if not info.get("ok"): raise ValueError(info.get("error"))
    if info["kind"] != "soccer": raise ValueError("not a soccer app")
    setup = dict(setup or {}); setup.setdefault("ball", [0.0, 0.0]); setup.setdefault("A", list(SF.KICKOFF["A"])); setup.setdefault("B", list(SF.KICKOFF["B"]))
    setup["props"] = {k: v for k, v in (setup.get("props") or {}).items() if k in SF.PROPS}
    errs = [m for l, m in SF.validate(setup) if l == "error"]
    if errs: raise ValueError("invalid setup: " + "; ".join(errs))
    rd = new_run_dir("soccer")
    (rd / "setup.json").write_text(json.dumps(setup, indent=1))
    env = base_env()
    env.update(VS_SEED=str(int(seed)), VS_OUT=str(rd), VS_TMAX=str(float(tmax)), VS_SAVEVIS="1", JHS_SPEED=str(float(speed)), SC_LIVE="1",
               JHS_LIVE="1" if live else "0", JHS_LIVE_DIR=str(rd / "live"), JHS_STOP=str(rd / "STOP"), SC_SETUP=str(rd / "setup.json"), SC_PUSHABLE="1" if pushable else "0", SC_GOALS=str(max(1, min(10, int(goals_to_win)))),
               JHS_LIVE_W=str(live_size[0]), JHS_LIVE_H=str(live_size[1]))
    from .pushable import variant
    scene = variant(MAPS["soccer"], rd / "scene.map", pushable=bool(pushable), moves=setup["props"] or None)
    app_for_play = Path(info["path"])
    if os.environ.get("JHS_FORCE_LOCAL_CONTROLLER") == "1":
        app_for_play = strip_controller(app_for_play, rd / "app_local_controller.app")
    (rd / "local_run.json").write_text(json.dumps(dict(app=str(app_for_play), app_original=info["path"], scene=str(scene), kind="soccer",
                                                       pushable=bool(pushable), setup=setup, seed=seed, tmax=tmax, goals_to_win=int(goals_to_win), physics_hz=physics_hz or 1000,
                                                       speed=speed, started=time.strftime("%Y-%m-%d %H:%M:%S"), info=info), indent=1))
    cmd = [sys.executable, str(PKG / "soccer" / "run.py"), "--app", str(app_for_play), "--scene", str(scene),
           "--backend", "native", "--device", "cpu", "--headless", "--steps", "999999999"]
    if physics_hz: cmd += ["--physics-hz", str(int(physics_hz))]
    log = open(rd / "log.txt", "w")
    proc = subprocess.Popen(cmd, cwd=str(rd), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (rd / "pid").write_text(str(proc.pid))
    return proc, rd


def start_tidy(app, seed=1, tmax=420.0):
    """EXPERIMENTAL: a tidy-up vision app (vision/tidy_vision.py) on the shipped tidy-up room, its own toy layout.
    No hand placement, no live picture, no MP4 (its recordings use a different format); the log shows progress."""
    info = inspect(app)
    rd = new_run_dir("tidy")
    vis = extract_vision(Path(info["path"]), rd / "app_brain")
    src = (vis / "tidy_vision.py").read_text()
    src = src.replace('REPO = Path("/workspace/jumper")', 'REPO = Path(os.environ.get("JUMPER_REPO", "/workspace/jumper"))', 1)
    (vis / "tidy_vision.py").write_text(src)
    env = base_env(); env.update(VS_SEED=str(int(seed)), VS_OUT=str(rd), VS_TMAX=str(float(tmax)), VS_SAVEVIS="0")
    (rd / "local_run.json").write_text(json.dumps(dict(app=info["path"], scene=str(MAPS["tidy_vision"]), kind="tidy_vision", seed=seed, tmax=tmax), indent=1))
    cmd = [sys.executable, str(vis / "tidy_vision.py"), "--app", info["path"], "--scene", str(MAPS["tidy_vision"]),
           "--backend", "native", "--device", "cpu", "--headless", "--steps", "999999999"]
    log = open(rd / "log.txt", "w")
    proc = subprocess.Popen(cmd, cwd=str(rd), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (rd / "pid").write_text(str(proc.pid))
    return proc, rd


def strip_controller(app: Path, out: Path) -> Path:
    """A copy of the app without its prebuilt controller extensions, so the one built on this computer is used
    (what always happens on a Mac, whose apps carry a Linux controller). For testing that path on Linux."""
    import zipfile
    with zipfile.ZipFile(app) as zi, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
        for it in zi.infolist():
            data = zi.read(it.filename)
            if it.filename == "bundle.json":
                b = json.loads(data); b.setdefault("runtimes", {}).setdefault("mjlab", {})["extensions"] = {}; data = json.dumps(b, indent=1).encode()
            zo.writestr(it, data)
    return out


def start_official(app, scene=None, viewer=True):
    """A plain app (no vision brain): the official player with MuJoCo's viewer -- you drive with the keyboard/gamepad.
    On macOS MuJoCo's passive viewer must run under `mjpython`."""
    rd = new_run_dir("official")
    app = Path(app).expanduser().resolve()
    env = base_env()
    if viewer and sys.platform == "darwin":
        env.pop("MUJOCO_GL", None); env.pop("PYOPENGL_PLATFORM", None)
        py = shutil.which("mjpython", path=str(Path(sys.executable).parent)) or str(Path(sys.executable).parent / "mjpython")
    else:
        py = sys.executable
    cmd = [py, str(TOOLKIT / "scripts" / "play.py"), "--app", str(app), "--backend", "native", "--device", "cpu"]
    if scene: cmd += ["--scene", str(scene)]
    if not viewer: cmd += ["--headless", "--steps", "500"]
    (rd / "local_run.json").write_text(json.dumps(dict(app=str(app), scene=str(scene) if scene else None, kind="official", cmd=cmd), indent=1))
    log = open(rd / "log.txt", "w")
    proc = subprocess.Popen(cmd, cwd=str(TOOLKIT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (rd / "pid").write_text(str(proc.pid))
    return proc, rd


def save_mp4(run_dir, out=None, size="1080", folder=None, audio=None):
    """Re-render a recorded run as an MP4 with the demo overlay at 1x. Returns (Popen, out_path)."""
    rd = Path(run_dir)
    if out is None:
        folder = Path(folder).expanduser() if folder else DEFAULT_MOVIES
        folder.mkdir(parents=True, exist_ok=True)
        r = {}
        if (rd / "result.json").exists(): r = json.loads((rd / "result.json").read_text())
        tag = f"{r.get('n_gathered', '?')}of{r.get('n_toys', '?')}"
        if r.get("kind") == "soccer":
            sc = r.get("score") or {}; out = folder / f"jumper-soccer-{rd.name}-RED{sc.get('RED', 0)}-BLUE{sc.get('BLUE', 0)}-1x-{size}p.mp4"
        else:
            out = folder / (f"jumper-flybrain-{rd.name}-1x-{size}p.mp4" if r.get("kind") == "flybrain" else f"jumper-hide-seek-{rd.name}-{tag}-1x-{size}p.mp4")
    out = Path(out).expanduser()
    log = open(rd / f"render_{out.stem}.log", "w")
    cmd = [sys.executable, "-m", "jhs.render_run", str(rd), str(out), "--size", str(size)]
    if audio and audio.get("path"):
        # audio = {path, start, delay, volume, fade, loop}: mux your own audio (AAC 192k); video stays 1x and unchanged
        cmd += ["--audio", str(Path(audio["path"]).expanduser()), "--audio-start", str(float(audio.get("start") or 0)),
                "--audio-delay", str(float(audio.get("delay") or 0)), "--audio-volume", str(float(audio.get("volume", 1.0) if audio.get("volume") is not None else 1.0)),
                "--audio-fade", str(float(audio.get("fade", 3.0) if audio.get("fade") is not None else 3.0))]
        if audio.get("loop"): cmd.append("--audio-loop")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=base_env(),
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, out


def run_state(rd: Path) -> dict:
    """Status of a run directory, from its files."""
    rd = Path(rd); st = dict(run=rd.name, dir=str(rd))
    try: st.update(json.loads((rd / "live" / "status.json").read_text()))
    except Exception: pass
    if (rd / "result.json").exists():
        r = json.loads((rd / "result.json").read_text())
        st.update(finished=True, result={k: r.get(k) for k in ("n_found", "n_gathered", "n_toys", "t_all_gathered", "t_end", "falls", "coverage", "placed", "hidden_behind", "n_correct", "props_moved_m",
                                                "kind", "counts", "path_m", "time_s", "stopped", "toys_moved_m",
                                                "score", "goals", "own_goals", "drop_balls", "ball_idle_s", "final", "winner")})
    log = rd / "log.txt"
    if log.exists():
        txt = log.read_text(errors="replace")
        lines = [l for l in txt.splitlines() if not l.startswith("[play]   ")]
        st["log_tail"] = "\n".join(lines[-14:])
        if "Traceback" in txt and "result" not in st: st["error"] = True
        m = re.findall(r"\[run\] t=(\d+)s", txt)
        if m and "t" not in st: st["t"] = float(m[-1])
    try: kind = json.loads((rd / "local_run.json").read_text()).get("kind")
    except Exception: kind = None
    st["kind"] = kind
    need = ("qpos.npz", "vision_meta.json", "vision_tof.npz", "result.json") + (() if kind in ("flybrain", "soccer") else ("maps.npz",))
    st["can_save_mp4"] = all((rd / f).exists() for f in need)
    return st
