"""Jumper Hide & Seek -- Local Sim: the setup/live page, served on this computer only (127.0.0.1).
    python -m jhs.server [APP] [--pick] [--port N] [--no-browser]
SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, signal, socket, subprocess, sys, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jhs import ROOT, PKG, VERSION  # noqa: E402
from jhs import runner  # noqa: E402
from jhs.apps import inspect, extract_vision  # noqa: E402
from jhs.room import Room, load_placement, random_hidden  # noqa: E402
from jhs.audio import AUDIO_EXT  # noqa: E402

APPS = ROOT / "apps"
STATE = dict(extra_apps=[], procs={}, renders={})     # run name -> Popen ; (run, out) -> Popen
LOCK = threading.Lock()
ROOM = Room()


def list_apps():
    paths = sorted(APPS.glob("*.app")) + [Path(p) for p in STATE["extra_apps"]]
    seen, out = set(), []
    for p in paths:
        rp = str(Path(p).resolve())
        if rp in seen: continue
        seen.add(rp); out.append(inspect(rp))
    return out


_pl_cache = {}
def placement_for(app):
    """The app's own vision/placement.py (falls back to the hide & seek app's)."""
    app = Path(app or APPS / "jumper_hide_seek_vision.app")
    key = hashlib.sha1(f"{app}:{app.stat().st_mtime}".encode()).hexdigest()[:12]
    if key not in _pl_cache:
        d = runner.RUNS / ".cache" / key
        vis = extract_vision(app, d)
        if not (vis / "placement.py").exists(): vis = extract_vision(APPS / "jumper_hide_seek_vision.app", runner.RUNS / ".cache" / "default")
        _pl_cache[key] = load_placement(vis)
    return _pl_cache[key]


def running_run():
    with LOCK:
        for name, p in list(STATE["procs"].items()):
            if p.poll() is None: return name
    return None


def run_dir(name):
    d = (runner.RUNS / name).resolve()
    if runner.RUNS.resolve() not in d.parents or not d.is_dir(): raise ValueError("no such run")
    return d


class H(BaseHTTPRequestHandler):
    server_version = "JumperLocalSim/" + VERSION
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        elif isinstance(body, str): body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (PKG / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/info":
                return self._send(200, dict(version=VERSION, platform=sys.platform, runs=str(runner.RUNS), movies=str(runner.DEFAULT_MOVIES),
                                            preselect=STATE.get("preselect"), running=running_run()))
            if u.path == "/api/apps": return self._send(200, list_apps())
            if u.path == "/api/room": return self._send(200, ROOM.to_json())
            if u.path == "/api/soccer_field":
                from .soccer import field as SF
                return self._send(200, dict(FX=SF.FX, FY=SF.FY, GW=SF.GW, GD=SF.GD, CENTRE_R=SF.CENTRE_R, BALL_R=SF.BALL_R, CORNER_R=SF.CORNER_R, kickoff=SF.KICKOFF,
                                            props={k: dict(label=v[1], hx=v[2], hy=v[3]) for k, v in SF.PROPS.items()}))
            if u.path == "/api/runs":
                rs = sorted([d for d in runner.RUNS.glob("2*") if d.is_dir()], reverse=True)[:30]
                return self._send(200, [dict(runner.run_state(d), running=(STATE["procs"].get(d.name) is not None and STATE["procs"][d.name].poll() is None)) for d in rs])
            if u.path == "/api/status":
                d = run_dir(q["run"]); st = runner.run_state(d); p = STATE["procs"].get(d.name)
                st["running"] = p is not None and p.poll() is None
                st["exit_code"] = None if p is None else p.poll()
                return self._send(200, st)
            if u.path == "/api/live.jpg":
                f = run_dir(q["run"]) / "live" / "live.jpg"
                return self._send(200, f.read_bytes(), "image/jpeg") if f.exists() else self._send(404, {"error": "no frame yet"})
            if u.path == "/api/render_status":
                out = Path(q["out"]); pf = Path(str(out) + ".progress.json")
                pr = json.loads(pf.read_text()) if pf.exists() else {}
                p = STATE["renders"].get(str(out)); pr["running"] = p is not None and p.poll() is None
                pr["exit_code"] = None if p is None else p.poll()
                if pr["exit_code"] not in (None, 0): pr["error"] = "render failed"
                return self._send(200, pr)
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/upload":
                name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(self.headers.get("X-Filename", "uploaded.app")).name) or "uploaded.app"
                if not name.endswith(".app"): name += ".app"
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > 2_000_000_000: return self._send(400, {"error": "empty or too large"})
                APPS.mkdir(exist_ok=True); dest = APPS / name
                with open(dest, "wb") as f:
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk: break
                        f.write(chunk); left -= len(chunk)
                info = inspect(dest)
                if not info.get("ok"): dest.unlink(missing_ok=True)
                return self._send(200, info)
            if u.path == "/api/upload_audio":
                name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(self.headers.get("X-Filename", "audio.mp3")).name) or "audio.mp3"
                if Path(name).suffix.lower() not in AUDIO_EXT: return self._send(400, {"ok": False, "error": f"not an audio file I know ({', '.join(AUDIO_EXT)})"})
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > 1_000_000_000: return self._send(400, {"ok": False, "error": "empty or too large"})
                AUD = runner.RUNS / ".audio"; AUD.mkdir(parents=True, exist_ok=True); dest = AUD / name
                with open(dest, "wb") as f:
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk: break
                        f.write(chunk); left -= len(chunk)
                return self._send(200, {"ok": True, "path": str(dest), "name": name})
            b = self._json()
            if u.path == "/api/pick_audio":
                p = pick_file(audio=True)
                return self._send(200, {"ok": True, "path": p, "name": Path(p).name} if p else {"ok": False, "error": "no file chosen"})
            if u.path == "/api/pick":
                p = pick_file()
                if p:
                    STATE["extra_apps"].append(p); return self._send(200, inspect(p))
                return self._send(200, {"ok": False, "error": "no file chosen"})
            if u.path == "/api/check":
                toys = {k: v for k, v in b["toys"].items()}; start = b["start"]; room = ROOM.moved(b.get("moves"))
                r = room.check_layout(toys, start)
                try: r["visible"] = room.visible_from_start(placement_for(b.get("app")), toys, start)
                except Exception as e: r["visible_error"] = str(e)  # noqa: BLE001
                return self._send(200, r)
            if u.path == "/api/soccer_check":
                from .soccer import field as SF
                msgs = SF.validate(b.get("setup") or {})
                return self._send(200, dict(ok=not any(l == "error" for l, _ in msgs), messages=[dict(level=l, text=t) for l, t in msgs]))
            if u.path == "/api/soccer_random":
                from .soccer import field as SF
                return self._send(200, dict(ok=True, props=SF.random_layout(int(b.get("seed", 1)), b.get("n"))))
            if u.path == "/api/random":
                start = b.get("start") or [0, 0, 0]
                toys, hid = random_hidden(ROOM.moved(b.get("moves")), placement_for(b.get("app")), b.get("toyset", ["ball", "duck", "duck2"]), start, int(b.get("seed", 1)))
                if toys is None: return self._send(200, {"ok": False, "error": f"could not hide the toys from that start spot ({hid}); move the crab nearer the middle"})
                return self._send(200, {"ok": True, "toys": toys, "hidden_behind": hid})
            if u.path == "/api/start":
                if running_run(): return self._send(409, {"error": f"a run is already going ({running_run()}); stop it first"})
                info = inspect(b["app"])
                if info.get("kind") == "hide_seek":
                    toys = b.get("toys"); start = b.get("start")
                    moves = b.get("moves") or None
                    if toys or moves:
                        chk = ROOM.check_layout(toys or {}, start or [0, 0, 0], moves)
                        if not chk["ok"]: return self._send(400, {"error": "invalid layout", **chk})
                    proc, rd = runner.start_hide_seek(info["path"], toys=toys, start=start, seed=int(b.get("seed", 1)), tmax=float(b.get("tmax", 600)),
                                                      speed=float(b.get("speed", 1.0)), live=bool(b.get("live", True)), physics_hz=b.get("physics_hz"),
                                                      pushable=bool(b.get("pushable", True)), moves=moves)
                elif info.get("kind") == "flybrain":
                    toys = b.get("toys") or None; start = b.get("start"); moves = b.get("moves") or None
                    if toys or moves:
                        chk = ROOM.check_layout(toys or {}, start or [0, 0, 0], moves)
                        if not chk["ok"]: return self._send(400, {"error": "invalid layout", **chk})
                    proc, rd = runner.start_flybrain(info["path"], toys=toys, start=start, seed=int(b.get("seed", 1)), tmax=float(b.get("tmax", 120)),
                                                     speed=float(b.get("speed", 1.0)), live=bool(b.get("live", True)), physics_hz=b.get("physics_hz"),
                                                     pushable=bool(b.get("pushable", True)), moves=moves)
                elif info.get("kind") == "soccer":
                    proc, rd = runner.start_soccer(info["path"], setup=b.get("setup"), seed=int(b.get("seed", 1)), tmax=float(b.get("tmax", 300)),
                                                   speed=float(b.get("speed", 1.0)), live=bool(b.get("live", True)), physics_hz=b.get("physics_hz"),
                                                   pushable=bool(b.get("pushable", True)), goals_to_win=int(b.get("goals_to_win", 3)))
                elif info.get("kind") == "tidy_vision":
                    proc, rd = runner.start_tidy(info["path"], seed=int(b.get("seed", 1)), tmax=float(b.get("tmax", 420)))
                else:
                    scene = {"hide_seek": runner.MAPS["hide_seek"], "tidy": runner.MAPS["tidy_vision"]}.get(b.get("map"), None)
                    proc, rd = runner.start_official(info["path"], scene=scene, viewer=True)
                with LOCK: STATE["procs"][rd.name] = proc
                return self._send(200, {"ok": True, "run": rd.name, "kind": info.get("kind")})
            if u.path == "/api/stop":
                d = run_dir(b["run"]); (d / "STOP").write_text("stop"); p = STATE["procs"].get(d.name)
                if p is not None and b.get("kill"):
                    try: os.killpg(p.pid, signal.SIGTERM)
                    except Exception: p.terminate()
                return self._send(200, {"ok": True})
            if u.path == "/api/save_mp4":
                d = run_dir(b["run"])
                if not runner.run_state(d)["can_save_mp4"]: return self._send(400, {"error": "this run has no recording to render (still running, or not a hide & seek / flybrain / soccer run)"})
                folder = b.get("folder") or str(runner.DEFAULT_MOVIES)
                audio = b.get("audio") or None
                if audio and audio.get("path"):
                    ap_ = Path(audio["path"]).expanduser()
                    if not ap_.is_file(): return self._send(400, {"error": f"audio file not found: {ap_}"})
                    if ap_.suffix.lower() not in AUDIO_EXT: return self._send(400, {"error": f"not an audio file I know ({', '.join(AUDIO_EXT)})"})
                else: audio = None
                proc, out = runner.save_mp4(d, folder=folder, size=str(b.get("size", "1080")), audio=audio)
                STATE["renders"][str(out)] = proc
                return self._send(200, {"ok": True, "out": str(out)})
            if u.path == "/api/reveal":
                p = Path(b["path"]).expanduser()
                if sys.platform == "darwin": subprocess.Popen(["open", "-R", str(p)])
                elif shutil.which("xdg-open"): subprocess.Popen(["xdg-open", str(p.parent if p.is_file() else p)])
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})


def pick_file(audio=False):
    """A native file dialog for choosing an .app or an audio file (macOS: AppleScript; Linux: zenity if present)."""
    what = "an audio file (mp3, wav, m4a, aac) to add to the MP4" if audio else "a Jumper .app file"
    if sys.platform == "darwin":
        script = f'POSIX path of (choose file with prompt "Choose {what}"' + (' of type {"public.audio"})' if audio else ")")
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        return r.stdout.strip() or None
    if shutil.which("zenity"):
        r = subprocess.run(["zenity", "--file-selection", f"--title=Choose {what}"], capture_output=True, text=True)
        return r.stdout.strip() or None
    return None


def free_port(start):
    for p in range(start, start + 50):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try: s.bind(("127.0.0.1", p)); return p
            except OSError: continue
    raise SystemExit("no free port")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("app", nargs="?"); ap.add_argument("--pick", action="store_true"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    app = a.app
    if a.pick and not app:
        app = pick_file()
        if not app: print("no file chosen; opening the page anyway")
    if app:
        info = inspect(app)
        if not info.get("ok"): print(f"warning: {app}: {info.get('error')}")
        else: STATE["extra_apps"].append(info["path"]); STATE["preselect"] = info["path"]
    port = free_port(a.port)
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{port}/"
    print(f"Jumper Hide & Seek -- Local Sim {VERSION}  (SIM-ONLY VISION CONCEPT)\n  open {url}  (only this computer can reach it)\n  Ctrl-C here to quit.", flush=True)
    if not a.no_browser: threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    def _quit(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _quit)
    try: signal.signal(signal.SIGINT, _quit)       # also when started in the background (SIGINT ignored there)
    except Exception: pass
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        for name, p in STATE["procs"].items():
            if p.poll() is None:
                (runner.RUNS / name / "STOP").write_text("stop")
        t0 = time.time()
        while any(p.poll() is None for p in STATE["procs"].values()) and time.time() - t0 < 15: time.sleep(0.3)
        for p in STATE["procs"].values():
            if p.poll() is None:
                try: os.killpg(p.pid, signal.SIGTERM)
                except Exception: pass
        print("\nbye")


if __name__ == "__main__":
    main()
