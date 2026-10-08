"""Run hide & seek (or flybrain) without any window and (optionally) save the MP4 with the full overlay.
    ./run.sh --headless [--app APP] (--place layout.json | --random SEED) [--start x,y,yaw] [--tmax S]
                        [--mp4 OUT.mp4 | --mp4-folder DIR] [--size 1080|720] [--fast] [--uncapped] [--no-live]
layout.json: {"ball": [x, y], "duck": [x, y], "duck2": [x, y]}  (metres; optional "start": [x, y, yaw_deg],
             optional "obstacles": {"planterN": [x, y, yaw_deg], ...}; ids: stairE stairW planterN planterS crateNE crateSE crateSW crateNW)
With --app apps/flybrain.app the fly-inspired rules drive the crab instead (no task, no score; default --tmax 120;
toys only if you give --place/--random). SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jhs import ROOT  # noqa: E402
from jhs import runner  # noqa: E402
from jhs.apps import inspect, extract_vision  # noqa: E402
from jhs.room import Room, load_placement, random_hidden  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", default=str(ROOT / "apps" / "jumper_hide_seek_vision.app"))
    g = ap.add_mutually_exclusive_group(); g.add_argument("--place"); g.add_argument("--random", type=int)
    ap.add_argument("--start", help="crab start x,y,yaw_deg (default 0,0,0 = rug centre facing east)")
    ap.add_argument("--tmax", type=float, default=None, help="sim seconds (default 600; flybrain 120)"); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mp4"); ap.add_argument("--mp4-folder"); ap.add_argument("--size", default="1080", choices=["1080", "720", "480"])
    ap.add_argument("--fast", action="store_true", help="200 Hz physics instead of 1000 Hz (faster; not what our evaluations used)")
    ap.add_argument("--uncapped", action="store_true", help="do not hold the sim to 1x real time")
    ap.add_argument("--no-live", action="store_true", help="skip the live picture (a bit faster)")
    ap.add_argument("--fixed-obstacles", action="store_true", help="stairs/planters fixed, crates as shipped (the map exactly as in our tests); "
                    "default: pushable obstacles")
    ap.add_argument("--move", action="append", default=[], metavar="ID=x,y[,yaw]", help="move an obstacle before the start, e.g. --move planterN=-1.0,1.6,30")
    ap.add_argument("--render-only", metavar="RUN_DIR", help="skip the sim: just save the MP4 of an earlier run")
    au = ap.add_argument_group("optional audio for the MP4 (video stays 1x and unchanged; AAC 192k)")
    au.add_argument("--audio", help="mp3/wav/m4a/aac/... file"); au.add_argument("--audio-start", type=float, default=0.0, help="skip the first S s of the song")
    au.add_argument("--audio-delay", type=float, default=0.0, help="song begins S s into the video")
    au.add_argument("--audio-volume", type=float, default=1.0, help="loudness multiplier (1.0 = unchanged)")
    au.add_argument("--audio-fade", type=float, default=3.0, help="fade-out over the last S s (0 = none)")
    au.add_argument("--audio-loop", action="store_true", help="loop the song if shorter than the video (default: once, then silence)")
    a = ap.parse_args()
    audio = None
    if a.audio:
        if not Path(a.audio).expanduser().is_file(): sys.exit(f"audio file not found: {a.audio}")
        audio = dict(path=a.audio, start=a.audio_start, delay=a.audio_delay, volume=a.audio_volume, fade=a.audio_fade, loop=a.audio_loop)
    if a.render_only:
        rd = Path(a.render_only).expanduser().resolve()
        if not (a.mp4 or a.mp4_folder): a.mp4_folder = str(runner.DEFAULT_MOVIES)
        return render(rd, a, audio)
    info = inspect(a.app)
    if not info.get("ok"): sys.exit(f"{a.app}: {info.get('error')}")
    if info["kind"] not in ("hide_seek", "flybrain"): sys.exit("--headless is for hide & seek and flybrain apps; use ./run.sh for the others")
    FLY = info["kind"] == "flybrain"
    if a.tmax is None: a.tmax = 120.0 if FLY else 600.0
    start = [0.0, 0.0, 0.0]
    toys = None; moves = {}
    lay = json.loads(Path(a.place).expanduser().read_text()) if a.place else {}
    moves.update({k: [float(x) for x in v] for k, v in (lay.pop("obstacles", None) or {}).items()})
    for mv in a.move:
        k, v = mv.split("=", 1); moves[k.strip()] = [float(x) for x in v.split(",")]
    from jhs.pushable import OBST
    bad = [k for k in moves if k not in OBST]
    if bad: sys.exit(f"unknown obstacle id(s) {bad}; use {list(OBST)}")
    room = Room().moved(moves)
    if a.place:
        if "start" in lay: start = [float(v) for v in lay.pop("start")]
        toys = {k: [float(v[0]), float(v[1])] for k, v in lay.items()}
    if a.start: start = [float(v) for v in a.start.split(",")]
    if len(start) == 2: start.append(0.0)
    seed = a.seed if a.seed is not None else (a.random if a.random is not None else 1)
    if a.random is not None:
        tmp = runner.RUNS / ".cache" / "headless"; vis = extract_vision(Path(info["path"]), tmp)
        if not (vis / "placement.py").exists(): vis = extract_vision(ROOT / "apps" / "jumper_hide_seek_vision.app", runner.RUNS / ".cache" / "headless_default")
        pl = load_placement(vis)
        toys, hid = random_hidden(room, pl, ["ball", "duck", "duck2"], start, a.random)
        if toys is None: sys.exit(f"could not hide the toys from that start spot: {hid}")
        print("random hidden placement:", json.dumps(toys), "behind", json.dumps(hid))
    if toys and not info.get("local_hooks"):
        sys.exit("this app predates the local-sim hooks: it cannot take a hand-placed layout (leave out --place/--random)")
    if toys or moves:
        chk = room.check_layout(toys or {}, start)
        if not chk["ok"]:
            for k, e in chk["errors"].items(): print(f"  {k}: {e}")
            sys.exit("invalid layout")
        for k, n in chk["notes"].items(): print(f"  note: {k} {n}")
    starter = runner.start_flybrain if FLY else runner.start_hide_seek
    proc, rd = starter(info["path"], toys=toys, start=start if (toys or a.start) else None, seed=seed, tmax=a.tmax,
                                      speed=0.0 if a.uncapped else 1.0, live=not a.no_live, physics_hz=200 if a.fast else None,
                                      pushable=not a.fixed_obstacles, moves=moves or None)
    print(f"obstacles: {'fixed (as in our tests)' if a.fixed_obstacles else 'pushable'}" + (f", moved: {moves}" if moves else ""))
    print(f"run folder: {rd}\n(SIM-ONLY VISION CONCEPT) running up to {a.tmax:.0f} sim-seconds; Ctrl-C stops early and keeps the recording.", flush=True)
    last = 0
    try:
        while proc.poll() is None:
            time.sleep(2)
            st = runner.run_state(rd)
            if time.time() - last > 15 and st.get("t") is not None:
                last = time.time()
                if FLY: print(f"  sim t={float(st['t']):6.1f}s  rate={st.get('rate', 0) or 0:.2f}x  {st.get('state', '')} ({st.get('sub', '')})  {st.get('counts', '')}", flush=True)
                else: print(f"  sim t={float(st['t']):6.1f}s  rate={st.get('rate', 0) or 0:.2f}x  {st.get('state', '')}  scorer (SIM TRUTH) {st.get('n_truth', '?')}/{st.get('ntoys', '?')}", flush=True)
    except KeyboardInterrupt:
        print("stopping (saving the recording)...", flush=True); (rd / "STOP").write_text("stop")
        try: proc.wait(120)
        except KeyboardInterrupt: proc.kill()
    st = runner.run_state(rd)
    r = st.get("result")
    if not r:
        print(st.get("log_tail", "")); sys.exit(f"the run ended without a result (exit code {proc.returncode}); see {rd/'log.txt'}")
    if FLY:
        print(f"flybrain result (SIM-ONLY, fly-inspired rules): ended at {r['t_end']} s, falls {r['falls']}, walked {r['path_m']} m, "
              f"behaviours {json.dumps(r.get('counts'))}, time per behaviour (s) {json.dumps(r.get('time_s'))}")
    else: print(f"result (SIM TRUTH): found {r['n_found']}/{r['n_toys']}, on the rug {r['n_gathered']}/{r['n_toys']}"
          + (f", all on the rug at {r['t_all_gathered']} s" if r.get("t_all_gathered") is not None else "") + f", falls {r['falls']}, ended at {r['t_end']} s")
    pm = json.loads((rd / "result.json").read_text()).get("props_moved_m")
    if pm: print("obstacles moved by the crab (m, SIM TRUTH): " + ", ".join(f"{k} {v:.2f}" for k, v in pm.items()))
    if a.mp4 or a.mp4_folder:
        render(rd, a, audio)


def render(rd, a, audio):
    if not runner.run_state(rd)["can_save_mp4"]: sys.exit(f"{rd} has no complete recording to render")
    if True:
        rproc, out = runner.save_mp4(rd, out=a.mp4, size=a.size, folder=a.mp4_folder, audio=audio)
        print(f"rendering the MP4 at 1x -> {out}", flush=True)
        pf = Path(str(out) + ".progress.json"); last = 0
        while rproc.poll() is None:
            time.sleep(3)
            if pf.exists() and time.time() - last > 20:
                last = time.time(); p = json.loads(pf.read_text())
                if p.get("total"): print(f"  {p['done']}/{p['total']} frames" + (f", about {round((p.get('eta_s') or 0)/60)} min left" if p.get("eta_s") else ""), flush=True)
        if rproc.returncode != 0 or not out.exists():
            sys.exit(f"rendering failed; see {rd}/render_{out.stem}.log")
        p = json.loads(pf.read_text()) if pf.exists() else {}
        print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB, {p.get('seconds', '?')} s at 1x" + (f"; {p['audio']}" if p.get("audio") else "; silent") + ")")


if __name__ == "__main__":
    main()
