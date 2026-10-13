"""Film two crabs dancing in the moonlit ballroom.

    python -m jhs.ballroom.show --audio TRACK [--seconds N] [--out DIR]
The page starts this with the track, the length and whether to play the music aloud.
Frames go to a live picture and to an MP4 with the track already in time.
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np

from jhs import PKG, ROOT
from jhs.audio import ffmpeg_exe
from jhs.ballroom.choreo import at, clip_root, figure_at, gestures, lerp_angle, plan
from jhs.ballroom.clips import Library
from jhs.ballroom.gait import Steps, held_body
from jhs.ballroom.music import analyze
from jhs.ballroom.scene import build
from jhs.runner import DEFAULT_MOVIES


FPS = 20
W, H = 1280, 720
BLEND = 0.45
# gestures() says which crabs keep the front claws. The legs underneath still step.


def _mix(a, b, w):
    """Blend two clip samples. w = 1 means all b."""
    j = (1 - w) * a[0] + w * b[0]
    dxy = (1 - w) * a[1] + w * b[1]
    z = (1 - w) * a[2] + w * b[2]
    rpy = np.empty(3)
    rpy[0] = (1 - w) * a[3][0] + w * b[3][0]
    rpy[1] = (1 - w) * a[3][1] + w * b[3][1]
    rpy[2] = lerp_angle(float(a[3][2]), float(b[3][2]), w)
    return j, dxy, z, rpy


def _place(moon_steps, candle_steps, data, fig, t, beat, sa, sb):
    """Body from the clip, feet from the tripod. Candle is half a cycle out of step.

    A bow may lean. A traveling figure keeps the clip's claws and drops its sway,
    because that sway is what skates a foot that was just planted.
    """
    if not clip_root(fig.kind):
        sa, sb = held_body(sa), held_body(sb)
    mxy, myaw, cxy, cyaw = at(fig, min(t, fig.t1), beat)
    moon_g, candle_g = gestures(fig.kind)
    moon_steps.apply(data, mxy, myaw, sa, t, beat, moon_g, 0.0)
    candle_steps.apply(data, cxy, cyaw, sb, t, beat, candle_g, 0.5)
    return mxy, myaw, cxy, cyaw


# The free camera sits on the opposite side of `azimuth` from its look-at point.
# Keep that spot inside the columns, or the picture is a wall.
_ROOM_R = 2.25


def _keep_inside(az, el, dist, look):
    el = min(float(el), -0.20)
    dist = max(float(dist), 1.65)
    h = dist * math.cos(el)
    px = look[0] - h * math.cos(az)
    py = look[1] - h * math.sin(az)
    r = math.hypot(px, py)
    if r > _ROOM_R:
        dist *= _ROOM_R / r
    return az, el, dist


class _Cam:
    def __init__(self):
        import mujoco
        self.cam = mujoco.MjvCamera()
        # Azimuth -90 puts the camera due north of the pair: both crabs in profile,
        # the moon in the south windows behind them, neither one in the lens.
        self.az, self.el, self.dist = math.radians(-90), math.radians(-22), 3.05
        self.look = np.array([0.0, 0.0, 0.28])

    def shot(self, fig, u, mid, heading, t, beat, meter):
        w = u * u * (3 - 2 * u)
        look = np.array([mid[0], mid[1], 0.28])
        if fig.shot == "entrance":
            az = lerp_angle(math.radians(-90), math.radians(-70), w)
            el = math.radians(-22 + 4 * w)
            dist = 3.05 + (2.55 - 3.05) * w
        elif fig.shot == "close":
            az = math.radians(-80 + 24 * w)
            el = math.radians(-18)
            dist = 2.7 + (2.15 - 2.7) * w
        elif fig.shot == "orbit":
            az = math.radians(-90) + math.radians(150) * u
            el = math.radians(-20)
            dist = 2.7
        elif fig.shot == "crane":
            az = math.radians(-110 + 60 * w)
            el = math.radians(-42 + 14 * w)
            dist = 3.05
        elif fig.shot == "side":
            az = math.radians(-90) + math.radians(16) * math.sin(2 * math.pi * u)
            el = math.radians(-18)
            dist = 2.55
        else:
            az = math.radians(-78) + math.radians(20) * math.sin(2 * math.pi * u)
            el = math.radians(-18)
            dist = 2.55
        frac = (t / beat) - round(t / beat)
        pulse = math.exp(-frac * frac * 80)
        bar_pos = (t / beat) % meter
        down = min(bar_pos, meter - bar_pos) < 0.18
        dist -= pulse * (0.10 if down else 0.035)
        az, el, dist = _keep_inside(az, el, dist, look)
        alpha = 0.18 if fig.shot in ("entrance", "orbit", "close", "crane") else 0.10
        self.az = lerp_angle(self.az, az, alpha)
        self.el += (el - self.el) * alpha
        self.dist += (dist - self.dist) * alpha
        self.look = self.look + (look - self.look) * alpha
        self.az, self.el, self.dist = _keep_inside(self.az, self.el, self.dist, self.look)
        self.cam.azimuth = math.degrees(self.az)
        self.cam.elevation = math.degrees(self.el)
        self.cam.distance = float(self.dist)
        self.cam.lookat[:] = self.look
        return self.cam


_FONTS = None


def _fonts():
    global _FONTS
    if _FONTS is None:
        from PIL import ImageFont
        path = str(PKG / "fonts" / "DejaVuSans-Bold.ttf")
        _FONTS = tuple(ImageFont.truetype(path, s) for s in (28, 18, 16))
    return _FONTS


def _caption(frame, title, aside, clock, pulse, flash):
    from PIL import Image, ImageDraw
    title_f, aside_f, small_f = _fonts()
    im = Image.fromarray(frame).convert("RGBA")
    bar = Image.new("RGBA", (im.width, 86), (6, 8, 18, 150))
    im.paste(bar, (0, 0), bar)
    foot = Image.new("RGBA", (im.width, 48), (6, 8, 18, 140))
    im.paste(foot, (0, im.height - 48), foot)
    d = ImageDraw.Draw(im)
    gold = (255, 214, 130, 255) if flash < 0.45 else (255, 236, 190, 255)
    d.text((28, 14), title, font=title_f, fill=(236, 230, 214, 255))
    d.text((28, 50), aside, font=aside_f, fill=gold)
    d.text((28, im.height - 36), "Moon", font=aside_f, fill=(186, 206, 232, 255))
    d.text((92, im.height - 36), "·", font=aside_f, fill=(180, 170, 140, 255))
    d.text((114, im.height - 36), "Candle", font=aside_f, fill=(240, 176, 110, 255))
    right = pulse + "   " + clock
    tw = d.textlength(right, font=small_f)
    d.text((im.width - 28 - tw, im.height - 34), right, font=small_f, fill=(210, 206, 196, 255))
    return np.asarray(im.convert("RGB"))


def _clock(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _ffmpeg(w, h, fps, audio, out: Path):
    cmd = [ffmpeg_exe(), "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
           "-i", str(audio),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", default=os.environ.get("JHS_AUDIO"))
    ap.add_argument("--seconds", type=float, default=float(os.environ["JHS_SECONDS"]) if os.environ.get("JHS_SECONDS") else None)
    ap.add_argument("--out", default=os.environ.get("VS_OUT"))
    ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--film", default=os.environ.get("JHS_FILM"))
    a = ap.parse_args(argv)
    if not a.audio:
        raise SystemExit("no track — pass --audio or set JHS_AUDIO")
    outd = Path(a.out) if a.out else Path(os.environ.get("JHS_LIVE_DIR", "ballroom_run")).parent
    outd.mkdir(parents=True, exist_ok=True)
    live = Path(os.environ.get("JHS_LIVE_DIR", str(outd / "live")))
    live.mkdir(parents=True, exist_ok=True)
    stopf = Path(os.environ.get("JHS_STOP", str(outd / "STOP")))
    play = (not a.no_play) and os.environ.get("JHS_PLAY_AUDIO", "1") == "1" and sys.platform == "darwin"
    show_live = os.environ.get("JHS_LIVE", "1") == "1"

    print(f"[ballroom] listening to {Path(a.audio).name}", flush=True)
    info = analyze(a.audio)
    dur = float(info["duration"] if a.seconds is None else min(a.seconds, info["duration"]))
    info["phrases"] = [p for p in info["phrases"] if p["t0"] < dur - 0.2]
    if info["phrases"]:
        info["phrases"][-1]["t1"] = min(info["phrases"][-1]["t1"], dur)
    else:
        info["phrases"] = [dict(t0=0.0, t1=dur, energy=0.5)]
    (outd / "music.json").write_text(json.dumps({k: v for k, v in info.items() if k != "beats"}))
    print(f"[ballroom] {info['pulse']}", flush=True)

    print("[ballroom] learning the steps", flush=True)
    lib = Library()
    figs = plan(info, lib)
    (outd / "dance.json").write_text(json.dumps([
        dict(t0=f.t0, t1=f.t1, kind=f.kind, shot=f.shot, aside=f.aside) for f in figs]))
    print("[ballroom] opening the ballroom", flush=True)
    import mujoco
    model, data, moon, candle = build()
    moon_steps, candle_steps = Steps(model, data, moon, lib.names), Steps(model, data, candle, lib.names)
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0          # collision meshes stay off the picture; the drawn shells remain
    rend = mujoco.Renderer(model, H, W)
    cam = _Cam()
    _fonts()

    nframes = max(1, int(round(dur * a.fps)))
    folder = DEFAULT_MOVIES
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        folder = outd
    film = Path(a.film) if a.film else folder / f"moonlit-ballroom-{info['bpm']:.0f}bpm-{info['style']}-{outd.name}.mp4"
    film.parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg(W, H, a.fps, a.audio, film)
    player = None
    prev = None
    prev_fig = None
    beat = float(info["beat"])
    meter = int(info["meter"])
    stopped = False
    try:
        # One discarded frame pays for the shader compile. A short burst then decides whether the
        # picture can keep up with the music, so the song is only started when the two can stay together.
        mujoco.mj_kinematics(model, data)
        rend.update_scene(data, camera=cam.cam, scene_option=opt)
        rend.render()
        bench = time.time()
        for i in range(8):
            t = i / a.fps
            fig = figs[0]
            mxy, myaw, cxy, cyaw = _place(moon_steps, candle_steps, data, fig, t, beat,
                                          lib.sample(fig.clip_a, t, fig.rate),
                                          lib.sample(fig.clip_b, max(0.0, t - fig.delay), fig.rate))
            mujoco.mj_kinematics(model, data)
            rend.update_scene(data, camera=cam.cam, scene_option=opt)
            _caption(rend.render(), info["title"], fig.aside, _clock(t), info["pulse"], 0.2)
        pace = (8 / max(1e-3, time.time() - bench)) > a.fps * 1.08
        if pace and play:
            player = subprocess.Popen(["afplay", "-t", f"{dur:.2f}", str(a.audio)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[ballroom] the music is playing with the picture", flush=True)
        elif not pace:
            print("[ballroom] filming faster than the music can be played along; the saved film keeps the song in time", flush=True)
        print(f"[ballroom] filming {dur:.0f}s -> {film.name}", flush=True)
        moon_steps.reset()
        candle_steps.reset()
        t0 = time.time()
        for i in range(nframes):
            if stopf.exists():
                stopped = True
                print("[ballroom] stopped", flush=True)
                break
            t = i / a.fps
            fig = figure_at(figs, t)
            if prev_fig is not fig:
                if prev_fig is not None:
                    end = prev_fig.t1 - prev_fig.t0
                    prev = (lib.sample(prev_fig.clip_a, end, prev_fig.rate),
                            lib.sample(prev_fig.clip_b, max(0.0, end - prev_fig.delay), prev_fig.rate))
                prev_fig = fig
                print(f"[ballroom] {t:6.1f}s  {fig.aside}", flush=True)
            sa = lib.sample(fig.clip_a, t - fig.t0, fig.rate)
            sb = lib.sample(fig.clip_b, max(0.0, t - fig.t0 - fig.delay), fig.rate)
            if prev is not None and (t - fig.t0) < BLEND:
                w = (t - fig.t0) / BLEND
                sa, sb = _mix(prev[0], sa, w), _mix(prev[1], sb, w)
            mxy, myaw, cxy, cyaw = _place(moon_steps, candle_steps, data, fig, t, beat, sa, sb)
            mujoco.mj_kinematics(model, data)
            mid = (np.asarray(mxy) + np.asarray(cxy)) / 2
            heading = myaw
            u = (t - fig.t0) / max(1e-3, fig.t1 - fig.t0)
            shot = cam.shot(fig, u, mid, heading, t, beat, meter)
            rend.update_scene(data, camera=shot, scene_option=opt)
            rgb = rend.render()
            frac = (t / beat) - round(t / beat)
            frame = _caption(rgb, info["title"], fig.aside, _clock(t), info["pulse"], abs(frac))
            if ff.poll() is not None:
                err = ff.stderr.read().decode("utf-8", "replace")[-500:]
                raise RuntimeError(f"the film encoder stopped: {err}")
            ff.stdin.write(np.ascontiguousarray(frame).tobytes())
            now = time.time()
            if show_live and (i % 4 == 0):
                from PIL import Image
                import io
                buf = io.BytesIO()
                Image.fromarray(frame).save(buf, "JPEG", quality=80)
                jpg = live / "live.jpg"
                tmp = jpg.with_suffix(".jpg.tmp")
                tmp.write_bytes(buf.getvalue())
                os.replace(tmp, jpg)
            if i % 10 == 0:
                wall = now - t0
                _write_json(live / "status.json", dict(
                    t=round(t, 2), wall=round(wall, 1), rate=round((t / wall) if wall > 0.5 else 0, 3),
                    state=fig.aside, kind="ballroom", bpm=info["bpm"], meter=meter, title=info["title"],
                    pulse=info["pulse"]))
            if pace:
                ahead = t - (time.time() - t0)
                if ahead > 0.002:
                    time.sleep(min(ahead, 0.25))
        ff.stdin.close()
        ff.wait(timeout=120)
        if ff.returncode != 0:
            err = ff.stderr.read().decode("utf-8", "replace")[-500:]
            raise RuntimeError(err or "ffmpeg failed")
    finally:
        if player is not None and player.poll() is None:
            player.terminate()
        if ff.poll() is None:
            try:
                ff.stdin.close()
            except Exception:
                pass
            try:
                ff.wait(timeout=30)
            except Exception:
                ff.kill()
    t_end = (nframes / a.fps) if not stopped else t
    result = dict(kind="ballroom", title=info["title"], bpm=info["bpm"], meter=meter, style=info["style"],
                  pulse=info["pulse"], t_end=round(float(t_end), 2), phrases=len(figs),
                  mp4=str(film) if film.is_file() else None, stopped=stopped)
    (outd / "result.json").write_text(json.dumps(result))
    print(f"[ballroom] saved {film}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
