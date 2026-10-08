"""Add your own music/audio to a saved run MP4. The video stream is copied untouched (still 1x, same frames);
only an AAC 192k audio track is added. Uses the ffmpeg bundled with imageio-ffmpeg (no system ffmpeg needed).
    python -m jhs.audio VIDEO.mp4 AUDIO.(mp3|wav|m4a|aac|...) [OUT.mp4] [--start S] [--delay S] [--volume V]
                        [--fade S] [--loop]
  --start S   skip the first S seconds of the audio file (start the song S seconds in)
  --delay S   let the audio begin S seconds after the video starts (silence before it)
  --volume V  loudness multiplier (1.0 = unchanged, 0.5 = half, 1.5 = louder)
  --fade S    fade the audio out over the last S seconds of the video (default 3; 0 = no fade)
  --loop      repeat the audio if it is shorter than the video (default: play once, then silence)
The audio is always cut to the video's length. SIM-ONLY VISION CONCEPT."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".aif", ".aiff", ".flac", ".ogg", ".opus", ".caf", ".mp4", ".mov")


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def video_seconds(video) -> float:
    import imageio_ffmpeg
    nframes, secs = imageio_ffmpeg.count_frames_and_secs(str(video))
    return float(secs)


def mux(video, audio, out=None, start=0.0, delay=0.0, volume=1.0, fade=3.0, loop=False, log=None) -> Path:
    """Write OUT (default: VIDEO with the audio, replacing it in place). Returns the output path."""
    video, audio = Path(video).expanduser(), Path(audio).expanduser()
    if not video.is_file(): raise FileNotFoundError(f"no video: {video}")
    if not audio.is_file(): raise FileNotFoundError(f"no audio file: {audio}")
    start, delay, volume, fade = max(0.0, float(start)), max(0.0, float(delay)), max(0.0, float(volume)), max(0.0, float(fade))
    D = video_seconds(video)
    if D <= 0: raise RuntimeError("could not read the video length")
    in_place = out is None or Path(out).expanduser().resolve() == video.resolve()
    final = video if in_place else Path(out).expanduser()
    tmp = final.with_name(final.stem + ".with-audio.part.mp4")
    af = []
    if delay > 0: af.append(f"adelay=delays={int(round(delay * 1000))}:all=1")
    af.append(f"volume={volume:.3f}")
    af.append("apad")                                   # silence after a short song, so the track spans the whole video
    if fade > 0: af.append(f"afade=t=out:st={max(0.0, D - min(fade, D)):.3f}:d={min(fade, D):.3f}")
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video)]
    if loop: cmd += ["-stream_loop", "-1"]
    if start > 0: cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-af", ",".join(af), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-t", f"{D:.3f}", "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if log: Path(log).write_text(" ".join(cmd) + "\n" + r.stdout + r.stderr)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg could not add the audio: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1])
    os.replace(tmp, final)
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video"); ap.add_argument("audio"); ap.add_argument("out", nargs="?")
    ap.add_argument("--start", type=float, default=0.0); ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--volume", type=float, default=1.0); ap.add_argument("--fade", type=float, default=3.0)
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    out = a.out or str(Path(a.video).with_name(Path(a.video).stem + "-with-audio.mp4"))
    try:
        p = mux(a.video, a.audio, out, a.start, a.delay, a.volume, a.fade, a.loop)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"error: {e}")
    print(f"saved {p}")


if __name__ == "__main__":
    main()
