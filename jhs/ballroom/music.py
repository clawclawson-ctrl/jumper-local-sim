"""Tempo, meter and phrase energy of a track. Numpy and scipy only.

A waltz in the usual ballroom band (about 70–120 beats a minute) is allowed to win
a close call against 4/4. A march that is clearly in four still comes out in four.
"""
from __future__ import annotations
import subprocess, tempfile
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile

from jhs.audio import ffmpeg_exe


def _decode(path: Path) -> tuple[int, np.ndarray]:
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    try:
        r = subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "22050", str(tmp)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
            raise RuntimeError(r.stderr.strip() or "could not read that audio file")
        sr, x = wavfile.read(tmp)
        x = np.asarray(x, dtype=np.float64)
        peak = np.max(np.abs(x)) or 1.0
        return int(sr), x / peak
    finally:
        tmp.unlink(missing_ok=True)


def _flux(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    hop, nfft = 512, 2048
    _, _, z = signal.stft(x, sr, nperseg=nfft, noverlap=nfft - hop)
    mag = np.abs(z)
    diff = np.diff(mag, axis=1, prepend=mag[:, :1])
    flux = np.maximum(diff, 0.0).sum(axis=0)
    ker = np.hanning(9); ker /= ker.sum()
    flux = np.convolve(flux, ker, mode="same")
    return flux, sr / hop


def _tempo(flux: np.ndarray, fr: float) -> float:
    acf = np.correlate(flux - flux.mean(), flux - flux.mean(), mode="full")
    acf = acf[len(acf) // 2:]
    acf /= acf[0] or 1.0
    best = (0.0, 100.0)
    for bpm in np.arange(66.0, 210.0, 0.5):
        lag = 60.0 * fr / bpm
        i = int(round(lag))
        if not (2 < i < len(acf) - 2):
            continue
        score = acf[i - 1] * 0.2 + acf[i] * 0.6 + acf[i + 1] * 0.2
        for mul, w in ((2.0, 0.45), (0.5, 0.35), (3.0, 0.15), (4.0, 0.1)):
            j = int(round(lag * mul))
            if 2 < j < len(acf) - 2:
                score += w * acf[j]
        if score > best[0]:
            best = (score, float(bpm))
    return best[1]


def _meter(flux: np.ndarray, fr: float, bpm: float, duration: float) -> int:
    """How many beats in a bar. 3 and 4 are the ballroom choices; 2 is a fallback."""
    period = 60.0 / bpm
    scores = {}
    for m in (2, 3, 4):
        bar = period * m
        best = 0.0
        for phase in np.linspace(0.0, bar, 30, endpoint=False):
            ones, rest = [], []
            t = phase
            while t + period < duration:
                for k in range(m):
                    i = int((t + k * period) * fr)
                    if 0 <= i < len(flux):
                        (ones if k == 0 else rest).append(flux[i])
                t += bar
            if ones and rest:
                best = max(best, float(np.mean(ones) / (np.mean(rest) + 1e-9)))
        scores[m] = best
    winner = max(scores, key=scores.get)
    # A close call inside the slow-waltz band prefers three. A clear four still wins.
    if winner != 3 and 72.0 <= bpm <= 120.0 and scores[3] >= scores[winner] * 0.92:
        return 3
    return int(winner)


def _beats(flux: np.ndarray, fr: float, bpm: float, duration: float) -> list[float]:
    period = 60.0 / bpm
    # Phase: the grid whose beats land on the strongest onsets.
    best, phase = -1.0, 0.0
    for p in np.linspace(0.0, period, 48, endpoint=False):
        acc = 0.0
        t = float(p)
        n = 0
        while t < duration:
            i = int(t * fr)
            if 0 <= i < len(flux):
                acc += float(flux[max(0, i - 1):i + 2].max())
            t += period
            n += 1
        if n and acc / n > best:
            best, phase = acc / n, float(p)
    out = []
    t = phase
    window = min(0.08, period * 0.35)
    while t < duration - 0.05:
        i0 = int((t - window) * fr)
        i1 = int((t + window) * fr)
        i0, i1 = max(0, i0), min(len(flux) - 1, i1)
        if i1 > i0:
            snapped = (i0 + int(np.argmax(flux[i0:i1 + 1]))) / fr
        else:
            snapped = t
        out.append(round(float(snapped), 4))
        t += period
    return out


def _rms_env(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    win = int(sr * 0.25)
    if win < 1 or len(x) < win:
        return np.array([float(np.sqrt(np.mean(x ** 2)))]), len(x) / sr
    hop = win
    n = 1 + (len(x) - win) // hop
    env = np.empty(n)
    for i in range(n):
        seg = x[i * hop:i * hop + win]
        env[i] = np.sqrt(np.mean(seg ** 2))
    return env, hop / sr


def analyze(path) -> dict:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"no audio file: {path}")
    sr, x = _decode(path)
    duration = len(x) / sr
    flux, fr = _flux(x, sr)
    bpm = _tempo(flux, fr)
    meter = _meter(flux, fr, bpm, duration)
    beats = _beats(flux, fr, bpm, duration)
    env, env_dt = _rms_env(x, sr)
    beat = 60.0 / bpm
    bar = meter * beat
    phrase_len = 8 * bar
    if duration < phrase_len * 2.2:
        phrase_len = max(4 * bar, duration)
    phrases = []
    t = 0.0
    while t < duration - beat:
        t1 = min(duration, t + phrase_len)
        i0 = int(t / env_dt)
        i1 = max(i0 + 1, int(t1 / env_dt))
        energy = float(np.mean(env[i0:min(i1, len(env))])) if len(env) else 0.0
        phrases.append(dict(t0=round(t, 3), t1=round(t1, 3), energy=energy))
        t = t1
    if phrases:
        lo, hi = min(p["energy"] for p in phrases), max(p["energy"] for p in phrases)
        span = hi - lo or 1.0
        for p in phrases:
            p["energy"] = round((p["energy"] - lo) / span, 3)
    if meter == 3 and 70 <= bpm <= 126:
        style = "waltz"
    elif meter == 3:
        style = "waltz"
    elif meter == 4 and bpm >= 140:
        style = "quickstep"
    elif meter == 4 and bpm >= 100:
        style = "foxtrot"
    elif meter == 4:
        style = "march"
    else:
        style = "ballroom"
    title = path.stem.split(" - ")[0].strip() or path.stem
    return dict(
        ok=True, title=title, path=str(path), duration=round(duration, 3),
        bpm=round(bpm, 2), meter=meter, style=style, beat=round(beat, 4),
        beats=beats, phrases=phrases,
        pulse=_pulse_line(style, bpm, meter, duration),
    )


def _pulse_line(style, bpm, meter, duration) -> str:
    names = {3: "three", 4: "four", 2: "two"}
    clock = f"{int(duration // 60)}:{int(duration % 60):02d}"
    kind = {"waltz": "a waltz", "quickstep": "a quickstep", "foxtrot": "a foxtrot",
            "march": "a march", "ballroom": "ballroom time"}[style]
    return f"{kind} at {bpm:.0f} beats a minute, in {names.get(meter, meter)}, {clock}"
