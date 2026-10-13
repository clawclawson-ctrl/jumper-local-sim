"""The crabs' own recorded dances, sampled as joint angles and a small body sway.

Positions in the files are a few centimetres of sway around a stand. The floor
pattern is added later. Measured channels are the ones MuJoCo actually stood in.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

from jhs import TOOLKIT

_TASKS = {
    "bow": "gesture_bow/media/bow.npz",
    "hello": "gesture_hello/media/hello.npz",
    "salute": "gesture_salute/media/salute.npz",
    "paw": "gesture_paw/media/paw.npz",
    "maze": "dance_maze/media/maze.npz",
    "wings": "dance_dream_wings/media/dream_wings.npz",
    "waist": "dance_waist/media/waist.npz",
    "brazilian": "dance_brazilian/media/brazilian.npz",
}
LOOP = {"maze"}


class Library:
    def __init__(self):
        if str(TOOLKIT) not in sys.path:
            sys.path.insert(0, str(TOOLKIT))
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        from tasks.jumper.common.dance.motion import load_source
        from tasks.jumper.common.constants import HOME
        self.names = tuple(HOME)
        self.clips = {}
        root = TOOLKIT / "tasks" / "jumper"
        for name, rel in _TASKS.items():
            clip = load_source(root / rel)
            n = len(clip.joint_pos)
            self.clips[name] = dict(
                joints=np.asarray(clip.joint_pos, dtype=np.float32),
                pos=np.asarray(clip.root_pos, dtype=np.float32),
                rpy=np.asarray(clip.root_rpy, dtype=np.float32),
                dt=float(clip.dt),
                dur=float((n - 1) * clip.dt),
            )
            # Sway relative to the clip's own first frame, so a dance can be placed anywhere.
            self.clips[name]["pos"] = self.clips[name]["pos"].copy()
            self.clips[name]["pos"][:, :2] -= self.clips[name]["pos"][0, :2]

    def sample(self, name: str, local_t: float, rate: float = 1.0) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Joints (22,), sway xy, body z, roll-pitch-yaw. Past the end of a gesture, the last pose holds."""
        c = self.clips[name]
        tt = max(0.0, float(local_t) * float(rate))
        if name in LOOP:
            tt = tt % c["dur"] if c["dur"] > 1e-6 else 0.0
        else:
            tt = min(tt, c["dur"])
        f = tt / c["dt"]
        i = int(f)
        i = min(max(i, 0), len(c["joints"]) - 2)
        a = float(f - i)
        j = (1 - a) * c["joints"][i] + a * c["joints"][i + 1]
        p = (1 - a) * c["pos"][i] + a * c["pos"][i + 1]
        r = (1 - a) * c["rpy"][i] + a * c["rpy"][i + 1]
        return j, p[:2].astype(np.float64), float(p[2]), r.astype(np.float64)
