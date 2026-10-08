"""The room for the setup screen: collision boxes from the map, the rug, and the rules for where toys and the
crab may start. Sim geometry, used by the setup screen only -- never by the crab."""
from __future__ import annotations
import copy, importlib.util, json, math
from pathlib import Path
import numpy as np
from . import ROOT

TOYR = {"ball": 0.055, "duck": 0.045, "duck2": 0.045, "football": 0.065, "minifb": 0.02}
CRAB_R = 0.25          # the crab's footprint radius (m) for start-spot checks
TOY_GAP = 0.20         # min centre distance between two toys
NAMES = {"ball": "blue ball", "duck": "yellow duck", "duck2": "2nd yellow duck", "football": "football", "minifb": "mini football"}


class Room:
    def __init__(self, path: Path | None = None):
        d = json.loads(Path(path or ROOT / "maps" / "room_v7.json").read_text())
        self.data = d; self.room = d["room"]; self.rug = d["rug"]; self.cam_z = d["cam_z"]
        self.boxes = [(np.array(b["c"]), np.array(b["R"]).reshape(3, 3), np.array(b["h"]), b["name"]) for b in d["boxes"] if b["top"] >= 0.10]

    # ---- movable obstacles (setup screen: drag a stair/planter/crate somewhere else) ----------------------------
    def moved(self, moves: dict | None):
        """A copy of the room with obstacles moved: moves = {obstacle id: [x, y, yaw_deg]} (ids from jhs.pushable.OBST)."""
        if not moves: return self
        from .pushable import geom_to_id
        r = copy.copy(self); out = []
        for c, R, h, name in self.boxes:
            k = geom_to_id(name)
            if k in moves:
                m = moves[k]; c = c.copy(); c[0], c[1] = float(m[0]), float(m[1])
                if len(m) > 2:
                    a = math.radians(float(m[2])); R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
            out.append((c, R, h, name))
        r.boxes = out
        return r

    @staticmethod
    def _corners(box):
        c, R, h, _ = box
        ex, ey = R[:2, 0] * h[0], R[:2, 1] * h[1]
        return np.array([c[:2] + sx * ex + sy * ey for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])

    @staticmethod
    def _overlap(P, Q, margin=0.0):
        """2-D separating-axis test for two convex quads (corner arrays); True if they overlap (closer than margin)."""
        for poly in (P, Q):
            for i in range(4):
                e = poly[(i + 1) % 4] - poly[i]; n = np.array([-e[1], e[0]]); n /= (np.linalg.norm(n) + 1e-12)
                a, b = P @ n, Q @ n
                if a.max() + margin <= b.min() or b.max() + margin <= a.min(): return False
        return True

    def check_obstacles(self, start=None):
        """{obstacle id: reason} for obstacles in impossible places (in a wall, on another obstacle, on the rug, on the crab)."""
        from .pushable import geom_to_id
        errs = {}; obs = [(geom_to_id(b[3]), b) for b in self.obstacles()]
        x0, y0, x1, y1 = self.room
        rq = np.array([[self.rug[0] + sx * self.rug[2], self.rug[1] + sy * self.rug[3]] for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
        for i, (k, b) in enumerate(obs):
            if k is None: continue
            P = self._corners(b)
            if P[:, 0].min() < x0 + 0.02 or P[:, 0].max() > x1 - 0.02 or P[:, 1].min() < y0 + 0.02 or P[:, 1].max() > y1 - 0.02:
                errs[k] = "inside a wall"; continue
            if self._overlap(P, rq): errs[k] = "on the rug (the gathering area must stay clear)"; continue
            hit = next((pretty(b2[3]) for j, (k2, b2) in enumerate(obs) if j != i and self._overlap(P, self._corners(b2), 0.03)), None)
            if hit: errs[k] = f"overlaps {hit}"; continue
            if start is not None and self.footprint_dist(start[:2], b) < CRAB_R + 0.05: errs[k] = "on the crab's start spot"
        return errs

    def wall_cl(self, p):
        x0, y0, x1, y1 = self.room
        return min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1])

    def obstacles(self):
        return [b for b in self.boxes if not ("wall" in b[3] or "boundary" in b[3])]

    @staticmethod
    def footprint_dist(p, box):
        c, R, h, _ = box
        o = R.T @ np.array([p[0] - c[0], p[1] - c[1], 0.0]); q = np.maximum(np.abs(o[:2]) - h[:2], 0.0)
        return float(np.hypot(*q))

    def nearest(self, p):
        obs = self.obstacles()
        if not obs: return None, 9.0
        b = min(obs, key=lambda b: self.footprint_dist(p, b)); return b[3], self.footprint_dist(p, b)

    def check_toy(self, k, p, others=(), start=None):
        """'' if toy k may start at p, else the reason."""
        r = TOYR.get(k, 0.06); p = np.asarray(p, float)
        if self.wall_cl(p) < r + 0.05: return "too close to (or inside) a wall"
        name, dist = self.nearest(p)
        if dist < r + 0.03: return f"inside or touching {pretty(name)}"
        for k2, q in others:
            if k2 != k and np.hypot(*(p - np.asarray(q, float))) < TOY_GAP: return f"on top of the {NAMES.get(k2, k2)}"
        if start is not None and np.hypot(*(p - np.asarray(start[:2], float))) < CRAB_R + r + 0.05: return "under the crab's start spot"
        return ""

    def check_start(self, s, toys=()):
        p = np.asarray(s[:2], float)
        if self.wall_cl(p) < CRAB_R + 0.10: return "too close to a wall for the crab"
        name, dist = self.nearest(p)
        if dist < CRAB_R + 0.05: return f"the crab would stand in {pretty(name)}"
        for k, q in toys:
            if np.hypot(*(p - np.asarray(q, float))) < CRAB_R + TOYR.get(k, 0.06) + 0.05: return f"on top of the {NAMES.get(k, k)}"
        return ""

    def check_layout(self, toys: dict, start, moves: dict | None = None):
        """{'ok': bool, 'errors': {name: reason}, 'notes': {...}} for a whole layout (obstacles moved by `moves`)."""
        if moves: return self.moved(moves).check_layout(toys, start)
        errs, notes = dict(self.check_obstacles(start)), {}
        e = self.check_start(start, toys.items())
        if e: errs["crab"] = e
        for k, p in toys.items():
            e = self.check_toy(k, p, [(k2, q) for k2, q in toys.items() if k2 != k], start)
            if e: errs[k] = e
            elif abs(p[0] - self.rug[0]) < self.rug[2] and abs(p[1] - self.rug[1]) < self.rug[3]: notes[k] = "starts on the rug (already counts as gathered)"
        return dict(ok=not errs, errors=errs, notes=notes)

    def visible_from_start(self, placement_mod, toys: dict, start):
        return {k: bool(placement_mod.visible_from_start(np.asarray(p, float), k, self.boxes, tuple(start[:2]), self.cam_z)) for k, p in toys.items()}

    def to_json(self):
        return dict(room=self.room, rug=self.rug, toyr=TOYR, crab_r=CRAB_R, toy_gap=TOY_GAP, names=NAMES,
                    boxes=[dict(name=b[3], id=_gid(b[3]), label=pretty(b[3]), c=b[0][:2].tolist(), R=b[1][:2, :2].ravel().tolist(), h=b[2][:2].tolist(),
                                yaw=round(math.degrees(math.atan2(b[1][1, 0], b[1][0, 0])), 2),
                                wall=("wall" in b[3] or "boundary" in b[3])) for b in self.boxes])


def _gid(name):
    from .pushable import geom_to_id
    return geom_to_id(name)


def pretty(name):
    if not name: return "something"
    n = name.replace("scn_", "").replace("_geom", "").replace("prop:", "")
    for a, b in (("stair_8", "the east stairs"), ("stair_9", "the west stairs"), ("planterN_10", "the north planter"), ("planterW_11", "the south planter"),
                 ("crate_blue", "the NE crate"), ("crate_medium_nw", "the NW crate"), ("crate_medium", "the SE crate"), ("crate_small", "the SW crate"), ("wall", "a wall")):
        if a in n: return b
    return n


def load_placement(vision_dir: Path):
    """The app's own test-harness placement (vision/placement.py inside the .app)."""
    spec = importlib.util.spec_from_file_location("app_placement", Path(vision_dir) / "placement.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def random_hidden(room: Room, placement_mod, toys, start, seed: int, wall_cl=0.55):
    """The app's 'hidden at random' placement: behind the props, none visible from the crab's start."""
    placement_mod.WALL_CL = wall_cl
    rng = np.random.default_rng(int(seed) * 7919 + 1)
    try:
        placed, hid = placement_mod.hide_toys(room.boxes, rng, list(toys), start_xy=tuple(start[:2]), cam_z=room.cam_z)
    except SystemExit as e:
        return None, str(e)
    return {k: [round(float(v), 3) for v in p] for k, p in placed.items()}, {k: pretty(v) for k, v in hid.items()}
