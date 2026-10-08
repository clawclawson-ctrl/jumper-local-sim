"""Dump the room's collision boxes (walls, stairs, planters, crates), the rug and the toys' bodies from a map, for the
setup screen and the placement checks.   python -m jhs.dump_room APP MAP OUT.json"""
from __future__ import annotations
import importlib.util, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jhs import TOOLKIT  # noqa: E402
APP, MAP, OUT = sys.argv[1:4]
REPO = Path(os.environ.get("JUMPER_REPO", str(TOOLKIT))); sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("play", REPO / "scripts" / "play.py")
play = importlib.util.module_from_spec(spec); sys.modules["play"] = play; spec.loader.exec_module(play)
import controller  # noqa: E402
class NoPad:
    connected = False; name = "none"; path = "none"
    def state(self): return controller.GamepadState()
    def close(self): pass
controller.open = lambda *a, **k: NoPad()

def _loop(env, policy, viewer, max_steps, speed=1.0, readout=None, monitor=None):
    import mujoco, numpy as np
    sim = env.unwrapped.sim; model, live = sim.env_mjdata(0); mujoco.mj_forward(model, live)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
    def root(b):
        while model.body_parentid[b] != 0: b = model.body_parentid[b]
        return b
    toys = {}
    for k, suf in {"ball": "prop:ball", "duck": "prop:toy_duck", "duck2": "prop:toy_duck_2", "football": "prop:football", "minifb": "prop:mini_football"}.items():
        ids = [i for i, n in enumerate(names) if n.endswith(suf)]
        if ids: toys[k] = names[ids[0]]
    boxes = []
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0: continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX: continue
        rn = names[root(model.geom_bodyid[g])]
        if rn.startswith("robot") or rn in toys.values(): continue
        c = live.geom_xpos[g]; R = live.geom_xmat[g].reshape(3, 3); h = model.geom_size[g]
        top = float(c[2] + (np.abs(R) @ h)[2])
        boxes.append(dict(name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or rn, root=rn, c=[float(v) for v in c], R=[float(v) for v in R.ravel()],
                          h=[float(v) for v in h], top=round(top, 3)))
    cid = model.camera("robot/onboard").id
    out = dict(room=[-2.65, -2.65, 2.65, 2.65], rug=[0.0, 0.0, 0.87, 0.64], cam_z=float(live.cam_xpos[cid][2]), toys=toys, boxes=boxes,
               note="collision boxes from the map (sim geometry) -- used by the setup screen and the placement checks only, never by the crab")
    Path(OUT).write_text(json.dumps(out, indent=1)); print(f"[room] wrote {OUT}: {len(boxes)} boxes, toys {list(toys)}", flush=True)

play._loop = _loop
sys.argv = [str(REPO / "scripts" / "play.py"), "--app", APP, "--scene", MAP, "--backend", "native", "--device", "cpu", "--headless", "--steps", "1"]
play.main()
