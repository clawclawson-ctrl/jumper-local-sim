"""What is inside an .app, and how this local sim can run it."""
from __future__ import annotations
import json, sysconfig, zipfile
from pathlib import Path


def inspect(path) -> dict:
    p = Path(path).expanduser().resolve()
    info = dict(path=str(p), name=p.name, ok=False)
    if not p.is_file() or not zipfile.is_zipfile(p):
        info["error"] = "not an .app (an .app is a zip with bundle.json inside)"; return info
    with zipfile.ZipFile(p) as z:
        names = set(z.namelist())
        if "bundle.json" not in names:
            info["error"] = "no bundle.json inside, so it is not a Jumper app"; return info
        b = json.loads(z.read("bundle.json"))
        notes = [n for n in b.get("notes", []) if isinstance(n, str)]
        title = next((n.split(":", 1)[1].strip() for n in notes if n.startswith("title:")), None)
        exts = sorted(((b.get("runtimes") or {}).get("mjlab") or {}).get("extensions") or {})
        brain = next((n for n in ("vision/hide_seek.py", "vision/tidy_vision.py") if n in names), None)
        src = z.read(brain).decode("utf-8", "replace") if brain else ""
    here = sysconfig.get_platform()
    own = here in exts or (here.startswith("macosx-") and "macosx-universal2" in exts)
    modes = sorted((b.get("modes") or {}).keys())
    kind = "hide_seek" if brain == "vision/hide_seek.py" else "tidy_vision" if brain else "flybrain" if "flybrain" in modes else "official"
    info.update(ok=True, kind=kind, title=title or b.get("name") or p.stem, brain=brain, modes=sorted((b.get("modes") or {}).keys()),
                controller_platforms=exts, this_platform=here,
                controller=("the app's own" if own else "built on this computer (mjrl_fsm)"),
                local_hooks=("VS_LIVE_HOOK" in src) if brain else (True if kind == "flybrain" else None),
                sim_only=any("SIM-ONLY" in n for n in notes))
    if kind == "hide_seek" and not info["local_hooks"]:
        info["warning"] = ("this hide & seek app's brain predates the local-sim hooks: it will run with random hidden toys, "
                           "no hand placement and no live picture (the MP4 can still be saved afterwards)")
    if kind == "flybrain":
        info["brain"] = "local sim: jhs/brains/flybrain_local.py + sim_only_flybrain.py"
        info["sim_only"] = True
        info["warning"] = ("flybrain: the official walking policy steered by hand-written fly-INSPIRED rules from the sim camera + dToF "
                           "(SIM-ONLY VISION CONCEPT, not a connectome). Not yet tested in full sim trials.")
    if kind == "tidy_vision":
        info["warning"] = "a tidy-up vision app: runs on the shipped tidy-up room with its own toy layout (no hand placement)"
    return info


def extract_vision(app: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(app) as z:
        for n in z.namelist():
            if n.startswith("vision/") and not n.endswith("/"): z.extract(n, dest)
    return dest / "vision"
