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
        soccer = "vision/soccer.json" in names
        tanks = "vision/tanks.json" in names
        surround = "vision/surround.json" in names
        warlords = "vision/warlords.json" in names
        src = z.read(brain).decode("utf-8", "replace") if brain else ""
    here = sysconfig.get_platform()
    own = here in exts or (here.startswith("macosx-") and "macosx-universal2" in exts)
    modes = sorted((b.get("modes") or {}).keys())
    kind = "warlords" if warlords else "surround" if surround else "tanks" if tanks else "soccer" if soccer else "hide_seek" if brain == "vision/hide_seek.py" else "tidy_vision" if brain else "flybrain" if "flybrain" in modes else "official"
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
    if kind == "soccer":
        info["brain"] = "local sim: jhs/soccer/brain.py (one per crab)"; info["sim_only"] = True; info["local_hooks"] = True
        info["warning"] = ("crab soccer 1v1: two crabs, each steered by its own soccer brain from its own sim camera + dToF + pose "
                           "(SIM-ONLY VISION CONCEPT). Experimental: only a few test matches so far.")
    if kind == "tanks":
        info["brain"] = "local sim: jhs/tanks/brain.py (one per crab)"; info["sim_only"] = True; info["local_hooks"] = True
        info["warning"] = ("crab tanks 1v1: two crabs with sim add-on turrets hunt each other with slow, visible shots; each brain uses only "
                           "its own sim camera + dToF + pose (SIM-ONLY VISION CONCEPT). Experimental: one short test match so far.")
    if kind == "warlords":
        info["brain"] = "local sim: jhs/warlords/brain.py (one per crab)"; info["sim_only"] = True; info["local_hooks"] = True
        info["warning"] = ("crab warlords (Atari Warlords): three crabs, one ball, castles of breakable bricks around a king; a ball on your king "
                           "puts you out; last crab standing takes the round. SIM-ONLY VISION CONCEPT: own camera + dToF + pose, brick counts = scoreboard read.")
    if kind == "surround":
        info["brain"] = "local sim: jhs/surround/brain.py (one per crab)"; info["sim_only"] = True; info["local_hooks"] = True
        info["warning"] = ("crab surround 1v1 (Atari Surround / light-cycle style): each crab lays a glowing trail wall; touch any wall or "
                           "trail and the round is lost. Each brain uses only its own sim camera + dToF + pose (SIM-ONLY VISION CONCEPT).")
    if kind == "tidy_vision":
        info["sim_only"] = True
        if info.get("local_hooks"):
            info["warning"] = ("tidy-up vision: the seed places the toys; the crab sorts them into the basket and the crate. "
                               "Live picture, and an MP4 afterwards.")
        else:
            info["warning"] = "this tidy-up app's brain has no live-picture hook, so the page can only show the log"
    return info


def extract_vision(app: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(app) as z:
        for n in z.namelist():
            if n.startswith("vision/") and not n.endswith("/"): z.extract(n, dest)
    return dest / "vision"
