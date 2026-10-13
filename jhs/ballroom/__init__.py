"""Two crabs, one moonlit ballroom, a track the user brings.

The first tune is Waltz of the Moonlit Ballroom. The dance is open-loop: the crabs'
own recorded bows and dances, placed on a floor pattern that keeps them about two
thirds of a metre apart. No claws are joined. SIM-ONLY.
"""
from pathlib import Path

BUILTIN = "builtin:ballroom"
DEFAULT_TUNE = Path.home() / "Downloads" / "Waltz of the Moonlit Ballroom - clawclawson.m4a"


def entry() -> dict:
    tune = DEFAULT_TUNE if DEFAULT_TUNE.is_file() else None
    return dict(
        path=BUILTIN, name="Moonlit Ballroom", ok=True, kind="ballroom",
        title="Moonlit Ballroom", modes=["ballroom"],
        controller="choreographer (the crabs' own recorded dances, on a floor pattern)",
        sim_only=True, local_hooks=True, tune=str(tune) if tune else None,
        warning=("Two crabs dance to a track you choose. Waltz of the Moonlit Ballroom is the first tune. "
                 "The camera follows them and turns when the phrase turns. They do not hold claws."),
    )
