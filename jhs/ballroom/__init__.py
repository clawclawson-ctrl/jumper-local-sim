"""Two crabs, one moonlit ballroom, a track the user brings.

The first tune is Waltz of the Moonlit Ballroom. The dance is open-loop: the crabs'
own recorded bows and dances, placed on a floor pattern that keeps them about two
thirds of a metre apart. No claws are joined. SIM-ONLY.
"""
from pathlib import Path

BUILTIN = "builtin:ballroom"
DEFAULT_TUNE = Path.home() / "Downloads" / "Waltz of the Moonlit Ballroom - clawclawson.m4a"
# The waltz stays first. Later tunes follow it, in this order.
TUNE_FILES = (
    ("Waltz of the Moonlit Ballroom", DEFAULT_TUNE),
    ("Square Waltz", Path.home() / "Downloads" / "Square Waltz - clawclawson.m4a"),
    ("Square Waltz (Banjo)", Path.home() / "Downloads" / "Square Waltz (Banjo) - clawclawson.m4a"),
    ("Waltz of the Moonlit Ballroom (Piano)", Path.home() / "Downloads" / "Waltz of the Moonlit Ballroom (Piano) - clawclawson.m4a"),
    ("Flamenco Waltz", Path.home() / "Downloads" / "Flamenco Waltz - clawclawson.m4a"),
    ("Flamenco Waltz (Accordion)", Path.home() / "Downloads" / "Flamenco Waltz (Accordion) - clawclawson.m4a"),
    ("Cuban Waltz (Brass)", Path.home() / "Downloads" / "Cuban Waltz (Brass) - clawclawson.m4a"),
    ("Cuban Waltz (Drums)", Path.home() / "Downloads" / "Cuban Waltz (Drums) - clawclawson.m4a"),
)


def tunes() -> list[dict]:
    return [{"name": name, "path": str(path)} for name, path in TUNE_FILES if path.is_file()]


def entry() -> dict:
    tune = DEFAULT_TUNE if DEFAULT_TUNE.is_file() else None
    return dict(
        path=BUILTIN, name="Moonlit Ballroom", ok=True, kind="ballroom",
        title="Moonlit Ballroom", modes=["ballroom"],
        controller="choreographer (the crabs' own recorded dances, on a floor pattern)",
        sim_only=True, local_hooks=True, tune=str(tune) if tune else None,
        warning=("Two crabs dance to a track you choose. Waltz of the Moonlit Ballroom is the first tune. "
                 "After it come Square Waltz, Square Waltz (Banjo), Waltz of the Moonlit Ballroom (Piano), "
                 "Flamenco Waltz, Flamenco Waltz (Accordion), Cuban Waltz (Brass), and Cuban Waltz (Drums). "
                 "They pirouette and dance a few square-dance figures, and they do not hold claws. "
                 "The camera follows them and turns when the phrase turns."),
    )
