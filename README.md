# Jumper Local Sim — hide & seek, crab soccer, Crab Tanks, a moonlit ballroom and more, on your Mac

> **SIM-ONLY VISION CONCEPT.** The crab's camera picture and dToF range readings come from the simulator, and the
> vision brains here are concepts. This is a simulation demo, not something a real Jumper can do today.
> flybrain uses **fly-inspired rules, not a connectome**.

This repo runs [KingKong Robotics' Jumper](https://github.com/KingKongRobotics/jumper) crab robot apps locally in
MuJoCo on a Mac. You can:

- pick an app (`.app` bundle) and a room,
- place toys, the crab's start point and heading, and **pushable obstacles**,
- watch the run live in your browser (served only on `127.0.0.1`),
- **save the run as a 1x-speed MP4** with the full overlay, and optionally add **your own audio track**,
- pit two crabs against each other in **Crab Tanks**, an Atari Combat-style turret duel,
- or skip the toys and watch two crabs dance in a moonlit ballroom to a track you choose.

Included apps (in `apps/`):

| app | what it does |
|---|---|
| `jumper_hide_seek_vision.app` | Hide & seek. The crab searches the room with its (simulated) camera and pushes every toy it finds onto the rug. |
| `jumper_soccer_vision.app` | Crab soccer 1v1. Two crabs (RED vs BLUE) each find the ball with their own simulated camera and push it into the other goal. Rounded, bouncy field walls; optional obstacles; the scorer celebrates, then a referee reset; choose *Goals to win* (1-10). |
| `jumper_tanks_vision.app` | Crab Tanks 1v1, inspired by Atari's Combat (1977). Two crabs hunt each other with their own simulated cameras, aim independent turrets while walking, and fire slow, visible shots the other crab can see and dodge. Three Combat arenas; choose *Hits to win* (1-10). |
| `jumper_surround_vision.app` | Crab Surround 1v1, inspired by Atari's Surround (1977) / light cycles, CYBER themed. Each crab leaves a low glowing wall (magenta / cyan) along its real path; touch any wall or trail and the round is lost, the last crab standing scores. Choose *Rounds to win* (1-10). |
| `jumper_warlords_vision.app` | Crab Warlords, after Atari's Warlords (1981), CASTLE themed. Three crabs (RED, BLUE, GREEN) defend castles of breakable bricks around a king in a triangle arena and push one ball with their bodies; the ball on your king puts you out, the last crab standing takes the round. Choose *Rounds to win* (1-10) and three aggression sliders. |
| `flybrain.app` | The crab wanders on its own, steered by hand-written fly-inspired rules: looming escape, avoidance, fixate/approach, optomotor turns, exploration saccades. No task and no score. |
| `jumper_tidy_vision.app` | Experimental tidy-up vision brain in the tidy-up room. |
| `jumper-tidy-up.app` | The official tidy-up sample (policies byte-identical to the official bundle), with no vision brain. You drive it yourself. |
| Moonlit Ballroom | Built into the page, not an `.app`. Two crabs dance to a track you choose. See below. |

Full user guide: **[README-LOCAL.md](README-LOCAL.md)**.

---

## Quick start (macOS, Apple Silicon or Intel)

You need [`uv`](https://docs.astral.sh/uv/) and Rust (`cargo`), plus `git` (Git LFS is **not** needed).

```bash
git clone https://github.com/clawclawson-ctrl/jumper-local-sim.git
cd jumper-local-sim
bash install.sh        # once, about 5-10 min: own Python 3.12 + .venv, MuJoCo, PyTorch, ONNX Runtime, the controller; ends with "ALL OK"
./run.sh               # opens the setup page in your browser
```

Other commands:

```bash
./run.sh apps/flybrain.app        # open the page with this app already chosen
./run.sh --pick                   # pick an .app with a Finder dialog
./run.sh --check                  # quick self-test (~20 s); --check --full adds a short end-to-end run
./run.sh --headless --random 31 --tmax 60 --mp4 ~/Movies/test.mp4 --size 720    # no window: run and record
./run.sh --headless --app apps/flybrain.app --random 31 --tmax 30 --mp4 ~/Movies/fly.mp4 --size 720
./run.sh --headless --app apps/jumper_soccer_vision.app --tmax 180 --seed 31 --obstacles 3 --goals-to-win 3
./run.sh --headless --app apps/jumper_tanks_vision.app --tmax 120 --seed 11 --openings random --camera follow
./run.sh --add-audio VIDEO.mp4 SONG.mp3 [OUT.mp4] [--start S --delay S --volume V --fade S --loop]
```

Everything stays inside the folder (`.venv/`, `runs/`). To uninstall, delete the folder. Saved MP4s go to
`~/Movies/Jumper Hide & Seek` by default. If the controller build fails, the installer falls back to the universal2
copy in `prebuilt/` by itself.

## Using it

1. **Load an app.** Under **1 · App**, choose one from the list (everything in `apps/`), or load any `.app` file.
   Apps built on Linux carry only a Linux controller, so on a Mac the page shows *"built on this computer (mjrl_fsm)"*.
   That is expected.
2. **Place toys and the crab.** The room is shown from above. Drag the blue ball and the two yellow ducks wherever
   you like, drag the red crab to its start spot, and set its heading with the slider. Or press **Random hidden
   placement** (seeded) to hide the toys behind props. You can also drag or rotate the props. The page refuses
   spots that don't work (red) and warns about ones that do but are odd (yellow).
3. **Pushable obstacles** (on by default): the stairs, planters and crates are loose, light props the crab can shove
   aside. The boundary walls stay fixed. Turn it off for the fixed room. (Push-test results are in
   [PUSH_TESTS.md](PUSH_TESTS.md).)
4. **Start / Stop.** A live picture shows the run with an overlay: what the crab "sees", its state and a minimap.
5. **Save run as MP4.** The run is re-rendered at **1x speed** with the full overlay. You can attach your own audio
   file with start offset, delay, volume, fade and loop options. If the audio can't be read, the silent MP4 is still
   saved.

### flybrain notes (SIM-ONLY VISION CONCEPT — not a connectome)

- The policy inside `flybrain.app` is the **official Jumper locomotion policy, unmodified** (renamed). The behaviour
  comes from small hand-written rules in [`jhs/brains/sim_only_flybrain.py`](jhs/brains/sim_only_flybrain.py),
  which steer it through a virtual gamepad.
- The rules read the simulated camera (optic flow, salient blobs) and the dToF depth grid. They pick one of
  ESCAPE / AVOID / FIXATE / OPTOMOTOR / SACCADE / WALK.
- dToF floor returns are ignored, and looming is measured on horizontal range, binned by world bearing (fix1). Rule
  turns map onto the controller's yaw band (|rx| > 0.5), so small turns aren't lost in the stick's dead zone (fix2).
- The overlay shows a camera inset with flow arrows and the target box, a dToF inset (red = near, grey = floor,
  box = looming), the state label, behaviour counts, a minimap path and the banner *"SIM-ONLY VISION CONCEPT:
  fly-inspired rules, not a connectome"*.
- Built-in unit tests: `./run.sh --check` runs them.

### Crab soccer notes (SIM-ONLY VISION CONCEPT)

- Each crab runs the official walking policy and its own brain ([`jhs/soccer/`](jhs/soccer/)). It sees only its own simulated camera, dToF and pose, never the ball's or the opponent's true position.
- The brain finds the ball, gets behind it, dribbles and shoots toward the other goal, and defends when the opponent is closer to the ball.
- The **referee** uses sim truth: it detects goals, keeps score, and puts the ball back on the centre spot (or the nearest free spot) after each goal. It drops the ball if nobody touches it for 20 s.
- After a goal, the scorer celebrates, then **both crabs walk back to their kickoff spots by themselves** (using their own pose and map), face the other goal and wait. The match clock pauses during the celebration and the walk-back. If a crab isn't back within 25 s, the referee moves only that crab, labelled *referee reset (timeout)*.
- If a scan doesn't find the ball, a crab **explores**: it walks to the spot that reveals the most unseen floor, especially behind obstacles, then scans again. It switches back to ball play as soon as its camera sees the ball (overlay state *EXPLORE*).
- With *Pushable obstacles* on, the crabs know props can be shoved. Their route planner treats sensed props as costly instead of solid, so a crab **pushes through** when going around is much longer or blocked, or when a prop sits where it needs to stand behind the ball (overlay state *PUSHING OBSTACLE*). Walls and goals stay solid.
- The setup page lets you place the ball, both crabs and **obstacles anywhere inside the field**, including goal mouths, the centre and kickoff spots. Overlaps only show a yellow warning. You can also set *Goals to win* (default 3) and a time cap.
- Testing so far: a few short matches with real pushed goals and 0 falls. With the walk-back, two returns took about 15 s each, with no timeouts. It is still experimental: the brains are simple, with no passing. In v1.1.0 the crab part of the referee reset didn't actually move the crabs, so its test score isn't comparable. That is fixed in v1.1.1.

### Crab Tanks notes (SIM-ONLY VISION CONCEPT)

- Inspired by the tank games in Atari's **Combat** (1977). Choose an arena: **Open field**, **Simple maze** (default) or **Complex maze**, rebuilt from the original manual and ROM playfields at 7.0 x 4.5 m. Some passages are widened so the crabs' legs fit (every gap is at least 0.67 m).
- Each crab runs the official walking policy plus its own brain ([`jhs/tanks/`](jhs/tanks/)) and sees only its own simulated camera, dToF and pose. A sim add-on **turret** aims on its own within the camera's view while the crab walks.
- **Shots** are slow, glowing cyan balls (0.6 m/s) that both crabs can actually see. One shot of your own in flight at a time, a 2.5 s cooldown, optional wall bounces, and obstacles block shots, so they work as cover.
- The brain keeps a **map of where the opponent can't be** (floor it has seen empty is crossed off, and spreads back at walking speed), searches the likeliest spots, leads its shots, and takes cover. **Dodging comes first**: any seen shot, including a bounce shot, whose predicted path comes close overrides everything else, and the crab steps out of the way while keeping 0.30 m leg clearance from walls.
- Each round, every crab rolls its own **opening** (peek around the near or far end of its barrier, hold and ambush, or push to the middle), so matches don't repeat. **Openings** can be *Random* or *Locked* to one opener per crab.
- A hit stuns the target for 1.5 s and scores a point. The shooter dances, both crabs walk back behind their barriers, and the first to N hits wins. The referee and scoring use sim truth. The crabs never read it.
- The **camera follows the action** (zooming to keep both crabs and shots in frame), or set it to a fixed overview. MP4 export at 1x includes both crabs' cameras, the belief heatmap and the scoreboard.
- Testing: unit tests plus short headless matches on several seeds and on every arena, with no leg bumps, no falls and no hits from a shot the crab had already seen. It's still experimental: shots are fairly rare, and full first-to-3 matches have only been watched a few times.

### Crab Warlords notes (SIM-ONLY VISION CONCEPT)

- Arena: an equilateral triangle with 8 m sides (bouncy stone walls), built per run by `jhs/warlords/arena.py` from the room v7 floor. One castle per corner: 13 bricks in an arc around a 26 cm king block. Crabs start 2.0 m from the centre, in front of their castle.
- One ball (the official football, lower rolling friction). **Kickoff:** every round the ball starts AT REST on the centre spot (all three crabs start 2.0 m from it) and the crabs race for it; the referee never serves or kicks the ball. After a king falls (play on) it goes back to the centre at rest. Only a ball that has been still for 10 s with no crab touching it (stuck where no crab body + leg clearance fits, e.g. against a wall) is put back on the centre spot, at rest (counted as `recentres`). Crabs move it only by walking into it (real contacts, no fake abilities).
- Referee (sim truth): a brick the ball touches breaks (its collision is switched off and it disappears); the ball touching a king puts that crab out. The eliminated crab is parked outside the walls, and its castle stays as an obstacle. Last crab standing takes the round. At the 150 s round limit (`WL_ROUND_MAX`), the crab still in with the most bricks left takes the round; a tie is a draw. Between rounds the referee rebuilds the castles and re-places the crabs.
- Brain (`jhs/warlords/brain.py`, one per crab): it sees the ball and the other crabs only in its own camera, knows its own pose and the fixed map, and reads **brick counts and broken-brick spots off the public scoreboard** (a scoreboard read, labelled on the overlay; not vision). DEFEND: predicts the ball's track (rolling deceleration + bounces off the angled walls) and stands on its guard ring where the ball would reach the castle within 3.5 s (9 s bounce-prediction horizon; it hurries to the intercept spot). ATTACK: lines up behind the ball and pushes it at the weakest enemy castle, through a gap when there is one. HOLD otherwise. **Ball belief** (`jhs/warlords/belief.py`, like Crab Tanks' opponent belief, "knows where it isn't"): particles roll on from the last sighting with the same deceleration + angled-wall bounce model, floor its camera sees without the ball is cleared, and a little probability regrows over unseen floor; with the ball out of view it SEARCHes the belief peak instead of scanning (its heatmap shows on the minimap in its colour). The referee's kickoff call ("ball at rest on the centre spot") is public, like the scoreboard. Per-crab aggression sets how far from home it chases the ball. Also random kickoff openers (rush straight at the centre ball / arc left / arc right to strike from an angle / guard first; defined once and rotated 120 degrees for each crab, so all three are equidistant and fair), jitter, a spin watchdog and a final safety layer (it never walks into a wall or a castle; it slides along it).
- The setup page has Rounds to win, Aggression (RED / BLUE / GREEN), Openings (Random/Locked + seed) and Camera (Follow/Fixed), Ball speed and Paddle boost. Headless: `./run.sh --headless --app apps/jumper_warlords_vision.app --seed 5 --rounds-to-win 3 --warlords-aggression 60,60,60 [--camera fixed] [--ball-speed 1.6] [--no-paddle]`
- Fast ball (default): **Ball speed** 1.6 m/s (setting 0.3-3.0) now sets the castle-test shot speed; with the kickoff rule the ball's pace comes only from crab strikes; the outer walls are near-elastic (restitution ~0.975, MuJoCo solref -40000 -3) and bricks / kings stiff bouncy stone (~0.9, solref -40000 -12); the bounce prediction and the ball belief use the same per-surface values, rolling deceleration 0.008 m/s^2, top speed 3.2 m/s. The 1000 Hz physics moves a 3 m/s ball 3 mm per step and the stone squashes at most ~1.5 cm, so it cannot tunnel through a 10 cm brick (checked in the unit tests and a 3 m/s real-sim castle test).
- **Ball-proof walls (ported from the soccer field):** the three corners are rounded (r 0.45 m, inside the castles), and a soccer-style kick strip (20 cm, 6 deg ramp; soccer uses 12 cm) runs along every wall foot, including the wall/castle junctions (optional stone junction fillets, `WL_FIL_H`, are off by default: they shielded too many bricks), so a ball at rest against them rolls back into play. Brains stage behind the ball on a spot with leg clearance, banking it off the wall when needed (soccer's staging logic). The bounce prediction and ball belief use the same geometry.
- Match length up to 25 min, and a per-round time limit (1-25 min, default 2:30; then most bricks left wins the round): setup page, or headless `--tmax` / `--round-max`. Runs are recorded for **Save as MP4** (1x, full overlay, optional audio with trim / delay / volume / fade, as the other apps); headless `--mp4 OUT.mp4`.
- **Paddle boost (sim setting, default ON, `--no-paddle` / unticked turns it off):** a crab walking into the ball hits it like a heavy elastic paddle: the ball's speed along the contact normal becomes at least 2x the crab's real measured walking speed there (the exact elastic result for a paddle far heavier than the 90 g ball). The crabs' own speed is never faked; attackers line up and strike at stick 0.75 (~0.6 m/s).
- Three crabs are slow in the local sim (about 0.06x real time on the box).

### Crab Surround notes (SIM-ONLY VISION CONCEPT)

- Inspired by Atari's **Surround** (1977) and light cycles, on the Crab Tanks open field (7.0 x 4.5 m), with a dark neon-grid floor, violet rim walls and neon-tinted crabs.
- Each crab lays a **trail wall** (8 cm wide, 6 cm tall, a sim add-on drawn into every camera) along its actual curved path, 0.24 m behind its body in the direction of travel. Turning on the spot lays nothing. **A crab may walk over its OWN trail freely**; the referee (sim truth) ends a crab's round only when the **opponent's trail** comes within 0.24 m of its centre. Outer walls are solid but harmless and strategic: crabs may hug them and use them to pin the opponent (planner: only a small near-wall cost; touches counted as `wall_contacts` in result.json); crab-to-crab bumps are harmless. A final safety layer looks ~0.5 m ahead along every walking command: the opponent's trail hard-blocks (steer/stop), a wall only removes the into-wall part (slide along it). Goals are clamped 0.25 m inside. At the 90 s round limit (`SR_ROUND_MAX`) the referee awards the round to the crab owning clearly (>15%) more reachable floor (sim-truth territory); `SR_TIEBREAK=off` makes it a plain draw, and trails clear between rounds.
- Brain ([`jhs/surround/`](jhs/surround/)): its own trail comes from proprioception, and the opponent's trail from glowing pixels in its own camera plus the line between sightings. Every 0.4 s it scores 16 travel directions on a 0.25 m grid. It probes the real turn-limited arc (0.65 rad/s, so it never makes a hairpin) and needs room to keep turning. It then runs a **Voronoi race** against the opponent's predicted spot (line-fit velocity; when out of view, the Tanks "where it isn't" belief grid). It fills its own pocket once sealed off. **Trap escape:** every replan it flood-fills the floor it can still reach (walls, the opponent's trail and its predicted next 2 s; its own trail is passable). If that area falls under ~2.5 m², shrinks by 40% in 3 s, or the route ahead closes while there is room elsewhere, it switches to **ESCAPE**. It walks straight out toward the biggest open region, doubling back over its own trail if needed. That's legal, and as the crab walks omnidirectionally it backs out with no turn on the spot. It has a 4 s cool-down. **Aggression** (0-100 %, per crab, default 60) blends safe space-seeking with hunting: the crab heads for a spot ahead of the opponent's predicted path to wall it off, leaves its opener early and keeps its camera on the opponent while strafing. Crabs walk omnidirectionally at normal walking speed.
- A random per-round **opener** (straight, sweep left/right, cut to the middle) can be *Random* or *Locked* with a seed. Starts have a small jitter, and a 3 s spin watchdog runs. The follow camera has a Follow/Fixed toggle, like Tanks. The overlay uses compact insets (own cam with boxes, dToF depth, small top view) so the arena fills the frame.
- The setup page has Rounds to win, Aggression (RED and BLUE sliders), Openings (Random/Locked + seed) and Camera (Follow/Fixed). Headless: `./run.sh --headless --app apps/jumper_surround_vision.app --seed 11 --rounds-to-win 3 --surround-aggression 60 [--surround-aggression-red 20 --surround-aggression-blue 80] [--camera fixed] [--openings locked --opener-red CUT_MID]`

### Moonlit ballroom

Choose **Moonlit Ballroom** on the page. It is the first item in the list, and it is built into the page rather than shipped as an `.app`.

- **Moon** is silver and **Candle** is gold. They stay about 0.68 m apart and do not take claws. They bow, box-step, promenade, pirouette, and walk a few square-dance figures: a do-si-do, an allemande, a star, a sashay, a swing, a balance, and a courtesy turn. They face the way they are walking.
- Give them a track (mp3, m4a, wav, aac, flac or ogg). **Listen for the pulse** reads the tempo and the meter. **Start the dance** plays the song on this Mac while the picture runs, and saves an MP4 under `~/Movies/Jumper Hide & Seek/`. You can dance the whole song, the first 75 seconds, or a 30-second opening.
- The tune list starts with Waltz of the Moonlit Ballroom, and later tunes follow it. The tracks ship in `jhs/ballroom/tunes` under the same Apache 2.0 license as the rest of this repo. You can still give the page a track of your own.
- The feet step the way this robot walks, three at a time, and the knees stay bent the way it stands. A planted foot that the next step has left behind is picked up. A bow, a wave, or a spin keeps the front claws for the gesture, and those claws stay just above the floor. The camera stays in the room and follows the pair.
- This is not a vision run. The crabs are posed from their own recorded dances plus that walking step. Nothing is uploaded.

The longer guide is section 7d of [README-LOCAL.md](README-LOCAL.md).

### Crabster (SCRIPTED STUNT, SIM-ONLY THRUSTER + SUCTION CONCEPT)

Pick `jumper_crabster_vision.app`. A Jumper crab in a pearl-white Roadster skin (black glass canopy band, red accents, light bar, rear
spoiler) gets two **sim-only** add-ons: a suction fan and cold-gas (N2) thrusters. Both are real applied forces in MuJoCo, fed from a
finite 0.6 kg gas tank. On the night stage it climbs a concrete arch tail-first, hangs upside-down on suction, drops, half-rolls and
lands with a landing burn; then it kickflips over a row of five stylized model cars; or it does a 0-1.5 m/s sprint (sim numbers).
The official walking policy walks it on the ground; the stunt script fires the add-ons, and on the arch the legs hold a stance pose
while the boost nozzle drives and suction holds. Landings are physics: a failed stunt is shown and reported as failed.
Setup: stunt (Both / Ramp / Kickflip / Sprint), camera (Follow / Fixed), seed, Replay. Overlay: label, sim clock, gas, suction, g.
MP4 at 1x with a title card, optional audio like the other apps. Code: `jhs/crabster/`.

## Real vs sim-only

| real (from the official toolkit) | sim-only (this repo) |
|---|---|
| Jumper robot model, walking/gesture policies, `mjrl_fsm` controller, `.app` bundle format | camera pictures and dToF readings rendered by MuJoCo |
| MuJoCo physics of the crab | vision brains (hide & seek, tidy vision, flybrain rules) |
| | the moonlit ballroom (recorded dances and a walking step, not a vision brain) |
| | Crab Tanks turrets, shots, hit scoring and referee (sim truth; the crabs never read it) |
| | the "scorer" line and soccer referee (sim truth; the crabs never read it) |
| | pushable obstacles and room setup UI |

Nothing here has been run on a real Jumper. The apps' `bundle.json` files say *"not ready for the board"* because
they carry no `.rknn` models.

## Known limits

- Single-run demos. The vision brains were not tuned over many trials. flybrain is **not yet tested in full sim
  trials**. Its FIXATE state can dither between two targets (brief left/right wobble). The controller's twist band
  overlaps part of the yaw band (rx 0.5–0.75 also twists the body a little).
- Speed depends on your Mac. About 0.95x real time was measured on a Mac mini. Turn off the live picture or use
  200 Hz physics if it's slow.
- `jumper_tidy_vision.app` has no hand placement, live picture or MP4 yet.
- The live page is only served locally (`127.0.0.1`). Only one sim runs at a time.
- macOS was the target. The same code runs on Linux, which is where it was developed and the self-check is run.

## What's inside

```
install.sh, run.sh     installer and launcher
apps/                  the six .app bundles above
maps/                  hide & seek room v7, tidy-up room, soccer field, tanks arena (MuJoCo scene packages)
jhs/                   the local sim: setup/live page, runner, overlay, MP4 renderer + audio, pushable obstacles,
                       checks; brains/ = flybrain runner + fly-inspired rules; soccer/ = crab soccer; tanks/ = Crab Tanks;
                       ballroom/ = moonlit ballroom (two crabs, a track you choose, tripod feet, camera, MP4)
toolkit/               the needed parts of the official Jumper toolkit (KingKongRobotics/jumper @ 7d3cc4b), unchanged
prebuilt/              fallback macOS (universal2) controller, built from toolkit/deploy/fsm
tests/, PUSH_TESTS.md  push-test results
```

## Credits and licences

- **[KingKong Robotics — Jumper toolkit](https://github.com/KingKongRobotics/jumper)**, commit `7d3cc4b`, Apache-2.0:
  the robot, policies, controller, `play.py` and app format. Shipped in `toolkit/` with its own
  [LICENSE](toolkit/LICENSE), [NOTICE](toolkit/NOTICE) and third-party notices in `toolkit/licenses/`. All walking and
  gesture **policies inside the apps are KingKong Robotics' official policies**. `flybrain.app` is the official
  locomotion policy, renamed (see [`jhs/brains/NOTICE-flybrain`](jhs/brains/NOTICE-flybrain)).
- **Rooms** in `maps/` are assembled from official KingKong scene pieces
  ([jumper-design](https://github.com/KingKongRobotics/jumper-design)). Each map keeps its own README and provenance.
  This repo's licence grants no new rights to those scene assets. See [NOTICE-LOCAL.md](NOTICE-LOCAL.md).
- DejaVu fonts: [`jhs/fonts/LICENSE-DejaVu.txt`](jhs/fonts/LICENSE-DejaVu.txt).
- The local sim code in this repo (`jhs/`, `install.sh`, `run.sh`, the vision-brain concepts) is released under the
  Apache License 2.0. See [LICENSE](LICENSE).

This is an unofficial community project. It is not affiliated with or endorsed by KingKong Robotics.
