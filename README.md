# Jumper Local Sim — hide & seek, crab soccer, flybrain and more, on your Mac

> **SIM-ONLY VISION CONCEPT.** The crab's camera picture and dToF range readings come from the simulator, and the
> vision brains here are concepts. This is a simulation demo, not something a real Jumper can do today.
> flybrain uses **fly-inspired rules, not a connectome**.

This repo runs [KingKong Robotics' Jumper](https://github.com/KingKongRobotics/jumper) crab robot apps locally in
MuJoCo on a Mac. You can:

- pick an app (`.app` bundle) and a room,
- place toys, the crab's start point and heading, and **pushable obstacles**,
- watch the run live in your browser (served only on `127.0.0.1`),
- **save the run as a 1x-speed MP4** with the full overlay, and optionally add **your own audio track**.

Included apps (in `apps/`):

| app | what it does |
|---|---|
| `jumper_hide_seek_vision.app` | Hide & seek. The crab searches the room with its (simulated) camera and pushes every toy it finds onto the rug. |
| `jumper_soccer_vision.app` | Crab soccer 1v1. Two crabs (RED vs BLUE) each find the ball with their own simulated camera and push it into the other goal. Rounded, bouncy field walls; optional obstacles; the scorer celebrates, then a referee reset; choose *Goals to win* (1-10). |
| `flybrain.app` | The crab wanders on its own, steered by hand-written fly-inspired rules: looming escape, avoidance, fixate/approach, optomotor turns, exploration saccades. No task and no score. |
| `jumper_tidy_vision.app` | Experimental tidy-up vision brain in the tidy-up room. |
| `jumper-tidy-up.app` | The official tidy-up sample (policies byte-identical to the official bundle), with no vision brain. You drive it yourself. |

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

## Real vs sim-only

| real (from the official toolkit) | sim-only (this repo) |
|---|---|
| Jumper robot model, walking/gesture policies, `mjrl_fsm` controller, `.app` bundle format | camera pictures and dToF readings rendered by MuJoCo |
| MuJoCo physics of the crab | vision brains (hide & seek, tidy vision, flybrain rules) |
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
apps/                  the five .app bundles above
maps/                  hide & seek room v7, tidy-up room, soccer field (MuJoCo scene packages)
jhs/                   the local sim: setup/live page, runner, overlay, MP4 renderer + audio, pushable obstacles,
                       checks; brains/ = flybrain runner + fly-inspired rules; soccer/ = crab soccer
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
