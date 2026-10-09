# Jumper Hide & Seek — Local Sim

**SIM-ONLY VISION CONCEPT.** The crab's camera picture and dToF range sensor come from the simulator. This is a
simulation demo, not something a real Jumper can do today. The "scorer" line you see is **sim truth**: the crab
never reads it. It is only there so you can see how well the run went.

This folder lets you run Jumper apps on your own Mac:

- **Hide & seek** (`jumper_hide_seek_vision.app`). You place the blue ball and the two yellow ducks anywhere in
  the room, pick where the crab starts and which way it faces, and press Start. The crab searches with its camera
  and pushes every toy it finds onto the rug. You can watch it live and save the run as an MP4.
- **flybrain** (`flybrain.app`). The crab wanders on its own, steered by hand-written **fly-inspired rules** (looming
  escape, avoidance, fixate/approach, optomotor turns, exploration saccades). Same room, same setup screen, live
  view and MP4. No task and no score. See section 7b. **Not yet tested in full sim trials.**
- **Plain official apps** (no vision brain). These open MuJoCo's 3D window and you drive the crab yourself.
- **Tidy-up vision** (`jumper_tidy_vision.app`). Experimental. It runs on the tidy-up room with its own toy layout.

Nothing is uploaded anywhere. The page you use is served by your own Mac at `http://127.0.0.1`, and no other
computer can reach it.

---

## 1. Install (once, about 5–10 minutes)

You need `uv` and Rust (`cargo`). You already have both. Open **Terminal** and run:

```bash
cd ~/JumperLocalSim          # the folder you unzipped
bash install.sh
```

What the installer does:

1. Installs its own Python 3.12 with `uv`. Your Homebrew Python is not used or changed.
2. Creates `.venv/` inside this folder and installs MuJoCo, PyTorch, ONNX Runtime and the official Jumper toolkit
   (shipped in `toolkit/`).
3. Builds the crab's controller (Rust, `toolkit/deploy/fsm`) for your Mac with `cargo`. If that build fails, it
   uses the copy in `prebuilt/`. That copy was built from the same source.
4. Runs a self-check.

It ends with:

```
...
ALL OK
Installed. Start with:  ./run.sh
```

Everything stays inside this folder. To uninstall, delete the folder. Saved MP4s are kept in
`~/Movies/Jumper Hide & Seek`.

## 2. Start

```bash
cd ~/JumperLocalSim
./run.sh                                        # opens the page in your browser
./run.sh apps/jumper_hide_seek_vision.app       # same, with this app already chosen
./run.sh --pick                                 # a Finder dialog asks for an .app first
```

Leave the Terminal window open while you use the page. To quit, press **Ctrl-C** in Terminal.

## 3. Load an app

At the top of the page, under **1 · App**:

- choose an app from the list (everything in `apps/` is listed), or
- click **Load an .app file…** and pick any `.app`. It is copied into `apps/`, or
- click **Choose with a file dialog…** to use a Finder dialog. The app stays where it is.

The line under the list shows what the app is and which controller it will use. Apps built for Linux carry a
Linux controller only, so on your Mac it says *"built on this computer (mjrl_fsm)"*. That is expected. Everything
else (walking policies, settings, vision brain) comes from the app itself.

## 4. Place the toys and the crab (hide & seek)

The square is the room seen from above: north at the top, 5.3 m across.

- **Grey** shapes are walls and props: the east and west stairs, the north and south planters, and the four crates.
- **Tan** is the rug, where the toys must be gathered.
- **Drag** the blue ball, yellow duck and yellow "duck 2" wherever you like.
- **Drag** the red crab to choose its start spot, and use the **Crab heading** slider to choose which way it faces
  (0° = east, 90° = north).
- **Random hidden placement** hides all three toys behind props, out of sight of the crab's start spot. This is
  the same placement our tests used. Change the seed for a different layout.

The page checks every spot as you drop it:

- **Red ring + red message**: not allowed. The toy is inside or touching a prop, too close to a wall, on top of
  another toy, or under the crab, or the crab is standing in a prop. **Start** stays disabled until every spot is
  fine.
- **Yellow message**: allowed, but worth knowing. For example, "in plain view from the crab's start" (then it's
  not much of a hide & seek), or "starts on the rug" (it already counts as gathered).

**Moving the props.** You can drag the stairs, planters and crates to new spots too, and double-click one to turn
it 15°. **Props back to their spots** undoes that. A prop is refused (red outline) if it is in a wall, on the rug,
on another prop or on the crab. Toys are then checked against the props where you put them, and **Random hidden
placement** hides the toys behind the props where they are now.

Options:

| option | what it does |
|---|---|
| Run length | sim-seconds of searching. After that, the crab gives up, bows and dances (up to 30 s more). 600 s is what our tests used. |
| Speed | "at most 1x real time" never runs faster than real life. "As fast as possible" removes that cap (only matters on a fast Mac). |
| Pushable obstacles | **on by default.** The stairs, planters and crates are loose objects the crab can shove when it walks or pushes into them. The boundary walls stay fixed. Off = the room exactly as in our tests (stairs and planters fixed in place, crates as the official map ships them). See "Pushable obstacles" below. |
| Physics | 1000 Hz is what our evaluations used. 200 Hz is faster and is the rate the walking policies were trained at, but our results were not measured at that rate. |
| live picture | turn off for a slightly faster run. You can still save the MP4 afterwards. |

Press **Start**.

## 5. Watch the run

Building the world takes about 20–40 s. After that the **Live** picture updates several times a second. It is the
demo layout:

- a follow camera of the crab;
- **top-left**: what the crab's camera sees, with the boxes its detector draws;
- the **dToF** range picture;
- **the crab's own map**: where it has looked, and where it believes the toys are;
- the **scorer** line, marked *SIM TRUTH, not used by the crab*;
- the sim clock and the SIM-ONLY label.

The status line shows the sim time and how fast the sim runs compared with real time.

**Stop** ends the run early. The recording up to that point is kept, and you can still save it as an MP4.

**Speed to expect:** on our Linux test machine, a single run goes at about **0.25x real time** at 1000 Hz physics.
A 600 s search therefore takes about 40 minutes there. Apple M-series cores are usually faster per core, so expect
roughly **0.3–0.6x real time** on your Mac mini. 200 Hz physics is faster again. We could not measure this on a Mac,
so the status line shows the real figure. The live picture runs at the sim's own pace. The saved MP4 always plays
at true 1x speed.

## 6. Save the run as an MP4

When a run ends (finished, time up, or stopped), a **Save this run as an MP4** box appears:

- **Folder**: `~/Movies/Jumper Hide & Seek` by default. Type another folder if you prefer.
- **Size**: 1920x1080, or 1280x720 (faster).
- Click **Save run as MP4**. A progress bar shows how far along it is. When it's done, **Show in Finder** opens the
  file.

The MP4 is re-rendered from the run's recording (the robot's motion, plus the crab's camera, detections, dToF and
map at every vision tick). It has the full overlay and plays at **1x real speed** from the sim clock, start to end
with no cuts. Rendering speed is about 6 frames per second at 1080p on our test machine (25 frames = 1 s of video),
so a 400 s run takes roughly 25 min. 720p is about twice as fast.

**Add your own audio (optional).** Open **Add your own audio** under the Save button:

- **Audio file**: use **Choose…** (the browser copies the file in) or **Finder dialog…** (it is used from where it is).
  It can be mp3, wav, m4a, aac, aiff or flac. Leave it empty and the MP4 is silent, as before.
- **Start the song at**: skip the first N seconds of the song.
- **Song begins … s into the video**: silence until then.
- **Volume** in %: 100 = unchanged.
- **Fade-out**: over the last N seconds of the video. Default 3 s; 0 = none.
- **If the song is shorter than the video**: *play once, then silence* (default) or *loop it*.

The song is always cut at the end of the video. The video picture is copied untouched: still 1x, same frames, same
length. Only an AAC 192 kbit/s audio track is added. This uses the ffmpeg that the installer bundles (no Homebrew
ffmpeg needed). If the audio can't be read, you still get the silent MP4 and the page says why.

You can also add audio to an MP4 you already saved:

```bash
./run.sh --add-audio ~/Movies/"Jumper Hide & Seek"/RUN.mp4 ~/Music/song.mp3            # writes RUN-with-audio.mp4
./run.sh --add-audio RUN.mp4 song.m4a OUT.mp4 --start 12 --delay 2 --volume 0.8 --fade 4 --loop
```

You don't need the live picture for this. Older runs are listed under **Previous runs**, each with its own **Save
as MP4** button. Run folders are in `runs/`.

## 6b. Pushable obstacles

With **Pushable obstacles** on, every stair step, planter and crate is a free body with this mass and friction:

| prop | size (m) | mass |
|---|---|---|
| east / west stairs | 0.30 x 1.13, 0.15 high | 1.8 kg each |
| north planter | 0.84 x 0.70, 0.44 high | 2.0 kg |
| south planter | 0.64 x 0.84, 0.32 high | 1.8 kg |
| NE / SE / NW / SW crates | about 0.26–0.30 wide | 1.6 / 1.8 / 1.8 / 1.4 kg |

- **Friction**: `0.7 0.03 0.003` (sliding / torsional / rolling), with condim 4. MuJoCo takes the larger value of the
  two surfaces, so sliding on the floor is 0.8, about the same as the toys (duck 0.8, ball 1.3). The torsional
  friction stops a nudged prop from spinning on the spot. Boxes can't roll.
- **Why these masses**: the crab weighs 2.54 kg. We tuned the masses so that a brush barely moves a prop, but the
  crab walking straight into one at its normal pushing speed shoves it along at a few cm/s.
- Real planters full of soil would be far heavier. Think of these as light plastic or foam props.

On the box, with the crab walking into the south planter for 10 s, the push speed depended on mass like this:

| mass | push speed |
|---|---|
| 1.0 kg | 9.1 cm/s |
| 1.5 kg | 5.8 cm/s |
| 2.0 kg | 1.9 cm/s |
| 4.0 kg | 0.25 cm/s |

The results of the final masses are in `PUSH_TESTS.md`.

What the crab knows about it: nothing changes for the crab. It never gets told where the props are. Its map of
obstacles comes only from its dToF, refreshed at every look. When a prop has moved, the dToF sees bare floor where
it used to be, those map cells clear, and the new spot fills in. The live view and the MP4 also draw thin **white
outlines** on the crab's map showing where every prop is right now. Those come from the simulator (SIM TRUTH, for
you to compare), not from the crab.

Our evaluation numbers (eval2, conf1, b3) were measured with fixed stairs and planters. Results with pushable
props may differ. Turn the option off to compare like for like.

## 7. Without the browser (headless) — run and record from Terminal

```bash
# random hidden toys (seed 31), whole run, save a 1080p MP4
./run.sh --headless --random 31 --mp4 ~/Movies/hide-seek-31.mp4

# your own layout from a file, crab starting at (0.5, -0.3) facing north, 300 s, 720p
./run.sh --headless --place my_layout.json --start 0.5,-0.3,90 --tmax 300 --mp4-folder ~/Movies --size 720
```

`my_layout.json` (metres; the rug centre is 0,0; x points east and y north; `start` is optional):

```json
{"ball": [1.25, -1.95], "duck": [-2.0, 0.45], "duck2": [0.1, 2.1], "start": [0, 0, 90]}
```

The same checks as on the page are applied. Other flags: `--fast` (200 Hz physics), `--uncapped`, `--no-live`,
`--fixed-obstacles` (the room as in our tests), `--move planterN=-1.0,1.6,30` (move a prop to x,y at a heading;
repeatable). The prop ids are `stairE stairW planterN planterS crateNE crateSE crateSW crateNW`. Props can also go in
the layout file as `"obstacles": {"planterN": [-1.0, 1.6, 30]}`. At the end, the headless run prints how far the crab
moved each prop.

To repeat our push test (the crab walks straight into the north planter for 10 s):

```bash
./run.sh --push-test --target planterN --start 0,0.8,90 --secs 10
```

Audio for the MP4 (all optional):

```bash
./run.sh --headless --random 31 --mp4 ~/Movies/hs31.mp4 \
         --audio ~/Music/song.mp3 --audio-start 10 --audio-delay 0 --audio-volume 0.8 --audio-fade 3 [--audio-loop]
```

Save an MP4 of an earlier run without running the sim again (with or without audio):

```bash
./run.sh --headless --render-only runs/<run folder> --mp4 ~/Movies/again.mp4 --size 720 [--audio song.mp3]
./run.sh --render runs/<run folder> [OUT.mp4] [--size 720] [--audio song.mp3 ...]     # same thing, fewer messages
```

## 7b. flybrain (fly-inspired rules) — SIM-ONLY VISION CONCEPT, not a connectome

*Added 2026-10-08. Untested in full sim trials: one 30 s smoke run on our Linux box only (see the end of this section).*

**What it is.** `flybrain.app` is the **official Jumper walking policy** (task `jumper.posture`, checkpoint `model_74800`,
Apache-2.0, KingKongRobotics/jumper; see `jhs/brains/NOTICE-flybrain`) under the mode name `flybrain`, plus a freeze
hold on LB / Q. That network only sees the joints and the IMU, and it walks where the gamepad tells it.
The "fly" part is `jhs/brains/sim_only_flybrain.py` (shipped in this folder, not in the .app). It reads the
**simulator's** camera and dToF and works the same virtual gamepad a person would, with hand-written rules,
highest priority first:

| state on screen | rule |
|---|---|
| **ESCAPE** / **FREEZE** | looming (something above the floor in front coming closer, short time-to-contact), something very close ahead, or a bump: freeze briefly (LB hold), then back up and turn away |
| **AVOID** | something close on one side (or very close ahead): turn away / back up |
| **FIXATE** / **APPROACH** | a dark vertical bar or a saturated colour blob: turn to centre it, walk toward it, stop at ~0.4 m |
| **OPTOMOTOR** | wide-field image motion that the crab's own turning does not explain: turn with it |
| **SACCADE** | a quick exploratory turn every few seconds |
| **WALK** | straight ahead between saccades |

**Real vs sim-only.** Real: the walking policy and the freeze hold (they are the official app's). Sim-only: the camera
picture, the dToF, and the rules. They are *inspired by* fly behaviours; it is **not** a FlyWire / Drosophila
connectome and not a trained network. A real Jumper cannot run this today.

**On the page.** Choose `flybrain.app [flybrain: fly-inspired rules, sim-only]` (or start with `./run.sh apps/flybrain.app`).
You get the same room setup as hide & seek: drag the toys (they are things it may fixate on), the crab and the props,
**Pushable obstacles** on/off, heading, physics. Run length defaults to 120 s. Press **Start**. The live picture shows:
the big current state, a camera inset with optic-flow arrows and the salient-blob box, the dToF inset with a red box
when something looms, behaviour counts, the crab's path on a minimap (white outlines = where the props are now, sim
truth), the sim clock and the banner *"SIM-ONLY VISION CONCEPT: fly-inspired rules, not a connectome"*. When it
ends (or you press Stop) it prints falls, distance walked and behaviour counts, and **Save run as MP4** (1x, optional
audio) works exactly as for hide & seek.

**Headless.**

```bash
./run.sh --headless --app apps/flybrain.app --random 31 --tmax 30 --mp4 ~/Movies/flybrain-test.mp4 --size 720
./run.sh --headless --app apps/flybrain.app --place layout.json --start 0,0,90 --tmax 120 --mp4-folder ~/Movies \
         --audio song.mp3                       # toys/props from a layout file, with your audio
```

Without `--place`/`--random` no toys are put in the room. All the hide & seek flags work (`--fixed-obstacles`,
`--move`, `--fast`, `--uncapped`, `--no-live`, `--size 1080|720|480`, the audio flags). Run files: `runs/<time>-flybrain/`
(`result.json`, `behaviours.jsonl` = one line per 0.1 s with state, pad command, flow, nearest ranges, loom, blob).

**Floor returns (fix1, 2026-10-08).** The dToF also sees the floor. Every return is now projected into the room with
the crab's own pose (the dToF's position and tilt) and the ray geometry, and anything less than 3 cm above the floor is
ignored (drawn grey in the dToF inset). Ranges are horizontal distances, so the crab tilting does not change the distance
to a wall. Looming is measured per world direction (2° bearings), not per pixel, so the crab's own pitching (for
example while it freezes) or turning cannot make a pixel jump from a far surface to a near one and look like an
approach. It needs at least 3 bearings of something above the floor closing in (closing speeds over 2 m/s are surface
swaps, not approaches). The rules' unit test (`./run.sh --check`) now also covers: an empty floor while pitching → WALK,
own pitch / own turn with a wall and a prop in view → no loom, a wall approached → ESCAPE, a prop close on one side →
AVOID.

**What we tested (2026-10-08, Linux box, using the same controller path as a Mac).** The rules' unit test, plus the same
30 s headless smoke run (random toys seed 31, pushable props, 720p MP4) before and after the fix. Every run started,
streamed live frames, finished and saved a 30.0 s 1x MP4. All had **0 falls**.

| 30 s smoke, seed 31 | ESCAPE/FREEZE | AVOID | FIXATE/APPROACH | WALK | SACCADE | OPTOMOTOR | escapes | avoids |
|---|---|---|---|---|---|---|---|---|
| before the fix | 14.3 s | 0 s | 5.3 s | 7.6 s | 2.2 s | 0.1 s | 9 | 0 |
| floor ignored, per-pixel looming | 8.0 s | 3.9 s | 8.3 s | 7.2 s | 2.0 s | 0.1 s | 5 | 5 |
| fix1 (floor) | 0 s | 18.2 s | 5.5 s | 3.9 s | 1.7 s | 0.2 s | 0 | 3 |
| **fix2 (floor + turning, shipped)** | **1.6 s** | **0.1 s** | 11.9 s | 12.8 s | 2.6 s | 0.5 s | 1 | 1 |

With fix1 there were no escapes in open space, but the crab sat in AVOID next to the east stairs for about 15 s
because of a second bug: it could not turn.

**Turning (fix2, 2026-10-08).** In this walking policy the right stick's x only turns the crab past half travel:
`ang_vel_z` uses |Rx| 0.5 → 1.0, linear from 0 to 4.0 rad/s. Below 0.5 the same stick twists the body
(`twist`, full at 0.5, fading to zero at 0.75). The rules' turn commands were mostly ≤ 0.5 (AVOID 0.50 = no turn at all,
OPTOMOTOR ≤ 0.40, approach steering ≤ 0.30, small FIXATE corrections), so those behaviours barely turned. Now every rule
turn t goes through `rx = sign(t) · (0.5 + 0.5 · min(1, |t| / 0.65))` (0 for |t| ≤ 0.02). The yaw rate is then
4.0 rad/s · min(1, |t|/0.65), for example AVOID 0.50 → rx 0.885 ≈ 3.1 rad/s. SACCADE (0.62) and ESCAPE (0.55) were
already past 0.5 and keep their strength (about 0.96 and 0.40 rad/s). The unit test asserts that every non-zero turn
lands past 0.5. Side effect: for rx between 0.5 and 0.75 (rule turns |t| < 0.33, such as approach steering and small
centring) the body-twist binding has not fully faded out yet, so those turns also twist the body a little.

fix2 smoke (same 30 s run): 0 falls, walked 7.0 m. AVOID fired once next to a prop and turned away at about 2.2 rad/s
right away; it left AVOID after 0.1 s. No long spinning: the longest turn in one direction was 1.5 s, during the one
escape. **Still there (not tuned, yours to judge):** a little dithering while fixating. In 26 of 295 ticks the turn
direction flipped. The worst case was a 1.4 s wobble of ±0.15 rad. That happened when the salient-blob picker jumped
between a weak bar on one side and a strong blob at the image edge, plus some overshoot on small corrections.

## 7c. Crab soccer 1v1 (`jumper_soccer_vision.app`) — SIM-ONLY VISION CONCEPT, experimental

Two crabs play 1v1 soccer on a walled **4.8 x 3.2 m field** with a goal at each end (mouth 0.8 m). **RED** defends the red goal
(left, -x) and attacks the blue one; **BLUE** the other way. Both crabs run the official walking app (the same controller and
policies as the hide & seek app). Each one is steered by its own soccer brain (`jhs/soccer/brain.py`) through a virtual gamepad,
and uses **only its own** simulated camera, dToF and pose. Neither brain is ever given the ball's or the other crab's position from
the simulator. The **referee** is the only code that uses sim truth. It detects goals (ball fully over the line), keeps the score,
alternates kickoffs, and drops the ball back into play if nobody has touched it for 20 s. It also counts falls.

**Setup (page):** choose `jumper_soccer_vision.app`, and the field appears from above.
- Drag the ball, the RED and BLUE crabs (and set the selected crab's heading with the slider).
- **Add obstacle** puts one of the official props on the field (2 planters, 2 stair steps, 4 crates). You can drag it, turn it
  45° or remove it.
- **Random obstacle layout** makes a mirror-symmetric layout from the seed, so neither team is favoured.
- The page keeps the goal mouths, the centre spot and both kickoff spots clear. Red = not allowed (Start stays off). Yellow = a
  warning (e.g. a crate close to a wall can trap the ball).
- **Pushable obstacles** works as in the room. Match length is 5 min by default (3, 2 and 10 are also offered). Live picture,
  Stop, and **Save run as MP4** (1x, full overlay, optional audio) work as for hide & seek.

**The overlay:**
- the scoreboard `RED 0 - 0 BLUE` with the match clock;
- both crabs' camera pictures with what their vision found (ball / opponent / goal boxes);
- small dToF pictures and each crab's state;
- a top view where rings and squares are **each crab's own belief** about the ball and the other crab, and the white dot is the
  real ball (sim truth, display only);
- the referee line marked **SIM TRUTH**, the sim clock `(1x speed)` and the SIM-ONLY VISION CONCEPT banner.

After a goal the referee **teleports** the crabs to their kickoff spots and puts the ball on the centre spot. This is shown as
*"referee reset"*, because a real referee would put things back by hand.

**How a crab plays:**
- **Search:** turn on the spot, then walk somewhere else to look.
- **Stage:** walk round the ball (never through it) to a spot behind it, as seen from the goal it attacks. If the ball is on a
  wall, the push line is turned off the wall.
- **Dribble:** push the ball toward the goal while keeping it centred.
- **Shoot:** a faster burst when it's close and lined up.
- **Defend:** go between the ball and its own goal when the other crab is clearly closer to the ball and coming at its goal.
- dToF obstacles, the walls (map knowledge) and the other crab are avoided. A crab that is stuck backs off.

How the vision tells things apart: the football is found with the hide & seek vision. The other crab is found by its team colour
and low height, and a goal by its colour and height (30 cm). A ball candidate next to a detected crab is dropped, so the other
crab's parts are not taken for the ball. Both crabs' light-grey parts are drawn dark so nothing on a crab looks like the white
ball.

**Goals, celebrations, winning:** when the ball fully crosses a goal line (referee, SIM TRUTH) the scoring crab celebrates with the
official app's own moves (a crab dance, ~3 s, pressed through its controller like the hide & seek finish); the other crab stands still.
Then comes **RETURNING TO KICKOFF**: the referee puts the ball on the centre spot ("referee: ball to centre"; if an obstacle
sits there, the nearest free spot), and each crab **walks back by itself**: its brain plans an A* path on its own occupancy (walls = map
knowledge, the centre circle around the ball blocked, obstacles it has seen with its dToF, the other crab where it last saw it), walks
to its kickoff spot (or the nearest free spot if an obstacle is on it), turns to face the opponent's goal and waits. The next kickoff
starts when both crabs are within 0.15 m and 15° of their spots; after 25 s the referee teleports only the late crab ("referee reset
(timeout)"). The team that conceded kicks off. The match clock pauses during the celebration and the walk back. **Goals to win** (setup page, default 3, 1-10; headless `--goals-to-win N`): the first crab to reach it wins,
does a longer celebration (bow + crab dance) and the overlay shows e.g. "RED WINS 3-1". The match length is a cap: when time runs out
the higher score wins ("... (time)") or it is a draw. The match clock is paused during celebrations and the walk back.

**Exploring and pushing (brain v2):** a crab that cannot find the ball after a full scan **EXPLOREs** like the hide & seek crab:
it keeps a searched-floor grid filled from its own camera view (blocked by obstacles it has sensed with its dToF; floor counts as
unseen again after 25 s because the ball moves), picks the next best viewpoint (most unseen floor, unseen floor behind obstacles
counts double, minus walking distance), walks there on an A* path and scans again; it goes back to ball play as soon as its camera
detects the ball. When **Pushable obstacles** is on, the brain is told so (prior knowledge: every obstacle on this field is an
official prop and all of them are pushable); obstacle positions still come only from its own dToF. In A*, cells of sensed obstacles
then cost extra instead of being blocked (walls and goals stay blocked), so if the way round is much longer or closed the crab walks
through and shoves the prop (**PUSHING OBSTACLE**). It also shoves a prop that sits on the spot behind the ball it needs, e.g.
when the ball is stuck against it. With fixed obstacles, sensed obstacles are blocked.

**Ball prediction + intercept:** each crab keeps a short track of its own ball detections (position + velocity). If it loses a
rolling ball it first walks to where the track says the ball rolled to (constant velocity with rolling deceleration, bouncing off the
walls and rounded corners; up to 4 s, confidence shown in the overlay sub-line), and only scans/explores if the ball is not there.
While staging it aims for where a rolling ball will be when it arrives (up to 2 s ahead). Turn off with `SOCCER_PREDICT=0`.

**Obstacles may go anywhere** inside the field: goal mouths, the centre spot, kickoff spots, against the walls. The only hard rule
is that an obstacle must be inside the boundary. Overlaps with a crab, the ball or another obstacle give a yellow WARNING but do not
block Start: the ball or crab then starts on the nearest free spot, and a pushable obstacle that overlaps another is lifted so it drops
on top. "Random obstacle layout" still keeps the goal mouths, the centre and the kickoff spots clear.

**Field:** built from the official room v7 package (`jhs/soccer/field.py` writes `maps/jumper-soccer-field.map`). It uses the same
floor (tinted turf green), wall material, football, planters, stairs and crates. The goals are simple coloured walls, and the
lines are non-colliding floor decals. The football's rolling friction is raised to 0.0025 so it slows down like a ball on turf
instead of rolling forever. The four corners are **rounded**: each is a quarter circle (radius 0.45 m) made of 8 short wall segments, so the ball and the crabs glide round them instead of getting stuck. The brains' wall model, the setup checks and both top views use the same rounded outline. The boundary walls and corners (not the goal nets) are
slightly **bouncy** (contact solref `-4000 -25`: light damping, wall contact parameters take priority; wall friction 0.3), and a
12 cm wide, 1.2 cm high **kick strip** (6° ramp) runs along the foot of every wall and corner. Bench check (MuJoCo only): a ball rolling
into a wall/corner at 0.8 m/s rebounds 0.45-0.54 m (was 0.07 m), hop < 7 mm; a ball left resting against a wall or in a corner rolls
back ~0.35 m into play. Override for experiments: `SC_WALL_SOLREF`, `SC_KICK_DEG` (0 = no strip), then `python -m jhs.soccer.field`.

**From Terminal:**
```bash
./run.sh --headless --app apps/jumper_soccer_vision.app --tmax 180 --seed 11                 # 3-minute match, prints the score
./run.sh --headless --app apps/jumper_soccer_vision.app --obstacles 2 --mp4 ~/Movies/soccer.mp4 --size 720
```

**Speed:** two crabs are heavy. On our Linux box a match runs at about 0.12x real time at 1000 Hz physics (0.10x with the live
picture), so a 3-minute match takes about 25 minutes. The Mac mini should be faster. 500 Hz physics is offered as a faster
option, but it was not tested.

SOCCER_RESULTS

## 8. Plain official apps and the tidy-up app

- **Plain apps** (any app without a vision brain): choose it, pick a room, and press **Start**. MuJoCo's 3D window
  opens and you drive the crab with the keys the Terminal prints (or a gamepad). Close the window to stop. On a Mac
  this window runs under `mjpython`, which the installer sets up.
- **jumper_tidy_vision.app**: experimental. It runs on the shipped tidy-up room with the app's own toy layout.
  There is no hand placement, no live picture and no MP4 for this one. Progress shows in the log under the Live box,
  and its results are written to its run folder.

## 9. Checking the install

```bash
./run.sh --check          # quick self-check (about 20 s): libraries, controller, off-screen rendering, ffmpeg + AAC, apps, flybrain rules, map
./run.sh --check --full   # also runs a 15 s hide & seek world end to end (a few minutes)
```

Both end with `ALL OK`.

A fuller test, about 15–25 minutes on a Mac mini. Each step should end as shown:

```bash
./run.sh --check --full                                                     # ... ALL OK
./run.sh --push-test --target planterN --start 0,0.8,90 --secs 10           # [push_test] {... "moved_m": ~0.2-0.4, "robot_falls": 0 ...}
./run.sh --headless --random 31 --tmax 60 --mp4 ~/Movies/jhs-test.mp4 --size 720 \
         --audio /System/Library/Sounds/Glass.aiff --audio-loop --audio-volume 0.5   # ... saved ~/Movies/jhs-test.mp4 (... s at 1x; audio added: Glass.aiff)
./run.sh --headless --app apps/flybrain.app --random 31 --tmax 30 --mp4 ~/Movies/flybrain-test.mp4 --size 720   # flybrain result ... falls 0 ...; saved ... (30.0 s at 1x; silent)
./run.sh                                                                    # the page opens; place toys, Start, Stop, Save run as MP4
```

## 10. Troubleshooting

| problem | fix |
|---|---|
| `install.sh` says uv not found | `curl -LsSf https://astral.sh/uv/install.sh \| sh`, then open a new Terminal |
| controller build fails (cargo) | the installer falls back to `prebuilt/` by itself. To force that: `JHS_USE_PREBUILT=1 bash install.sh` |
| "rustup: no default toolchain" | `rustup default stable`, then run the installer again (it tries this itself) |
| macOS says a file "can't be opened" / is from the internet | `xattr -dr com.apple.quarantine ~/JumperLocalSim`, then run the installer again |
| the page does not open | open the address that Terminal prints (e.g. `http://127.0.0.1:8765/`). If that port is busy, the next free one is used |
| no live picture, or the check fails at "off-screen render" | try `JHS_GL=glfw ./run.sh`. The default on a Mac is `cgl` |
| MuJoCo window does not open for a plain app | run `.venv/bin/mjpython toolkit/scripts/play.py --app apps/YOUR.app` in Terminal to see the error |
| "a run is already going" | press Stop, or wait. Only one sim runs at a time |
| the sim is slow | turn off the live picture, use 200 Hz physics, or a shorter run length. Close other heavy apps |
| MP4 render failed | see `runs/<run>/render_*.log` |
| "AUDIO NOT ADDED" | the audio file could not be read. Try converting it to .m4a or .wav (e.g. with the Music app). The silent MP4 is still saved |
| start over | delete `.venv` and run `bash install.sh` again |

Logs: `install.log` (installer), `runs/<run>/log.txt` (each run).

## What's inside

```
install.sh, run.sh          installer and launcher
apps/                       jumper_hide_seek_vision.app, flybrain.app, jumper_soccer_vision.app, jumper_tidy_vision.app (+ any you load)
maps/                       hide & seek room v7, soccer field, tidy-up room, room outline for the setup page
jhs/                        the local sim: setup/live page (server.py, web/), runner, live overlay, MP4 renderer + audio,
                            pushable obstacles (pushable.py), push test, checks,
                            brains/ = flybrain runner + the fly-inspired rules (SIM-ONLY), fly_overlay.py
                            soccer/ = crab soccer 1v1 (field, two brains, referee, overlay, MP4)
PUSH_TESTS.md, tests/push/  the push-test results from our Linux box
toolkit/                    the official Jumper toolkit (KingKongRobotics/jumper, commit 7d3cc4b), only the parts needed
prebuilt/                   fallback controller for macOS (universal2), built from toolkit/deploy/fsm
NOTICE-LOCAL.md             licences
```

The toolkit is Apache-2.0. See `toolkit/LICENSE`, `toolkit/NOTICE` and `toolkit/licenses/`. The DejaVu fonts are
covered by `jhs/fonts/LICENSE-DejaVu.txt`.
