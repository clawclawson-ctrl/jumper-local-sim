
## Crabster (branch crab-roadster) — chunk 1 (2026-10-16 PT)
- Renamed: display name "Crabster"; package now jhs/crabster (was roadster); artifacts in /workspace/crabster (/workspace/roadster symlinks to it); zip will be JumperLocalSim-crabster.zip.
- jhs/crabster/flight.py: N2 cold-gas tank (Isp 70 s, 0.2 kg), lift/roll/pitch/yaw nozzles, PD attitude + vz controller; unit tests pass (tank, rigid-body kickflip).
- Real-sim kickflip (jhs/crabster/probe_kickflip.py, walking policy + xfrc_applied): 4/4 SUCCESS (start-time seeds 3.0/3.37/3.71/4.13 s), apex ~0.91 m, touchdown vz -0.1..-0.3 m/s, ~0.07 kg gas.
- Key fixes: write sim.data.xfrc_applied (torch tensor) directly; lift line of action through whole-crab COM computed from xipos (subtree_com is stale in native backend); min-jerk flip profile; lift gated on uprightness.
- NEXT: run.py app + scene (arch, 5 cars, night stage, crowd), suction+arch climb, sprint, overlay/camera/setup panel, MP4, --check, zip.

## Crabster — chunk 2 (2026-10-16 PT)
- jhs/crabster/scene.py: night stage map (from the open-field map): concrete half-loop arch (R 1.0 m, top 2 m, 40 slabs, polished concrete mu 0.45), five stylized cars in a row at x=-0.85 (Model 3, Model Y, Semi centre, Cybercab, Cybertruck; blocky, no logos, collide), crowd silhouettes + barrier, 5 stage spot lights on masts, launch pad, sprint lane with ticks. Built to /workspace/crabster/stage.map.
- jhs/crabster/run.py: stunt runner (official walking policy via virtual pad + Show script; suction, lift, roll/pitch/yaw, boost nozzles as xfrc_applied from one tank, GAS0 now 0.60 kg); acts ramp / kickflip / both / sprint; honest pass/fail (upright, past cars, car-contact check); writes qpos.npz (+ thrust/suction/gas/g per frame for overlay/MP4), events.jsonl, result.json. jhs/crabster/trial.sh STUNT SEED.
- Kickflip act in the stage: seed 1 SUCCESS (cleared all five cars, landed at x -1.27).
- Ramp act: NOT working yet. The walking policy stalls on the arch at ~45-57 deg: head-first its claws jam into the concave wall (contact dump), tail-first it slips sideways and falls off at ~47 deg; gait paused + thrust can't break floor friction. Ideas next: start the climb from a low-friction ramp pad, lift-assist to unload legs, a gentler entry (clothoid), or a scripted "skate" with legs in a tuck pose.
- Not yet: sprint trial, Roadster skin + rear spoiler (render-time), overlay/camera/setup panel, MP4, --check hookup, zip.

## Crabster — chunk 3 (2026-10-16 PT)
- Climb solved (SCRIPTED STUNT): walk tail-first onto the arch; past 10 deg the legs hold a stance pose (skate), the boost nozzle cancels gravity's down-slope pull + sliding friction + speed loop along the loop tangent, suction gives the normal force (>= m g past 90 deg + 20 N), lateral centring nozzle, yaw trim. Arch: R 1.0 m, width 1.4 m, polished concrete mu 0.30 with priority (MuJoCo used the max foot friction before). Head-first jammed claws; world-up weight relief pushed it OFF the surface -- both abandoned.
- Hang 1.6 s inverted on suction; release -> 0.45 s half roll (+ computed twist; tail-first climb already lands facing the cars) -> landing burn that hovers until the twist is done.
- Results: ramp seeds 1,2,3 SUCCESS (8-11 deg off the cars); both seeds 1,4 SUCCESS (ramp + kickflip over all five cars); kickflip seed 1 SUCCESS; sprint seed 1 SUCCESS (0-1.5 m/s 0.73 s, 4 m 2.72 s, top 1.69 m/s, sim). Gas 0.60 kg tank, ~0.42 kg used for both.
- jhs/crabster/skin.py: Roadster skin (gloss red / black / chrome) + rear spoiler + thruster plume capsules (render only). CB_SHOTS env saves PNGs: /workspace/crabster/ramp_inverted.png, mid_kickflip.png.
- Next: night skybox (sky still renders blue), overlay, cinematic camera, setup panel + replay, MP4 export + audio, --check unit hookup, more seeds, zip.

## Crabster — chunk 4 (2026-10-16 PT)
- Skin now pearl WHITE (black glass canopy band, slim red shoulder accents, emissive front light bar, glowing nozzles that brighten + plume when firing, rear spoiler kept). Night skybox.
- jhs/crabster/view.py: Overlay (CRABSTER + SCRIPTED STUNT, SIM-ONLY THRUSTER + SUCTION CONCEPT, sim clock, gas bar, suction on/N, g-force, act/phase, stunt caller, results) + MP4 title card; Cam Follow (eased, slow orbit, pulls back on the arch) / Fixed (grandstand); Painter shared by live + MP4.
- App integration: apps/jumper_crabster_vision.app (vision/crabster.json), apps.inspect kind "crabster", runner.start_crabster (builds stage per run), server /api/start, setup panel (stunt picker, camera, speed, seed, live, Start + Replay), run list row, save_mp4 naming, render_run -> jhs/crabster/render.py (1x, optional audio via jhs.audio.mux), live.jpg/status.json + STOP + real-time cap. README section.
- ./run.sh --check ALL OK incl. Crabster unit tests (flight, stage builds/loads, overlay/camera).
- Runner run Both seed 1: ramp SUCCESS (10 deg off the cars) + kickflip SUCCESS. /workspace/crabster/show.mp4 1080p 25 fps 18.7 s with test audio; stills ramp_inverted.png, mid_kickflip.png, title.png.
- Known: Follow cam frames the kickflip apex too high (crab near the top edge under the overlay box) -- widen distance in flight next.

## Crabster — chunk 5 (2026-10-16 PT)
- Follow cam in flight: farther (3.6 m), lower look-at (0.75 z), lower elevation, faster spring -> the crab stays clear of the overlay through the kickflip.
- show.mp4 re-rendered silent (Both, seed 1, 1x, 1080p, 18.7 s); stills refreshed (title, ramp_inverted, mid_kickflip).
- Both success: seeds 1, 2, 3, 4, 5 -> 5/5 (ramp landings 6-16 deg off the cars; all five cars cleared each time). Ramp 3/3, kickflip 1/1, sprint 1/1.
