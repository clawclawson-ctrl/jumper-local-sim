
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
