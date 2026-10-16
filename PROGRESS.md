
## Crabster (branch crab-roadster) — chunk 1 (2026-10-16 PT)
- Renamed: display name "Crabster"; package now jhs/crabster (was roadster); artifacts in /workspace/crabster (/workspace/roadster symlinks to it); zip will be JumperLocalSim-crabster.zip.
- jhs/crabster/flight.py: N2 cold-gas tank (Isp 70 s, 0.2 kg), lift/roll/pitch/yaw nozzles, PD attitude + vz controller; unit tests pass (tank, rigid-body kickflip).
- Real-sim kickflip (jhs/crabster/probe_kickflip.py, walking policy + xfrc_applied): 4/4 SUCCESS (start-time seeds 3.0/3.37/3.71/4.13 s), apex ~0.91 m, touchdown vz -0.1..-0.3 m/s, ~0.07 kg gas.
- Key fixes: write sim.data.xfrc_applied (torch tensor) directly; lift line of action through whole-crab COM computed from xipos (subtree_com is stale in native backend); min-jerk flip profile; lift gated on uprightness.
- NEXT: run.py app + scene (arch, 5 cars, night stage, crowd), suction+arch climb, sprint, overlay/camera/setup panel, MP4, --check, zip.
