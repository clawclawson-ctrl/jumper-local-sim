# Push tests (box, Linux) -- SIM-ONLY VISION CONCEPT

The crab (2.54 kg) starts 0.35-0.45 m from an obstacle facing it, stands 2 s, walks straight into it for 10 s at the brain's
top push command (ly = -0.4), then stands 3 s. Obstacle pose from sim truth every 10 steps. 1000 Hz physics. `./run.sh --push-test ...`

| file | target | pushable | mass kg | moved in 10 s (m) | push speed (cm/s) | yaw change (deg) | max tilt (deg) | drift after push (mm) | max speed at rest (m/s) | crab falls | other props moved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cSE_1.0.json | crateSE | True | 1.0 | 0.9002 | 10.72 | 6.69 | 0.08 | 0.81 | 0.00024 | 0 | 0.0 |
| final_crateSE.json | crateSE | True | 1.8 | 0.3634 | 4.4 | -3.79 | 0.08 | 2.28 | 0.00122 | 0 | 0.0 |
| final_planterN.json | planterN | True | 2.0 | 0.3036 | 3.37 | -2.9 | 0.02 | 1.77 | 0.00072 | 0 | 0.0 |
| fixed_planterN.json | planterN | False | None | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |
| pS_1.0.json | planterS | True | 1.0 | 0.8311 | 9.08 | 0.0 | 0.04 | 0.1 | 0.00026 | 0 | 0.0 |
| pS_1.5.json | planterS | True | 1.5 | 0.5266 | 5.75 | 4.88 | 0.05 | 4.21 | 0.00114 | 0 | 0.0 |
| pS_2.0.json | planterS | True | 2.0 | 0.1701 | 1.89 | -1.64 | 0.02 | 0.73 | 0.00018 | 0 | 0.0 |
| pS_4.0.json | planterS | True | 4.0 | 0.0158 | 0.25 | 0.07 | 0.0 | 0.4 | 0.00015 | 0 | 0.0 |
| sE_1.6.json | stairE | True | 1.6 | 0.4842 | 5.26 | 1.4 | 0.04 | 3.85 | 0.00043 | 0 | 0.0 |

Final masses (jhs/pushable.py): stairs 1.8 kg, north planter 2.0 kg, south planter 1.8 kg, crates NE 1.6 / SE 1.8 / NW 1.8 / SW 1.4 kg;
friction 0.7 0.03 0.003 (sliding/torsional/rolling), condim 4, solref 0.02 1; floor 0.8 0.01 0.001 (MuJoCo uses the larger: sliding 0.8).
Files named pS_* / sE_* / cSE_* are the tuning trials (mass overridden with --mass); final_* use the final masses; fixed_* is the control with
pushable obstacles off (planter fixed: 0 m). Rest stability (all 8 props, no robot, 3 s): drift 0.00 mm horizontal, 0.04 mm settling, velocity 0.

## Full hide & seek run with pushable obstacles (box, 2026-10-08)

`./run.sh --headless --random 31 --tmax 600 --move planterN=-0.9,1.6,20 --move crateNE=1.5,1.9,10 --mp4 ... --size 720 --audio ... --audio-loop`

- Result (SIM TRUTH): found 3/3, 2/3 on the rug, **0 falls**, ended at 610.9 s (time-up routine). The crab's base never went below
  0.099 m, and its up-vector z was never below 0.990.
- Props moved by the crab during the run: NW crate 0.42 m, west stairs 0.28 m, south planter 0.06 m, north planter 0.04 m,
  east stairs 0.02 m, SE crate 0.01 m, NE and SW crates 0.00 m.
- Stability: every prop's height stayed constant to within 1 mm the whole run (no tipping, no sinking, no NaN/instability
  warnings). The only motion was short shoves while the crab was in contact. The fastest single 0.1 s sample was 0.35 m/s
  (west stairs) and the prop stopped right after.
