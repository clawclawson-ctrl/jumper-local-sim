# Licences — Jumper Hide & Seek, Local Sim

SIM-ONLY VISION CONCEPT.

- `toolkit/`: the parts of the official Jumper toolkit this sim needs (KingKongRobotics/jumper, commit 7d3cc4b), unchanged
  except that example training checkpoints were left out. Apache License 2.0. See `toolkit/LICENSE` and `toolkit/NOTICE`,
  plus the third-party notices in `toolkit/licenses/`.
- `prebuilt/macosx-universal2/mjrl_fsm.abi3.so`: compiled from `toolkit/deploy/fsm` (same commit), under the same licence.
  `install.sh` normally rebuilds it on your Mac, and this file is only a fallback.
- `apps/*.app`: Jumper apps. Each one bundles the toolkit's policies and controller (Apache-2.0) and a vision-brain concept
  layered on top.
- `apps/flybrain.app` + `jhs/brains/`: the official Jumper locomotion policy (jumper.posture, model_74800), unmodified and
  renamed `flybrain`, Apache-2.0 KingKongRobotics/jumper (see `jhs/brains/NOTICE-flybrain`). The fly-inspired rules in
  `jhs/brains/sim_only_flybrain.py` are a local SIM-ONLY VISION CONCEPT, not a connectome.
- `apps/jumper-tidy-up.app`: the official tidy-up sample, rebuilt with the toolkit's `scripts/deploy.py` from
  KingKongRobotics/jumper 7d3cc4b. Its policies are byte-identical to the official bundle (`out/bundle_example`) in that
  public Apache-2.0 repository. Credit: KingKong Robotics.
- `maps/*.map`: two rooms assembled for this sim from official KingKong scene pieces (the jumper-design library maps:
  Shelf-maze, Home, Plaza, Bedroom, Warehouse). Each map keeps its own `README.md` and `scene-package.json` with the
  provenance of the pieces, plus the embedded robot's notices under `robot/`. The source pieces come from
  KingKongRobotics/jumper-design, a public repository. Its NOTICE says third-party assets are not relicensed by its
  Apache-2.0 licence, and the source maps carry no separate asset licence field. **The Apache-2.0 licence of this
  repository does not grant any new rights to those scene assets.** Any rights in them stay with their owners.
- `jhs/fonts/DejaVuSans*.ttf`: DejaVu fonts. See `jhs/fonts/LICENSE-DejaVu.txt`.
- Python packages (MuJoCo, PyTorch, ONNX Runtime, NumPy, Pillow, imageio, ...) are downloaded by `install.sh` under their
  own licences. They are not shipped in this folder.
