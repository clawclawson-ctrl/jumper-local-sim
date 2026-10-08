#!/usr/bin/env bash
# Jumper Hide & Seek -- Local Sim.  SIM-ONLY VISION CONCEPT.
#   ./run.sh                     open the setup page in your browser (pick or load an app there)
#   ./run.sh path/to/some.app    same, with that app selected
#   ./run.sh --pick              choose the .app in a file dialog first
#   ./run.sh --headless ...      no window: run and save an MP4 (see ./run.sh --headless --help)
#   ./run.sh --render RUN_DIR [OUT.mp4] [--size 720]   re-render a recorded run as an MP4 (1x, full overlay)
#   ./run.sh --add-audio VIDEO.mp4 SONG.mp3 [OUT.mp4] [--start S --delay S --volume V --fade S --loop]
#   ./run.sh --check [--full]    self-test
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"
if [ ! -x "$PY" ]; then echo "Not installed yet: run  bash install.sh  first."; exit 1; fi
cd "$HERE"
case "${1:-}" in
  --check) shift; exec "$PY" -m jhs.check "$@";;
  --headless) shift; exec "$PY" -m jhs.headless "$@";;
  --render) shift; exec "$PY" -m jhs.render_run "$@";;
  --add-audio) shift; exec "$PY" -m jhs.audio "$@";;
  --push-test) shift; export JUMPER_REPO="$HERE/toolkit"; exec "$PY" -m jhs.push_test "$@";;
  *) exec "$PY" -m jhs.server "$@";;
esac
