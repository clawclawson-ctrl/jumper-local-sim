#!/usr/bin/env bash
# Jumper Hide & Seek -- Local Sim: one-time installer.  SIM-ONLY VISION CONCEPT.
#   bash install.sh                    (re-run any time; it is safe to repeat)
#   bash install.sh --controller-only  (only rebuild/reinstall the robot controller into the existing .venv)
# Makes ./.venv with Python 3.12 (via uv), installs the official Jumper toolkit (./toolkit, Apache-2.0) and its
# dependencies, builds the robot's controller (Rust, ./toolkit/deploy/fsm) for this computer, and self-tests.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PYVER="${JHS_PYTHON:-3.12}"
ONLY_CTRL=0; if [ "${1:-}" = "--controller-only" ]; then ONLY_CTRL=1; fi
LOG="$HERE/install.log"
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mINSTALL FAILED: %s\033[0m\nFull log: %s\n' "$*" "$LOG"; exit 1; }
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
OS="$(uname -s)"; ARCH="$(uname -m)"
say "Jumper Hide & Seek -- Local Sim installer ($OS $ARCH)"
case "$HERE" in *" "*) echo "note: the folder path contains spaces; that is fine, but if anything odd happens move it to e.g. ~/JumperLocalSim";; esac

# ---- uv ---------------------------------------------------------------------------------------------------
UV="$(command -v uv || true)"
for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do if [ -z "$UV" ] && [ -x "$c" ]; then UV="$c"; fi; done
[ -n "$UV" ] || fail "uv not found. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh   (then open a new Terminal and re-run)"
echo "uv: $UV ($("$UV" --version))"

if [ "$ONLY_CTRL" = "1" ]; then
  [ -x "$HERE/.venv/bin/python" ] || fail "--controller-only needs an existing install (.venv); run  bash install.sh  first"
  PY="$HERE/.venv/bin/python"
else
# ---- Python + venv ----------------------------------------------------------------------------------------
say "Python $PYVER (uv-managed, not Homebrew's)"
"$UV" python install "$PYVER"
if [ ! -x "$HERE/.venv/bin/python" ] || ! "$HERE/.venv/bin/python" -c "import sys; assert sys.version.startswith('$PYVER')" 2>/dev/null; then
  rm -rf "$HERE/.venv"; "$UV" venv --python "$PYVER" "$HERE/.venv"
fi
PY="$HERE/.venv/bin/python"; "$PY" -V

# ---- Python packages --------------------------------------------------------------------------------------
say "Installing the toolkit and its dependencies (first time: several minutes, ~1-2 GB download)"
EXTRA=()
if [ "$OS" = "Linux" ]; then EXTRA=(--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match); fi
"$UV" pip install --python "$PY" ${EXTRA[@]+"${EXTRA[@]}"} -e "$HERE/toolkit" pillow "imageio>=2.31" imageio-ffmpeg || fail "pip install"
fi

# ---- controller -------------------------------------------------------------------------------------------
say "Robot controller (Rust, from toolkit/deploy/fsm)"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')"
DEST="$SITE/mjrl_fsm.abi3.so"
built=""
CARGO="$(command -v cargo || true)"
if [ -z "$CARGO" ] && [ -x "$HOME/.cargo/bin/cargo" ]; then CARGO="$HOME/.cargo/bin/cargo"; fi
try_import() { "$PY" -c "import mjrl_fsm; print('mjrl_fsm imports OK:', mjrl_fsm.__file__)"; }
if [ "${JHS_USE_PREBUILT:-0}" != "1" ] && [ -n "$CARGO" ]; then
  export PATH="$(dirname "$CARGO"):$PATH"
  if ! "$CARGO" --version >/dev/null 2>&1; then
    echo "cargo has no default toolchain yet; installing Rust stable with rustup"
    RUSTUP="$(command -v rustup || echo "$HOME/.cargo/bin/rustup")"
    if [ -x "$RUSTUP" ]; then "$RUSTUP" default stable || true; fi
  fi
  if "$CARGO" --version >/dev/null 2>&1; then
    "$CARGO" --version
    if ( cd "$HERE/toolkit/deploy/fsm" && CARGO_TARGET_DIR="$HERE/.build/fsm" "$CARGO" build --release --no-default-features --features py ); then
      LIB=""
      if [ -f "$HERE/.build/fsm/release/libmjrl_fsm.dylib" ]; then LIB="$HERE/.build/fsm/release/libmjrl_fsm.dylib"; fi
      if [ -z "$LIB" ] && [ -f "$HERE/.build/fsm/release/libmjrl_fsm.so" ]; then LIB="$HERE/.build/fsm/release/libmjrl_fsm.so"; fi
      if [ -n "$LIB" ]; then
        rm -f "$DEST"; cp "$LIB" "$DEST"
        if try_import; then built="built here with cargo"; else echo "the cargo-built controller does not import"; rm -f "$DEST"; fi
      else
        echo "cargo finished but no library was found in .build/fsm/release"
      fi
    else
      echo "cargo build failed (see above)"
    fi
  else
    echo "cargo is not usable"
  fi
else
  if [ "${JHS_USE_PREBUILT:-0}" = "1" ]; then echo "JHS_USE_PREBUILT=1: skipping the cargo build"; else echo "cargo not found"; fi
fi
if [ -z "$built" ] && [ "$OS" = "Darwin" ]; then
  echo "using the prebuilt macOS controller shipped in prebuilt/ (same source, cross-built for universal2)"
  rm -f "$DEST"; cp "$HERE/prebuilt/macosx-universal2/mjrl_fsm.abi3.so" "$DEST"
  xattr -d com.apple.quarantine "$DEST" 2>/dev/null || true
  codesign --force --sign - "$DEST" 2>/dev/null || true
  if try_import; then built="prebuilt universal2"; fi
fi
[ -n "$built" ] || fail "no working controller: install Rust (https://rustup.rs), then re-run  bash install.sh"
if [ "$OS" = "Darwin" ]; then xattr -dr com.apple.quarantine "$HERE" 2>/dev/null || true; fi
echo "controller: $built -> $DEST"
if [ "$OS" = "Darwin" ] && [ ! -x "$HERE/.venv/bin/mjpython" ]; then echo "note: mjpython not found in .venv/bin (only needed for plain apps' 3D window)"; fi

if [ "$ONLY_CTRL" = "1" ]; then say "Controller reinstalled."; exit 0; fi
# ---- self-test --------------------------------------------------------------------------------------------
say "Self-test"
"$PY" -m jhs.check || fail "self-test (see messages above)"
chmod +x "$HERE/run.sh" 2>/dev/null || true
say "Installed.  Start it with:   ./run.sh      (opens the setup page in your browser)"
