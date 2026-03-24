#!/bin/zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

# macOS Python refuses to create venvs anywhere inside a path containing ':'.
if [[ "$SCRIPT_DIR" == *:* ]]; then
  VENV_ROOT="$HOME/.shakespeare-text-lab/venvs"
  PROJECT_HASH="$(printf '%s' "$SCRIPT_DIR" | shasum -a 256 | awk '{print $1}')"
  VENV_DIR="$VENV_ROOT/$PROJECT_HASH"
fi

PYTHON_BIN="$VENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  mkdir -p "$(dirname "$VENV_DIR")"
  python3 -m venv "$VENV_DIR"
fi

if ! "$PYTHON_BIN" -c "import numpy" >/dev/null 2>&1; then
  printf '%s\n' "The text lab needs NumPy before it can start."
  printf '%s\n' "This will install the local requirements into $VENV_DIR from requirements.txt."
  printf '%s' "Install now? [y/N] "
  read -r answer
  case "$answer" in
    y|Y|yes|YES)
      "$PYTHON_BIN" -m pip install --requirement requirements.txt
      ;;
    *)
      printf '%s\n' "Setup canceled."
      exit 1
      ;;
  esac
fi

exec "$PYTHON_BIN" text_generator.py shell
