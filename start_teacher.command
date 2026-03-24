#!/bin/zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN=".venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  python3 -m venv .venv
  PYTHON_BIN=".venv/bin/python"
fi

if ! "$PYTHON_BIN" -c "import numpy" >/dev/null 2>&1; then
  printf '%s\n' "The teacher package needs NumPy before it can start."
  printf '%s\n' "This will install the local requirements into .venv from requirements.txt."
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

printf '%s\n' "Teacher mode includes advanced commands such as checkpoint loading and deletion."
printf '%s' "Continue? [y/N] "
read -r teacher_ok
case "$teacher_ok" in
  y|Y|yes|YES)
    ;;
  *)
    printf '%s\n' "Teacher mode canceled."
    exit 1
    ;;
esac

exec "$PYTHON_BIN" text_generator.py shell
