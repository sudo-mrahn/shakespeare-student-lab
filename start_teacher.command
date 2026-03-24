#!/bin/zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec /bin/zsh "$SCRIPT_DIR/start_shell.command"
