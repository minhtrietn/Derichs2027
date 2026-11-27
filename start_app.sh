#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PY_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv_linux/bin/python" ]; then
  PY_BIN="$SCRIPT_DIR/.venv_linux/bin/python"
elif command -v python3.11 >/dev/null 2>&1; then
  PY_BIN="$(command -v python3.11)"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="$(command -v python3)"
else
  echo "[ERROR] Khong tim thay python3. Hay cai Python 3.11 truoc (VD: sudo apt install python3.11 python3.11-venv)."
  exit 1
fi

exec "$PY_BIN" "$SCRIPT_DIR/run.py" "$@"
