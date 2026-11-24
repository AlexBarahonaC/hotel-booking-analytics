#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
for PY in "${PYTHON_BIN:-}" python3.13 python3.12; do
  if [ -n "$PY" ] && command -v "$PY" >/dev/null 2>&1; then PYTHON_BIN="$PY"; break; fi
done
if ! command -v "${PYTHON_BIN:-}" >/dev/null 2>&1; then
  echo "ERROR: se necesita Python 3.12 o 3.13." >&2
  exit 1
fi
[ -d .venv ] || "$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/smoke_test.py
python -m streamlit run dashboard/app.py
