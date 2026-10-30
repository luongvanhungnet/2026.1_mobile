#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_ENV="${GRIDRESTORE_CONVERTER_HOME:-$HOME/.local/share/gridrestore}"
UV="$TASK_ENV/bin/uv"
PY="$TASK_ENV/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11"
VENV="$TASK_ENV/venv"
mkdir -p "$ROOT/tools/work"
exec > >(tee -a "$ROOT/tools/work/linux-setup.log") 2>&1
test "$(uname -m)" = x86_64 || { echo 'This pinned converter environment requires Linux x86_64'; exit 1; }
if [ ! -x "$UV" ]; then
  mkdir -p "$TASK_ENV/bin"
  curl --fail --location --retry 2 --output "$TASK_ENV/uv.tar.gz" https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-gnu.tar.gz
  echo "2cf10babba653310606f8b49876cfb679928669e7ddaa1fb41fb00ce73e64f66  $TASK_ENV/uv.tar.gz" | sha256sum --check
  tar -xzf "$TASK_ENV/uv.tar.gz" --strip-components=1 -C "$TASK_ENV/bin"
fi
if [ ! -x "$PY" ]; then "$UV" python install 3.11.14 --install-dir "$TASK_ENV/python"; fi
if [ ! -x "$VENV/bin/python" ]; then "$UV" venv --python "$PY" "$VENV"; fi
"$UV" pip install --python "$VENV/bin/python" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
REQUIREMENTS="$ROOT/tools/requirements-convert.txt"
if [ -f "$ROOT/tools/requirements-convert-lock.txt" ]; then REQUIREMENTS="$ROOT/tools/requirements-convert-lock.txt"; fi
"$UV" pip install --python "$VENV/bin/python" -r "$REQUIREMENTS"
"$UV" pip freeze --python "$VENV/bin/python" > "$ROOT/tools/work/conversion-environment.txt"
"$VENV/bin/python" -c 'import torch,litert_torch; from ai_edge_litert.interpreter import Interpreter; print("CONVERTER_IMPORT_OK", torch.__version__)'
