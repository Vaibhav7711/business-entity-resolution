#!/usr/bin/env bash
# One-time node setup (x86_64 Linux). Creates ~/venv with Python 3.12 and the pinned requirements.
# Usage: bash cloud/setup_node.sh [gpu]   (run from code/business_entity_resolution)
set -euo pipefail
[ "$(uname -m)" = "x86_64" ] || { echo "Refusing: this pipeline must run on x86_64 (candidate reproducibility)"; exit 1; }
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
[ -d "$HOME/venv" ] || uv venv -p 3.12 "$HOME/venv"
. "$HOME/venv/bin/activate"
uv pip install -r requirements.txt kaggle awscli pytest
if [ "${1:-}" = "gpu" ]; then uv pip install -r requirements-gpu.txt; fi
python -c "import platform,sys;print('python',sys.version.split()[0],platform.machine())"
lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2
python -m pytest -q 2>&1 | tail -1
