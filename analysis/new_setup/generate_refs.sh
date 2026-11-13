#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
python="${PYTHON_BIN:-python}"
cd "$repo"
exec "$python" analysis/glorbot2_effort_tilt20_20261111/generate_refs.py
