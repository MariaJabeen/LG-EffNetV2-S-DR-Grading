#!/usr/bin/env bash
# Run a notebook top to bottom without a browser.
# Usage: [DRXAI_TRAIN=1] [DRXAI_SMOKE=1] bash scripts/run_notebook.sh <folder>/<name>.ipynb
set -euo pipefail
NB="${1:?usage: run_notebook.sh <notebook.ipynb>}"
cd "$(dirname "$NB")"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output "$(basename "${NB%.ipynb}").executed.ipynb" "$(basename "$NB")"
