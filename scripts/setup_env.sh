#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
WHEELHOUSE_PATH="${1:-}"

python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

if [[ -n "${WHEELHOUSE_PATH}" ]]; then
  python -m pip install --no-index --find-links "${WHEELHOUSE_PATH}" setuptools
  python -m pip install --no-index --find-links "${WHEELHOUSE_PATH}" -r "${PROJECT_ROOT}/requirements/train.txt"
else
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r "${PROJECT_ROOT}/requirements/train.txt"
fi

python -m pip install -e "${PROJECT_ROOT}"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("devices", torch.cuda.device_count())
PY
