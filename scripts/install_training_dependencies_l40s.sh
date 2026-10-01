#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
source "${HOME}/activate_queroquero.sh"

python -m pip install --requirement requirements-train-l40s.txt

python - <<'PY'
import torch
import transformers

assert torch.__version__ == "2.7.1+cu118", torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
assert transformers.__version__ == "5.14.1", transformers.__version__
assert torch.distributed.is_available()
assert torch.distributed.is_nccl_available()
print("Ambiente de treino L40S instalado e pinado.")
PY
