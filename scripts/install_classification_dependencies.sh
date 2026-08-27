#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
source "${HOME}/activate_queroquero.sh"
python -m pip install -r requirements-classification.txt
python - <<'PY'
from importlib.metadata import version

versions = {
    "numpy": version("numpy"),
    "scipy": version("scipy"),
    "scikit-learn": version("scikit-learn"),
}
if versions["scikit-learn"] != "1.9.0":
    raise SystemExit("scikit-learn version mismatch")
for name, value in versions.items():
    print(f"{name}={value}")
PY
