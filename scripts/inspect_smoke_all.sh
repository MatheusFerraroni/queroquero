#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Erro: uv não foi encontrado no PATH." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "Erro: ${PROJECT_DIR}/.env não foi encontrado." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Erro: ${PROJECT_DIR}/.venv/bin/python não foi encontrado." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

output_root="${PTBR_OUTPUT_ROOT:-derived}"
if [[ "${output_root}" != /* ]]; then
  output_root="${PROJECT_DIR}/${output_root}"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/cache/uv}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

datasets=(
  brwac
  wackywacky
  multiwoz_ptbr
  outerspace
  adrenaline
  gigaverbo
)

failed_datasets=()

for dataset in "${datasets[@]}"; do
  dataset_root="${output_root%/}/${dataset}"
  preparation_dir="$(
    .venv/bin/python -c '
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifests = []
for path in root.glob("*/dataset_manifest.json"):
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continue
    if manifest.get("profile") == "smoke":
        manifests.append(path)

if not manifests:
    raise SystemExit(1)

manifest = max(manifests, key=lambda path: path.stat().st_mtime_ns)
print(manifest.parent)
' "${dataset_root}"
  )"

  if [[ -z "${preparation_dir}" ]]; then
    echo >&2
    echo "===== Smoke não encontrado: ${dataset} (${dataset_root}) =====" >&2
    failed_datasets+=("${dataset}")
    continue
  fi

  echo
  echo "==================== ${dataset} ===================="
  echo "Preparação: ${preparation_dir}"

  if ! "${SCRIPT_DIR}/inspect_preparation.sh" \
    "${preparation_dir}" \
    --decode-sample \
    --split train \
    --row 0
  then
    failed_datasets+=("${dataset}")
    echo "===== Falhou: ${dataset} =====" >&2
  fi
done

echo
if (( ${#failed_datasets[@]} > 0 )); then
  echo "Datasets com falha: ${failed_datasets[*]}" >&2
  exit 1
fi

echo "As rows 0 dos seis smokes foram inspecionadas."
