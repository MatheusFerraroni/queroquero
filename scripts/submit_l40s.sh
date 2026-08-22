#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-}"

case "${MODE}" in
  preflight|smoke-stop|smoke-resume|mvp|mvp-resume|real-preflight|real|real-resume|validate) ;;
  *)
    echo "Uso: $0 {preflight|smoke-stop|smoke-resume|mvp|mvp-resume|real-preflight|real|real-resume|validate}" >&2
    exit 2
    ;;
esac

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

SBATCH_TIME="1-00:00:00"
if [[ "${MODE}" == real* ]]; then
  SBATCH_TIME="13:00:00"
  REAL_CONFIG_PATH="${REAL_TRAINING_CONFIG:-configs/training/l40s-real.json}"
  if [[ ! -f "${REAL_CONFIG_PATH}" ]]; then
    echo "Erro: config real ausente; execute capacity/allocate-real e versione os budgets primeiro." >&2
    exit 2
  fi
fi

sbatch \
  --parsable \
  --partition="${L40S_PARTITION:-l40s}" \
  --time="${SBATCH_TIME}" \
  --export=ALL \
  "${PROJECT_DIR}/scripts/train_l40s.sbatch" \
  "${MODE}"
