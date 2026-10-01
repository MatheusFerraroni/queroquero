#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-}"

case "${MODE}" in
  preflight|smoke-stop|smoke-resume|mvp|mvp-resume|real-general-preflight|real-general|real-general-resume|real-forum-tech-preflight|real-forum-tech|real-forum-tech-resume|validate) ;;
  real|real-preflight|real-resume)
    echo "Erro: modo real ambíguo; escolha explicitamente real-general ou real-forum-tech." >&2
    exit 2
    ;;
  *)
    echo "Uso: $0 {preflight|smoke-stop|smoke-resume|mvp|mvp-resume|real-general-preflight|real-general|real-general-resume|real-forum-tech-preflight|real-forum-tech|real-forum-tech-resume|validate}" >&2
    exit 2
    ;;
esac

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

SBATCH_TIME="1-00:00:00"
if [[ "${MODE}" == real-general* ]]; then
  REAL_CONFIG_PATH="configs/training/l40s-real-general.json"
  if [[ ! -f "${REAL_CONFIG_PATH}" ]]; then
    echo "Erro: config Geral ausente; execute a auditoria e versione a alocação pareada primeiro." >&2
    exit 2
  fi
elif [[ "${MODE}" == real-forum-tech* ]]; then
  REAL_CONFIG_PATH="configs/training/l40s-real-forum-tech.json"
  if [[ ! -f "${REAL_CONFIG_PATH}" ]]; then
    echo "Erro: config Fórum/Tec ausente; execute a auditoria e versione a alocação pareada primeiro." >&2
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
