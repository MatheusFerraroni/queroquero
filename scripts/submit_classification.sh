#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-}"
shift || true

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

SBATCH_ARGS=(
  --parsable
  --partition="${CLASSIFICATION_PARTITION:-l40s}"
  --time=1-00:00:00
  --export=ALL
  --output="${PROJECT_DIR}/logs/classification-%j.out"
  --error="${PROJECT_DIR}/logs/classification-%j.err"
)

case "${MODE}" in
  build)
    sbatch "${SBATCH_ARGS[@]}" \
      "${PROJECT_DIR}/scripts/prepare_classification.sbatch" build
    ;;
  validate-dataset)
    DATASET_PATH="${1:-}"
    if [[ -z "${DATASET_PATH}" ]]; then
      echo "Erro: informe o diretório do dataset canônico." >&2
      exit 2
    fi
    sbatch "${SBATCH_ARGS[@]}" \
      "${PROJECT_DIR}/scripts/prepare_classification.sbatch" \
      validate-dataset "${DATASET_PATH}"
    ;;
  split)
    DATASET_PATH="${1:-}"
    TASK="${2:-}"
    SEED="${3:-}"
    OUTPUT_PATH="${4:-}"
    if [[ -z "${DATASET_PATH}" || -z "${OUTPUT_PATH}" ]]; then
      echo "Erro: informe dataset, tarefa, seed e saída." >&2
      exit 2
    fi
    case "${TASK}" in coarse|fine) ;; *) echo "Erro: tarefa deve ser coarse ou fine." >&2; exit 2 ;; esac
    case "${SEED}" in 42|43|44|45|46) ;; *) echo "Erro: seed deve estar entre 42 e 46." >&2; exit 2 ;; esac
    sbatch "${SBATCH_ARGS[@]}" \
      "${PROJECT_DIR}/scripts/prepare_classification.sbatch" \
      split "${DATASET_PATH}" "${TASK}" "${SEED}" "${OUTPUT_PATH}"
    ;;
  validate-split)
    DATASET_PATH="${1:-}"
    SPLIT_PATH="${2:-}"
    if [[ -z "${DATASET_PATH}" || -z "${SPLIT_PATH}" ]]; then
      echo "Erro: informe o dataset e o manifesto de split." >&2
      exit 2
    fi
    sbatch "${SBATCH_ARGS[@]}" \
      "${PROJECT_DIR}/scripts/prepare_classification.sbatch" \
      validate-split "${DATASET_PATH}" "${SPLIT_PATH}"
    ;;
  *)
    echo "Uso: $0 {build|validate-dataset <dataset>|split <dataset> <coarse|fine> <42-46> <output>|validate-split <dataset> <manifest>}" >&2
    exit 2
    ;;
esac
