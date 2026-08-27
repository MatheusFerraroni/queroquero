#!/usr/bin/env bash

#SBATCH --job-name=queroquero-classification-probe
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/classification-eval-%A_%a.out
#SBATCH --error=logs/classification-eval-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-}"

cd "${PROJECT_DIR}"
source "${HOME}/activate_queroquero.sh"

if [[ ! -f .env ]]; then
  echo "Erro: ${PROJECT_DIR}/.env não foi encontrado." >&2
  exit 2
fi
set -a
source .env
set +a

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nó: $(hostname)"
echo "Início: $(date --iso-8601=seconds)"
echo "Modo: ${MODE}"
echo "Python: $(python --version)"

case "${MODE}" in
  validate-embeddings|select-hyperparameters|report|validate-report)
    python -m queroquero.classification_eval "${MODE}"
    ;;
  tune-unit|evaluate-unit)
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      echo "Erro: ${MODE} exige um array Slurm." >&2
      exit 2
    fi
    python -m queroquero.classification_eval "${MODE}" \
      --unit-index "${SLURM_ARRAY_TASK_ID}"
    ;;
  *)
    echo "Uso: $0 {validate-embeddings|tune-unit|select-hyperparameters|evaluate-unit|report|validate-report}" >&2
    exit 2
    ;;
esac

echo "Fim: $(date --iso-8601=seconds)"
