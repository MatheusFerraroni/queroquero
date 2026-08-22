#!/usr/bin/env bash

#SBATCH --job-name=queroquero-paired-prepare
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/prepare-paired-%j.out
#SBATCH --error=logs/prepare-paired-%j.err

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

export HF_HOME="${HF_HOME:-${PROJECT_DIR}/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nó: $(hostname)"
echo "Início: $(date --iso-8601=seconds)"
echo "Modo: ${MODE}"
echo "Python: $(python --version)"

case "${MODE}" in
  capacity)
    python -m queroquero.prepare capacity \
      --dataset "${2:?dataset ausente}" \
      --candidate-documents "${3:?candidate_documents ausente}"
    ;;
  prepare)
    DATASETS=(
      adrenaline
      brwac
      gigaverbo
      multiwoz_ptbr
      outerspace
      wackywacky
    )
    DATASET_ID="${DATASETS[${SLURM_ARRAY_TASK_ID:?array task ausente}]}"
    python -m queroquero.prepare run \
      --dataset "${DATASET_ID}" \
      --profile paired_real
    ;;
  verify)
    : "${PTBR_OUTPUT_ROOT:?PTBR_OUTPUT_ROOT deve estar configurado}"
    python -m queroquero.prepare verify-paired-real \
      --general-config configs/training/l40s-real-general.json \
      --forum-tech-config configs/training/l40s-real-forum-tech.json \
      --output-root "${PTBR_OUTPUT_ROOT}" \
      --output "${PTBR_OUTPUT_ROOT}/.capacity/paired-real-verification.json"
    ;;
  *)
    echo "Uso: $0 {capacity <dataset> <candidate_documents>|prepare|verify}" >&2
    exit 2
    ;;
esac

echo "Fim: $(date --iso-8601=seconds)"
