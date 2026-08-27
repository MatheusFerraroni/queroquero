#!/usr/bin/env bash

#SBATCH --job-name=queroquero-classification-embed
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:2
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/classification-eval-%j.out
#SBATCH --error=logs/classification-eval-%j.err

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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nó: $(hostname)"
echo "Início: $(date --iso-8601=seconds)"
echo "Modo: ${MODE}"
echo "Python: $(python --version)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

case "${MODE}" in
  preflight)
    torchrun --standalone --nproc_per_node=2 \
      -m queroquero.classification_eval preflight
    ;;
  embed-base|embed-general|embed-forum)
    MODEL_NAME="${MODE#embed-}"
    torchrun --standalone --nproc_per_node=2 \
      -m queroquero.classification_eval embed --model "${MODEL_NAME}"
    ;;
  *)
    echo "Uso: $0 {preflight|embed-base|embed-general|embed-forum}" >&2
    exit 2
    ;;
esac

echo "Fim: $(date --iso-8601=seconds)"
