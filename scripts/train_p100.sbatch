#!/usr/bin/env bash

#SBATCH --job-name=queroquero-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

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

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nó: $(hostname)"
echo "Início: $(date --iso-8601=seconds)"
echo "Modo: ${MODE}"
echo "Python: $(python --version)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

TRAIN_PID=""

forward_usr1() {
  if [[ -n "${TRAIN_PID}" ]]; then
    kill -USR1 "${TRAIN_PID}" 2>/dev/null || true
  fi
}

run_python() {
  trap forward_usr1 USR1
  python "$@" &
  TRAIN_PID=$!
  local status
  while true; do
    if wait "${TRAIN_PID}"; then
      status=0
      break
    else
      status=$?
    fi
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      break
    fi
  done
  TRAIN_PID=""
  trap - USR1
  return "${status}"
}

case "${MODE}" in
  preflight)
    run_python -m queroquero.train preflight \
      --config configs/training/p100-smoke.json
    ;;
  smoke-stop)
    run_python -m queroquero.train run \
      --config configs/training/p100-smoke.json \
      --stop-after-checkpoint
    ;;
  smoke-resume)
    run_python -m queroquero.train run \
      --config configs/training/p100-smoke.json \
      --resume
    ;;
  mvp)
    run_python -m queroquero.train run \
      --config configs/training/p100-mvp.json
    ;;
  mvp-resume)
    run_python -m queroquero.train run \
      --config configs/training/p100-mvp.json \
      --resume
    ;;
  validate)
    if [[ -z "${MODEL_ARTIFACT_PATH:-}" ]]; then
      echo "Erro: MODEL_ARTIFACT_PATH é obrigatório no modo validate." >&2
      exit 2
    fi
    run_python -m queroquero.train validate \
      --artifact "${MODEL_ARTIFACT_PATH}"
    ;;
  *)
    echo "Uso: $0 {preflight|smoke-stop|smoke-resume|mvp|mvp-resume|validate}" >&2
    exit 2
    ;;
esac

echo "Fim: $(date --iso-8601=seconds)"
