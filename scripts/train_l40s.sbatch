#!/usr/bin/env bash

#SBATCH --job-name=queroquero-l40s
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:2
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/train-l40s-%j.out
#SBATCH --error=logs/train-l40s-%j.err

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
export TRAIN_INTERRUPT_FILE="${PROJECT_DIR}/logs/train-l40s-${SLURM_JOB_ID}.interrupt"
export TRAIN_STATUS_FILE="${PROJECT_DIR}/logs/train-l40s-${SLURM_JOB_ID}.status.json"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nó: $(hostname)"
echo "Início: $(date --iso-8601=seconds)"
echo "Modo: ${MODE}"
echo "Python: $(python --version)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

RUN_PID=""

record_usr1() {
  touch "${TRAIN_INTERRUPT_FILE}"
}

run_background() {
  trap record_usr1 USR1
  "$@" &
  RUN_PID=$!
  local status=0
  while true; do
    if wait "${RUN_PID}"; then
      status=0
      break
    else
      status=$?
    fi
    if ! kill -0 "${RUN_PID}" 2>/dev/null; then
      break
    fi
  done
  RUN_PID=""
  trap - USR1
  return "${status}"
}

run_ddp() {
  run_background torchrun --standalone --nproc_per_node=2 \
    -m queroquero.train "$@"
}

case "${MODE}" in
  preflight)
    run_ddp preflight --config configs/training/l40s-smoke.json
    ;;
  smoke-stop)
    run_ddp run --config configs/training/l40s-smoke.json \
      --stop-after-checkpoint
    ;;
  smoke-resume)
    run_ddp run --config configs/training/l40s-smoke.json --resume
    ;;
  mvp)
    run_ddp run --config configs/training/l40s-mvp.json
    ;;
  mvp-resume)
    run_ddp run --config configs/training/l40s-mvp.json --resume
    ;;
  real-preflight)
    run_ddp preflight --config \
      "${REAL_TRAINING_CONFIG:-configs/training/l40s-real.json}"
    ;;
  real)
    run_ddp run --config \
      "${REAL_TRAINING_CONFIG:-configs/training/l40s-real.json}"
    ;;
  real-resume)
    run_ddp run --config \
      "${REAL_TRAINING_CONFIG:-configs/training/l40s-real.json}" --resume
    ;;
  validate)
    if [[ -z "${MODEL_ARTIFACT_PATH:-}" ]]; then
      echo "Erro: MODEL_ARTIFACT_PATH é obrigatório no modo validate." >&2
      exit 2
    fi
    run_background python -m queroquero.train validate \
      --artifact "${MODEL_ARTIFACT_PATH}"
    ;;
  *)
    echo "Uso: $0 {preflight|smoke-stop|smoke-resume|mvp|mvp-resume|real-preflight|real|real-resume|validate}" >&2
    exit 2
    ;;
esac

if [[ -f "${TRAIN_STATUS_FILE}" ]]; then
  INTERRUPTED_EXIT_CODE="$(python -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value.get("exit_code", 0) if value.get("status") == "interrupted" else 0)
' "${TRAIN_STATUS_FILE}")"
  if [[ "${INTERRUPTED_EXIT_CODE}" -eq 99 ]]; then
    echo "Treino interrompido com checkpoint retomável." >&2
    exit 99
  fi
fi

echo "Fim: $(date --iso-8601=seconds)"
