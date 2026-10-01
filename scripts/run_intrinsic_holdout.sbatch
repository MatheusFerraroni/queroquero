#!/usr/bin/env bash
set -Eeuo pipefail
: "${HOLDOUT_PROJECT_DIR:?Submit with scripts/submit_intrinsic_holdout.sh}"
MODE="${1:?Missing mode}"
source "$HOME/activate_queroquero.sh"
cd "$HOLDOUT_PROJECT_DIR"
[[ -f .env ]] || { echo "Arquivo .env ausente" >&2; exit 2; }
set -a
source .env
set +a
: "${PTBR_DATASET_ROOT:?Missing dataset root}"
: "${PTBR_OUTPUT_ROOT:?Missing prepared-data root}"
export HF_HOME="${HF_HOME:-$HOLDOUT_PROJECT_DIR/cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "Job: ${SLURM_JOB_ID:-local}; modo: $MODE; projeto: $HOLDOUT_PROJECT_DIR"
python --version
exec python -m queroquero.intrinsic_holdout "$MODE" \
  --config configs/intrinsic/adrenaline-holdout-v1.json
