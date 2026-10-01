#!/usr/bin/env bash
# CPU preparation and single-GPU evaluation; never submits training.
set -Eeuo pipefail
HOLDOUT_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
case "$MODE" in
  prepare) RESOURCES=(--cpus-per-task=8 --mem=64G --time=1-00:00:00) ;;
  evaluate) RESOURCES=(--cpus-per-task=4 --mem=32G --time=08:00:00 --gres=gpu:L40S:1) ;;
  validate|report) RESOURCES=(--cpus-per-task=2 --mem=16G --time=01:00:00) ;;
  *) echo "Uso: bash scripts/submit_intrinsic_holdout.sh {prepare|validate|evaluate|report} [afterok-job-id]" >&2; exit 2 ;;
esac
if [[ -n "${2:-}" ]]; then
  [[ "$2" =~ ^[0-9]+$ ]] || { echo "ID de dependência inválido" >&2; exit 2; }
  RESOURCES+=(--dependency="afterok:$2")
fi
mkdir -p "$HOLDOUT_PROJECT_DIR/logs"
sbatch --parsable --job-name="adrenaline-$MODE" \
  --partition="${INTRINSIC_PARTITION:-l40s}" --nodes=1 --ntasks=1 \
  --chdir="$HOLDOUT_PROJECT_DIR" \
  --output="$HOLDOUT_PROJECT_DIR/logs/intrinsic-$MODE-%j.out" \
  --error="$HOLDOUT_PROJECT_DIR/logs/intrinsic-$MODE-%j.err" \
  --export="ALL,HOLDOUT_PROJECT_DIR=$HOLDOUT_PROJECT_DIR" \
  "${RESOURCES[@]}" \
  "$HOLDOUT_PROJECT_DIR/scripts/run_intrinsic_holdout.sbatch" "$MODE"
