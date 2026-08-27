#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-}"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

COMMON=(
  --parsable
  --time=1-00:00:00
  --export=ALL
)

case "${MODE}" in
  preflight|embed-base|embed-general|embed-forum)
    sbatch "${COMMON[@]}" \
      --partition="${CLASSIFICATION_GPU_PARTITION:-l40s}" \
      --cpus-per-task=8 \
      --mem=128G \
      --gres=gpu:L40S:2 \
      --output="${PROJECT_DIR}/logs/classification-eval-%j.out" \
      --error="${PROJECT_DIR}/logs/classification-eval-%j.err" \
      "${PROJECT_DIR}/scripts/classification_embeddings.sbatch" "${MODE}"
    ;;
  tune-unit|evaluate-unit)
    sbatch "${COMMON[@]}" \
      --partition="${CLASSIFICATION_CPU_PARTITION:-l40s}" \
      --cpus-per-task=16 \
      --mem=128G \
      --array=0-19%4 \
      --output="${PROJECT_DIR}/logs/classification-eval-%A_%a.out" \
      --error="${PROJECT_DIR}/logs/classification-eval-%A_%a.err" \
      "${PROJECT_DIR}/scripts/classification_probe.sbatch" "${MODE}"
    ;;
  validate-embeddings|select-hyperparameters|report|validate-report)
    sbatch "${COMMON[@]}" \
      --partition="${CLASSIFICATION_CPU_PARTITION:-l40s}" \
      --cpus-per-task=16 \
      --mem=128G \
      --output="${PROJECT_DIR}/logs/classification-eval-%j.out" \
      --error="${PROJECT_DIR}/logs/classification-eval-%j.err" \
      "${PROJECT_DIR}/scripts/classification_probe.sbatch" "${MODE}"
    ;;
  *)
    echo "Uso: $0 {preflight|embed-base|embed-general|embed-forum|validate-embeddings|tune-unit|select-hyperparameters|evaluate-unit|report|validate-report}" >&2
    exit 2
    ;;
esac
