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
CPU_PARTITION="${CLASSIFICATION_DIAGNOSTICS_CPU_PARTITION:-${CLASSIFICATION_CPU_PARTITION:-l40s}}"
GPU_PARTITION="${CLASSIFICATION_DIAGNOSTICS_GPU_PARTITION:-${CLASSIFICATION_GPU_PARTITION:-l40s}}"

case "${MODE}" in
  prepare-cohort|validate-cohort|validate-scores|report|validate-report)
    sbatch "${COMMON[@]}" \
      --partition="${CPU_PARTITION}" \
      --cpus-per-task=16 \
      --mem=64G \
      --output="${PROJECT_DIR}/logs/classification-diagnostics-%j.out" \
      --error="${PROJECT_DIR}/logs/classification-diagnostics-%j.err" \
      "${PROJECT_DIR}/scripts/classification_diagnostics_cpu.sbatch" "${MODE}"
    ;;
  low-shot-unit)
    sbatch "${COMMON[@]}" \
      --partition="${CPU_PARTITION}" \
      --cpus-per-task=16 \
      --mem=64G \
      --array=0-4%4 \
      --output="${PROJECT_DIR}/logs/classification-diagnostics-%A_%a.out" \
      --error="${PROJECT_DIR}/logs/classification-diagnostics-%A_%a.err" \
      "${PROJECT_DIR}/scripts/classification_diagnostics_cpu.sbatch" "${MODE}"
    ;;
  preflight)
    sbatch "${COMMON[@]}" \
      --partition="${GPU_PARTITION}" \
      --cpus-per-task=8 \
      --mem=64G \
      --gres=gpu:L40S:1 \
      --output="${PROJECT_DIR}/logs/classification-diagnostics-%j.out" \
      --error="${PROJECT_DIR}/logs/classification-diagnostics-%j.err" \
      "${PROJECT_DIR}/scripts/classification_diagnostics_gpu.sbatch" "${MODE}"
    ;;
  score-unit)
    sbatch "${COMMON[@]}" \
      --partition="${GPU_PARTITION}" \
      --cpus-per-task=8 \
      --mem=64G \
      --gres=gpu:L40S:1 \
      --array=0-8%2 \
      --output="${PROJECT_DIR}/logs/classification-diagnostics-%A_%a.out" \
      --error="${PROJECT_DIR}/logs/classification-diagnostics-%A_%a.err" \
      "${PROJECT_DIR}/scripts/classification_diagnostics_gpu.sbatch" "${MODE}"
    ;;
  *)
    echo "Uso: $0 {prepare-cohort|validate-cohort|preflight|low-shot-unit|score-unit|validate-scores|report|validate-report}" >&2
    exit 2
    ;;
esac
