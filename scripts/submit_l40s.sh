#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-}"

case "${MODE}" in
  preflight|smoke-stop|smoke-resume|mvp|mvp-resume|validate) ;;
  *)
    echo "Uso: $0 {preflight|smoke-stop|smoke-resume|mvp|mvp-resume|validate}" >&2
    exit 2
    ;;
esac

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

sbatch \
  --parsable \
  --partition="${L40S_PARTITION:-l40s}" \
  --export=ALL \
  "${PROJECT_DIR}/scripts/train_l40s.sbatch" \
  "${MODE}"
