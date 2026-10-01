#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-}"

if [[ -z "${P100_PARTITION:-}" ]]; then
  echo "Erro: defina P100_PARTITION com a partição atual da P100." >&2
  exit 2
fi

if [[ -z "${P100_NODE:-}" ]]; then
  echo "Erro: defina P100_NODE com o nó atual da P100." >&2
  exit 2
fi

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
  --partition="${P100_PARTITION}" \
  --nodelist="${P100_NODE}" \
  --export=ALL \
  "${PROJECT_DIR}/scripts/train_p100.sbatch" \
  "${MODE}"
