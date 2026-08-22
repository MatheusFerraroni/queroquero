#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-}"
shift || true

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

SBATCH_ARGS=(
  --parsable
  --partition="${PREPARATION_PARTITION:-l40s}"
  --time=1-00:00:00
  --export=ALL
)

case "${MODE}" in
  capacity)
    DATASET_ID="${1:-}"
    CANDIDATE_DOCUMENTS="${2:-}"
    case "${DATASET_ID}" in
      adrenaline|brwac|gigaverbo|multiwoz_ptbr|outerspace|wackywacky) ;;
      *)
        echo "Erro: dataset inválido para auditoria pareada." >&2
        exit 2
        ;;
    esac
    if [[ ! "${CANDIDATE_DOCUMENTS}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Erro: candidate_documents deve ser um inteiro positivo." >&2
      exit 2
    fi
    sbatch "${SBATCH_ARGS[@]}" \
      "${PROJECT_DIR}/scripts/prepare_paired_real.sbatch" \
      capacity "${DATASET_ID}" "${CANDIDATE_DOCUMENTS}"
    ;;
  prepare-all)
    for dataset in adrenaline brwac gigaverbo multiwoz_ptbr outerspace wackywacky; do
      if ! grep -q '"paired_real"' "configs/datasets/${dataset}.json"; then
        echo "Erro: profile paired_real ausente em ${dataset}." >&2
        exit 2
      fi
    done
    sbatch "${SBATCH_ARGS[@]}" --array=0-5%1 \
      "${PROJECT_DIR}/scripts/prepare_paired_real.sbatch" prepare
    ;;
  verify)
    for config in \
      configs/training/l40s-real-general.json \
      configs/training/l40s-real-forum-tech.json; do
      if [[ ! -f "${config}" ]]; then
        echo "Erro: config pareado ausente: ${config}" >&2
        exit 2
      fi
    done
    sbatch "${SBATCH_ARGS[@]}" \
      "${PROJECT_DIR}/scripts/prepare_paired_real.sbatch" verify
    ;;
  *)
    echo "Uso: $0 {capacity <dataset> <candidate_documents>|prepare-all|verify}" >&2
    exit 2
    ;;
esac
