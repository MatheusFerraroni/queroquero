#!/usr/bin/env bash

#SBATCH --job-name=queroquero-classification
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/classification-%j.out
#SBATCH --error=logs/classification-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-}"
CONFIG="configs/classification/adrenaline-v1.json"

cd "${PROJECT_DIR}"
source "${HOME}/activate_queroquero.sh"

if [[ ! -f .env ]]; then
  echo "Erro: ${PROJECT_DIR}/.env não foi encontrado." >&2
  exit 2
fi
set -a
source .env
set +a

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nó: $(hostname)"
echo "Início: $(date --iso-8601=seconds)"
echo "Modo: ${MODE}"
echo "Python: $(python --version)"

case "${MODE}" in
  build)
    python -m queroquero.classification_data build \
      --config "${CONFIG}"
    ;;
  validate-dataset)
    python -m queroquero.classification_data validate-dataset \
      --path "${2:?dataset canônico ausente}"
    ;;
  split)
    python -m queroquero.classification_data split \
      --config "${CONFIG}" \
      --dataset "${2:?dataset canônico ausente}" \
      --task "${3:?tarefa ausente}" \
      --seed "${4:?seed ausente}" \
      --output "${5:?saída do split ausente}"
    ;;
  validate-split)
    python -m queroquero.classification_data validate-split \
      --config "${CONFIG}" \
      --dataset "${2:?dataset canônico ausente}" \
      --path "${3:?manifesto de split ausente}"
    ;;
  *)
    echo "Uso: $0 {build|validate-dataset <dataset>|split <dataset> <coarse|fine> <seed> <output>|validate-split <dataset> <manifest>}" >&2
    exit 2
    ;;
esac

echo "Fim: $(date --iso-8601=seconds)"
