#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Erro: uv não foi encontrado no PATH." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "Erro: ${PROJECT_DIR}/.env não foi encontrado." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Erro: ${PROJECT_DIR}/.venv/bin/python não foi encontrado." >&2
  echo "Crie o ambiente com: uv venv --python 3.12 .venv" >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/cache/uv}"

datasets=(
  brwac
  wackywacky
  multiwoz_ptbr
  outerspace
  adrenaline
  gigaverbo
)

failed_datasets=()

for dataset in "${datasets[@]}"; do
  echo
  echo "===== Iniciando: ${dataset} ====="

  if time -p uv run \
    --env-file .env \
    --python .venv/bin/python \
    -m queroquero.prepare run \
    --dataset "${dataset}" \
    --profile smoke
  then
    echo "===== Concluído: ${dataset} ====="
  else
    exit_code=$?
    failed_datasets+=("${dataset}")
    echo "===== Falhou: ${dataset} (código ${exit_code}) =====" >&2
  fi
done

echo
if (( ${#failed_datasets[@]} > 0 )); then
  echo "Datasets com falha: ${failed_datasets[*]}" >&2
  exit 1
fi

echo "Todos os datasets smoke foram concluídos."
