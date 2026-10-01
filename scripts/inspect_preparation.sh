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
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/cache/uv}"

exec uv run \
  --env-file .env \
  --python .venv/bin/python \
  -m scripts.inspect_preparation "$@"
