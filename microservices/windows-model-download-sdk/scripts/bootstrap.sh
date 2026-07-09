#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DOWNLOAD_DIR="$(cd "${SDK_ROOT}/../model-download" && pwd)"
COMPOSE_FILE="${MODEL_DOWNLOAD_DIR}/docker/compose.yaml"
VENV_DIR="${SDK_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REGISTRY="${REGISTRY:-}"
TAG="${TAG:-latest}"
ENABLED_PLUGINS="${ENABLED_PLUGINS:-all}"
MODEL_PATH="${MODEL_PATH:-${HOME}/models}"
GETI_HOST="${GETI_HOST:-}"
GETI_TOKEN="${GETI_TOKEN:-}"
GETI_WORKSPACE_ID="${GETI_WORKSPACE_ID:-}"
HLS_3D_POSE_CHECKPOINT_URL="${HLS_3D_POSE_CHECKPOINT_URL:-}"
HLS_ECG_BASE_URL="${HLS_ECG_BASE_URL:-}"
HLS_RPPG_MODEL_URL="${HLS_RPPG_MODEL_URL:-}"
HUGGINGFACEHUB_API_TOKEN="${HUGGINGFACEHUB_API_TOKEN:-}"

SKIP_BACKEND=0
SKIP_INSTALL=0
HEALTH_URL="${HEALTH_URL:-http://localhost:8200/health}"
HEALTH_RETRIES="${HEALTH_RETRIES:-60}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-backend)
      SKIP_BACKEND=1
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    -h|--help)
      cat <<'HELP'
Usage: scripts/bootstrap.sh [options]

Options:
  --skip-backend   Do not start model-download backend docker compose
  --skip-install   Do not create/update local Python venv or install SDK
  -h, --help       Show this help message
HELP
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

log() {
  printf '[bootstrap] %s\n' "$1"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    printf '[bootstrap] ERROR: required command not found: %s\n' "$1" >&2
    exit 1
  }
}

require_cmd "${PYTHON_BIN}"

if [[ ${SKIP_BACKEND} -eq 0 ]]; then
  require_cmd docker
  require_cmd curl
  log "Starting model-download backend from ${COMPOSE_FILE}"
  REGISTRY="${REGISTRY}" TAG="${TAG}" ENABLED_PLUGINS="${ENABLED_PLUGINS}" MODEL_PATH="${MODEL_PATH}" GETI_HOST="${GETI_HOST}" GETI_TOKEN="${GETI_TOKEN}" GETI_WORKSPACE_ID="${GETI_WORKSPACE_ID}" HLS_3D_POSE_CHECKPOINT_URL="${HLS_3D_POSE_CHECKPOINT_URL}" HLS_ECG_BASE_URL="${HLS_ECG_BASE_URL}" HLS_RPPG_MODEL_URL="${HLS_RPPG_MODEL_URL}" HUGGINGFACEHUB_API_TOKEN="${HUGGINGFACEHUB_API_TOKEN}" docker compose -f "${COMPOSE_FILE}" up -d --build

  log "Waiting for backend health at ${HEALTH_URL}"
  for ((i=1; i<=HEALTH_RETRIES; i++)); do
    if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
      log "Backend is healthy"
      break
    fi
    if [[ ${i} -eq ${HEALTH_RETRIES} ]]; then
      printf '[bootstrap] ERROR: Backend did not become healthy at %s\n' "${HEALTH_URL}" >&2
      exit 1
    fi
    log "[${i}/${HEALTH_RETRIES}] backend not ready yet, retrying in ${HEALTH_INTERVAL}s"
    sleep "${HEALTH_INTERVAL}"
  done
else
  log "Skipping backend startup (--skip-backend)"
fi

if [[ ${SKIP_INSTALL} -eq 0 ]]; then
  log "Preparing virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR}/bin/python" -m pip install -e "${SDK_ROOT}"
else
  log "Skipping SDK install (--skip-install)"
fi

log "Running SDK health check"
"${VENV_DIR}/bin/python" - <<'PY'
import asyncio
from model_download_sdk.client import ModelDownloadSDK

async def main() -> None:
    client = ModelDownloadSDK()
    try:
        print(await client.health_check())
    finally:
        await client.close()

asyncio.run(main())
PY

log "Bootstrap completed"
log "Try: source ${VENV_DIR}/bin/activate && model-download jobs"
