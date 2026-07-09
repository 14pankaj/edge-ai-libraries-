#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SDK_ROOT}/.." && pwd)"

OPENAPI_SRC_DEFAULT="${REPO_ROOT}/microservices/model-download/docs/user-guide/_assets/openapi.yaml"
OPENAPI_SRC="${OPENAPI_SRC:-${OPENAPI_SRC_DEFAULT}}"
FIXED_OPENAPI="${SDK_ROOT}/generated/openapi.fixed.yaml"
GENERATED_ROOT="${SDK_ROOT}/generated"
TARGET_CLIENT_DIR="${GENERATED_ROOT}/model_download_service_api_client"
TMP_DIR="${SDK_ROOT}/.tmp-regeneration"

RUN_TESTS="${RUN_TESTS:-1}"
BUILD_PACKAGE="${BUILD_PACKAGE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() {
  printf '[regenerate-sdk] %s\n' "$1"
}

fail() {
  printf '[regenerate-sdk] ERROR: %s\n' "$1" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

ensure_python_module() {
  local module_name="$1"
  local install_name="$2"
  if ! "${PYTHON_BIN}" -c "import ${module_name}" >/dev/null 2>&1; then
    log "Installing missing Python dependency: ${install_name}"
    "${PYTHON_BIN}" -m pip install "${install_name}"
  fi
}

discover_generated_client_dir() {
  local search_root="$1"
  local flat_layout
  local legacy_layout
  local fallback

  # Preferred (new) layout:
  #   .tmp-regeneration/model_download_service_api_client
  flat_layout="${search_root}/model_download_service_api_client"
  if [[ -d "${flat_layout}" && -f "${flat_layout}/__init__.py" ]]; then
    printf '%s' "${flat_layout}"
    return
  fi

  # Backward-compatible (old) layout:
  #   .tmp-regeneration/model-download-service-api-client/model_download_service_api_client
  legacy_layout="${search_root}/model-download-service-api-client/model_download_service_api_client"
  if [[ -d "${legacy_layout}" && -f "${legacy_layout}/__init__.py" ]]; then
    printf '%s' "${legacy_layout}"
    return
  fi

  # Fallback: find any matching package path under temp root.
  fallback="$(find "${search_root}" -mindepth 1 -maxdepth 3 -type d -name model_download_service_api_client | head -n 1 || true)"
  printf '%s' "${fallback}"
}

validate_generated_package() {
  [[ -d "${TARGET_CLIENT_DIR}" ]] || fail "Generated package directory missing: ${TARGET_CLIENT_DIR}"
  [[ -f "${TARGET_CLIENT_DIR}/__init__.py" ]] || fail "Missing __init__.py in generated package: ${TARGET_CLIENT_DIR}"

  # Import validation uses PYTHONPATH pointing to generated/ so it works
  # immediately after file replacement, before `pip install -e .` is re-run.
  PYTHONPATH="${GENERATED_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
import importlib
importlib.import_module("model_download_service_api_client")
importlib.import_module("model_download_service_api_client.client")
print("Generated package import validation passed")
PY
}

run_tests() {
  cd "${SDK_ROOT}"
  if [[ ! -d tests ]]; then
    log "No tests directory found. Skipping tests."
    return
  fi

  if ! find tests -type f | grep -q .; then
    log "Tests directory is empty. Skipping tests."
    return
  fi

  ensure_python_module pytest pytest
  log "Running tests"
  "${PYTHON_BIN}" -m pytest tests -q
}

build_package() {
  cd "${SDK_ROOT}"
  ensure_python_module build build
  log "Building SDK package (sdist + wheel)"
  "${PYTHON_BIN}" -m build .
}

main() {
  require_cmd "${PYTHON_BIN}"
  ensure_python_module yaml pyyaml

  [[ -f "${OPENAPI_SRC}" ]] || fail "OpenAPI source file not found: ${OPENAPI_SRC}"

  rm -rf "${TMP_DIR}"
  mkdir -p "${TMP_DIR}" "${GENERATED_ROOT}"

  log "Fixing OpenAPI schema issues"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/fix_openapi.py" \
    --input "${OPENAPI_SRC}" \
    --output "${FIXED_OPENAPI}"

  ensure_python_module openapi_python_client openapi-python-client

  log "Generating Python client from fixed OpenAPI"
  pushd "${TMP_DIR}" >/dev/null
  "${PYTHON_BIN}" -m openapi_python_client generate --path "${FIXED_OPENAPI}" --meta none
  local generated_dir
  generated_dir="$(discover_generated_client_dir "${TMP_DIR}")"
  popd >/dev/null

  [[ -n "${generated_dir}" ]] || fail "Could not locate generated client directory in ${TMP_DIR}"
  log "Detected generated client path: ${generated_dir}"
  log "Destination generated client path: ${TARGET_CLIENT_DIR}"

  log "Replacing generated client code only (preserving handwritten SDK code)"
  rm -rf "${TARGET_CLIENT_DIR}"
  mkdir -p "${GENERATED_ROOT}"
  [[ -d "${generated_dir}" ]] || fail "Generated output missing model_download_service_api_client package at ${generated_dir}"
  mv "${generated_dir}" "${TARGET_CLIENT_DIR}"

  validate_generated_package

  rm -rf "${TMP_DIR}"

  if [[ "${RUN_TESTS}" == "1" ]]; then
    run_tests
  else
    log "RUN_TESTS=0 set. Skipping tests."
  fi

  if [[ "${BUILD_PACKAGE}" == "1" ]]; then
    build_package
  else
    log "BUILD_PACKAGE=0 set. Skipping package build."
  fi

  log "SDK regeneration pipeline completed successfully"
  log "Generated client location: ${TARGET_CLIENT_DIR}"
}

main "$@"