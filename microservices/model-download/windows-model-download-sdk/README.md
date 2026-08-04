# Windows Model Download SDK

A Python SDK and CLI for interacting with the Model Download Service.

The SDK wraps generated OpenAPI client code with stable, user-friendly APIs for:

- Downloading models from supported hubs
- Polling async jobs with timeout and backoff
- Listing plugins, jobs, and results
- Handling errors through SDK-specific exceptions
- Regenerating the OpenAPI client safely

---

## Prerequisites

- Python 3.10+
- Docker (required only if bootstrap starts the backend — not needed if service is already running)

---

## Setup — Choose a Path

### Path A: Bootstrap (recommended for fresh setup)

Bootstrap handles everything: starts the backend if not already running, creates the venv, installs the SDK, and runs a health check.

**Linux/macOS:**

```bash
./scripts/bootstrap.sh
```

**Windows PowerShell:**

```powershell
./scripts/bootstrap.ps1
```

**Make:**

```bash
make bootstrap
```

Bootstrap skips backend startup automatically if the service is already healthy. To force-skip it explicitly:

```bash
./scripts/bootstrap.sh --skip-backend   # service already running elsewhere
./scripts/bootstrap.sh --skip-install   # venv already exists
```

**Enabled plugins** (default: `huggingface`). To enable more:

```bash
ENABLED_PLUGINS=huggingface,openvino ./scripts/bootstrap.sh
```

**Hugging Face token** (required for gated models):

```bash
HUGGINGFACEHUB_API_TOKEN=hf_xxx ./scripts/bootstrap.sh
# or export before running:
export HUGGINGFACEHUB_API_TOKEN=hf_xxx
```

---

### Path B: SDK only (service already running)

Use this when the backend is already running (e.g. via `run_service.sh` or a remote host).

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -e .
```

This installs:
- Python package: `model_download_sdk`
- CLI command: `model-download`

Point the SDK at a non-local service:

```python
from model_download_sdk.client import ModelDownloadSDK, SDKConfig
client = ModelDownloadSDK(SDKConfig(base_url="http://<host>:8200"))
```

---

## Other Make Targets

```bash
make backend-up     # start backend only
make backend-down   # stop backend
make backend-logs   # tail backend logs
make install        # install SDK into .venv
make health         # run SDK health check
make test           # run test suite
```

---

## Quick Start (Python)

```python
import asyncio
from model_download_sdk.client import ModelDownloadSDK


async def main() -> None:
    client = ModelDownloadSDK()
    try:
        health = await client.health_check()
        print("Health:", health)

        result = await client.download_model(
            model_name="bert-base-uncased",
            hub="huggingface",
            download_path="/tmp/models",
            wait=False,
        )
        print("Download result:", result)
    finally:
        await client.close()


asyncio.run(main())
```

---

## CLI Usage

```bash
model-download health
model-download plugins
model-download jobs
model-download download --model-name bert-base-uncased --hub huggingface --download-path /tmp/models
```

If the command is not found, re-activate your virtual environment and reinstall with `pip install -e .`.

---

## Run Tests

```bash
python -m pytest -q
```

To run only integration tests, set environment variables/markers according to test docs and local service availability.

---

## Regenerate Generated OpenAPI Client

```bash
PYTHON_BIN=/absolute/path/to/windows-model-download-sdk/.venv/bin/python \
RUN_TESTS=0 BUILD_PACKAGE=0 scripts/regenerate_sdk.sh
```

Generated client output is copied into:

- `generated/model_download_service_api_client`

---

## Project Structure

| Path | Purpose |
| --- | --- |
| `model_download_sdk/` | Handwritten SDK package |
| `generated/model_download_service_api_client/` | Generated OpenAPI client package |
| `tests/` | Unit and integration tests |
| `scripts/bootstrap.sh` | Linux/macOS bootstrap (backend + venv + install + health) |
| `scripts/bootstrap.ps1` | Windows PowerShell bootstrap |
| `scripts/fix_openapi.py` | OpenAPI normalization/fix pipeline |
| `scripts/regenerate_sdk.sh` | End-to-end SDK regeneration pipeline |
| `Makefile` | Convenience targets for bootstrap, backend, health, and tests |
