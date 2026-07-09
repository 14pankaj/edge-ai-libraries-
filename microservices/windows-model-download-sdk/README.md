# Windows Model Download SDK

A Python SDK and CLI for interacting with the Model Download Service.

The SDK wraps generated OpenAPI client code with stable, user-friendly APIs for:

- Downloading models from supported hubs
- Polling async jobs with timeout and backoff
- Listing plugins, jobs, and results
- Handling errors through SDK-specific exceptions
- Regenerating the OpenAPI client safely

---

## Documentation

| Document | Description |
| --- | --- |
| [Package README](model_download_sdk/README.md) | Module-level SDK usage and package layout |
| [SDK Architecture](SDK_ARCHITECTURE.md) | Design and layering details |
| [Implementation Guide](IMPLEMENTATION_GUIDE.md) | Development notes and implementation flow |

---

## Prerequisites

- Python 3.10+
- Running Model Download service (default: `http://localhost:8200`)
- Docker (for backend bootstrap)

---

## Installation

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs:

- Python package: `model_download_sdk`
- CLI command: `model-download`

---

## One-Command Bootstrap

Use bootstrap scripts if you want backend + SDK venv + install + health check in one flow.

### Linux/macOS

```bash
./scripts/bootstrap.sh
```

Optional flags:

```bash
./scripts/bootstrap.sh --skip-backend
./scripts/bootstrap.sh --skip-install
```

### Windows PowerShell

```powershell
./scripts/bootstrap.ps1
```

Optional flags:

```powershell
./scripts/bootstrap.ps1 -SkipBackend
./scripts/bootstrap.ps1 -SkipInstall
```

### Make Targets

```bash
make bootstrap
make backend-up
make install
make health
make test
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
