# Model Download SDK Package

This package provides an async-first Python SDK for interacting with the Model Download Service.

It wraps the generated OpenAPI client with:

- Stable, user-friendly models
- Consistent SDK exception types
- Job polling with timeout and backoff
- Path normalization and filesystem helpers
- Optional CLI commands

## Package Structure

| Module | Purpose |
| ------ | ------- |
| `client.py` | Main SDK client (`ModelDownloadSDK`) and high-level operations |
| `_http_client.py` | Low-level service calls and response normalization |
| `_generated_adapter.py` | Boundary layer between SDK models and generated models |
| `models.py` | SDK data models (`ModelSpec`, `Job`, `DownloadResult`) |
| `exceptions.py` | SDK exception hierarchy |
| `filesystem.py` | Path normalization and writable directory checks |
| `dlstreamer.py` | DLStreamer-focused helper workflow |
| `cli.py` | Typer-based command-line entrypoints |
| `error_mapper.py` | Mapping from transport/generated errors to SDK errors |
| `types.py` | Shared type markers/utilities (`UNSET`, generic `Response`) |

## Installation

From the `windows-model-download-sdk` root directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

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

## CLI Usage

After installation, the `model-download` command is available:

```bash
model-download health
model-download plugins
model-download jobs
model-download download --model-name bert-base-uncased --hub huggingface --download-path /tmp/models
```

## Common Patterns

### Wait for completion

```python
result = await client.download_models(
    models=[...],
    output_directory="/tmp/models",
    wait=True,
    timeout=3600,
)

for job in result.successful_jobs:
    print("OK", job.id, job.output_directory)

for job in result.failed_jobs:
    print("FAILED", job.id, job.error)
```

### Error handling

```python
from model_download_sdk.exceptions import SDKError, ValidationError, ConnectionError

try:
    ...
except ValidationError as exc:
    print("Invalid input:", exc)
except ConnectionError as exc:
    print("Service/network error:", exc)
except SDKError as exc:
    print("SDK error:", exc)
```

## Notes

- Base URL defaults to `http://localhost:8200`.
- The SDK is async-first; use `asyncio.run(...)` from synchronous scripts.
- The generated OpenAPI package is intentionally wrapped so SDK APIs stay stable.
