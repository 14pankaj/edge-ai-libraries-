"""Unit tests for main SDK client behavior with mocked external dependencies."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from model_download_sdk.client import ModelDownloadSDK, SDKConfig
from model_download_sdk.exceptions import ConnectionError as SDKConnectionError
from model_download_sdk.exceptions import JobError, ValidationError
from model_download_sdk.exceptions import TimeoutError as SDKTimeoutError
from model_download_sdk.models import DownloadResult, Job, JobOperationType, JobStatus, ModelSpec


INTEGRATION_ENABLED = os.getenv("RUN_INTEGRATION_TESTS", "0") == "1"


def test_client_initialization_defaults() -> None:
    """Client should initialize with default config and DLStreamer helper."""
    client = ModelDownloadSDK()

    assert isinstance(client.config, SDKConfig)
    assert client.config.base_url == "http://localhost:8200"
    assert client.dlstreamer is not None


def test_client_lazy_http_client_creation(sdk_client: ModelDownloadSDK, monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP client should be created lazily only on first access."""
    fake_http_client = Mock(name="http_client")

    def fake_ctor(*args, **kwargs):
        return fake_http_client

    monkeypatch.setattr("model_download_sdk.client.ModelDownloadClient", fake_ctor)

    assert sdk_client._http_client is None
    first = sdk_client._get_http_client()
    second = sdk_client._get_http_client()

    assert first is fake_http_client
    assert second is fake_http_client


def test_health_check_success(sdk_client: ModelDownloadSDK, monkeypatch: pytest.MonkeyPatch) -> None:
    """health_check should return health payload from HTTP client."""
    http_client = Mock()
    http_client.health_check = Mock(return_value={"status": "healthy", "message": "ok"})

    monkeypatch.setattr(sdk_client, "_get_http_client", lambda: http_client)

    result = asyncio.run(sdk_client.health_check())

    assert result["status"] == "healthy"


def test_download_model_delegates_to_download_models(sdk_client: ModelDownloadSDK, monkeypatch: pytest.MonkeyPatch) -> None:
    """download_model should construct ModelSpec and delegate without duplicating logic."""
    captured: dict[str, object] = {}

    async def fake_download_models(
        models: list[ModelSpec],
        output_directory: str,
        wait: bool,
        timeout: int | None,
    ) -> DownloadResult:
        captured["models"] = models
        captured["output_directory"] = output_directory
        captured["wait"] = wait
        captured["timeout"] = timeout
        return DownloadResult(job_ids=["job-1"], message="ok", successful_jobs=[], failed_jobs=[])

    monkeypatch.setattr(sdk_client, "download_models", fake_download_models)

    result = asyncio.run(
        sdk_client.download_model(
            model_name="resnet50",
            hub="hf",
            download_path="models",
            model_type="image",
            convert_to_openvino=True,
            wait=True,
            timeout=123,
        )
    )

    assert result.job_ids == ["job-1"]
    models = captured["models"]
    assert isinstance(models, list)
    assert len(models) == 1
    spec = models[0]
    assert isinstance(spec, ModelSpec)
    assert spec.name == "resnet50"
    assert spec.hub.value == "huggingface"
    assert spec.type_.value == "vision"
    assert spec.convert_to_openvino is True
    assert captured["output_directory"] == "models"
    assert captured["wait"] is True
    assert captured["timeout"] == 123


def test_download_model_validates_hub(sdk_client: ModelDownloadSDK) -> None:
    """Invalid hub values should raise ValidationError."""
    with pytest.raises(ValidationError):
        asyncio.run(
            sdk_client.download_model(
                model_name="resnet50",
                hub="invalid-hub",
                download_path="models",
            )
        )


def test_pull_for_dlstreamer_delegates_to_helper(sdk_client: ModelDownloadSDK, monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-level pull_for_dlstreamer should delegate to DLStreamer helper method."""
    expected = Path("/tmp/fake")
    fake_pull = AsyncMock(return_value=expected)
    monkeypatch.setattr(sdk_client.dlstreamer, "pull_for_dlstreamer", fake_pull)

    result = asyncio.run(sdk_client.pull_for_dlstreamer(model_name="resnet50", hub="huggingface"))

    assert result == expected
    fake_pull.assert_awaited_once()


def test_get_job_failed_status_raises_job_error(sdk_client: ModelDownloadSDK, sample_job, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_job should raise JobError when backend reports failed status."""
    failed = sample_job
    failed.status = JobStatus.FAILED
    failed.error = "download failed"

    http_client = Mock()
    http_client.get_job_status = Mock(return_value=failed)

    monkeypatch.setattr(sdk_client, "_get_http_client", lambda: http_client)

    with pytest.raises(JobError):
        asyncio.run(sdk_client.get_job("job-123"))


def test_download_models_wait_timeout_returns_failed_job(
    sdk_client: ModelDownloadSDK,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout while polling should not crash; failed_jobs should contain valid Job."""
    http_client = Mock()
    http_client.download_models = Mock(return_value=["job-timeout"])
    http_client.get_job_status = Mock(
        return_value=Job(
            id="job-timeout",
            operation=JobOperationType.DOWNLOAD,
            model_name="resnet50",
            status=JobStatus.PROCESSING,
            created_at=datetime.now(timezone.utc),
            hub="huggingface",
            output_directory=str(tmp_path),
            plugin="huggingface",
        )
    )
    monkeypatch.setattr(sdk_client, "_get_http_client", lambda: http_client)
    monkeypatch.setattr(
        sdk_client,
        "wait_for_job",
        AsyncMock(
            side_effect=SDKTimeoutError(
                "timed out",
                timeout_seconds=1,
                operation="wait_for_job(job-timeout)",
            )
        ),
    )

    result = asyncio.run(
        sdk_client.download_models(
            models=[ModelSpec(name="resnet50", hub="huggingface")],
            output_directory=str(tmp_path),
            wait=True,
            timeout=1,
        )
    )

    assert result.failed_count == 1
    failed = result.failed_jobs[0]
    assert failed.id == "job-timeout"
    assert failed.status == JobStatus.FAILED
    assert failed.model_name == "resnet50"
    assert failed.hub == "huggingface"
    assert failed.created_at is not None
    assert "timed out" in (failed.error or "")


def test_download_models_wait_connection_error_returns_failed_job_with_spec_metadata(
    sdk_client: ModelDownloadSDK,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection error during polling should return failed_jobs and keep request metadata."""
    http_client = Mock()
    http_client.download_models = Mock(return_value=["job-conn"])
    http_client.get_job_status = Mock(side_effect=RuntimeError("status unavailable"))
    monkeypatch.setattr(sdk_client, "_get_http_client", lambda: http_client)
    monkeypatch.setattr(
        sdk_client,
        "wait_for_job",
        AsyncMock(side_effect=SDKConnectionError("polling connection error")),
    )

    result = asyncio.run(
        sdk_client.download_models(
            models=[ModelSpec(name="bert-base-uncased", hub="huggingface")],
            output_directory=str(tmp_path),
            wait=True,
            timeout=1,
        )
    )

    assert result.failed_count == 1
    failed = result.failed_jobs[0]
    assert failed.id == "job-conn"
    assert failed.status == JobStatus.FAILED
    assert failed.model_name == "bert-base-uncased"
    assert failed.hub == "huggingface"
    assert "polling connection error" in (failed.error or "")


def test_download_models_wait_generic_polling_failure_returns_failed_job(
    sdk_client: ModelDownloadSDK,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected polling failures should be captured in DownloadResult.failed_jobs."""
    http_client = Mock()
    http_client.download_models = Mock(return_value=["job-generic"])
    http_client.get_job_status = Mock(
        return_value=Job(
            id="job-generic",
            operation=JobOperationType.DOWNLOAD,
            model_name="my-model",
            status=JobStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            hub="ollama",
        )
    )
    monkeypatch.setattr(sdk_client, "_get_http_client", lambda: http_client)
    monkeypatch.setattr(
        sdk_client,
        "wait_for_job",
        AsyncMock(side_effect=RuntimeError("polling exploded")),
    )

    result = asyncio.run(
        sdk_client.download_models(
            models=[ModelSpec(name="my-model", hub="ollama")],
            output_directory=str(tmp_path),
            wait=True,
            timeout=1,
        )
    )

    assert result.failed_count == 1
    failed = result.failed_jobs[0]
    assert failed.id == "job-generic"
    assert failed.status == JobStatus.FAILED
    assert failed.model_name == "my-model"
    assert failed.hub == "ollama"
    assert "polling exploded" in (failed.error or "")


@pytest.mark.integration
@pytest.mark.skipif(
    not INTEGRATION_ENABLED,
    reason="Set RUN_INTEGRATION_TESTS=1 to run live service integration examples.",
)
def test_integration_health_check_example() -> None:
    """Example integration test against a live model-download service."""
    client = ModelDownloadSDK()
    try:
        result = asyncio.run(client.health_check())
        assert "status" in result
    finally:
        asyncio.run(client.close())
