"""Job polling and timeout behavior tests."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from model_download_sdk.exceptions import ConnectionError as SDKConnectionError
from model_download_sdk.exceptions import TimeoutError as SDKTimeoutError
from model_download_sdk.models import JobStatus


INTEGRATION_ENABLED = os.getenv("RUN_INTEGRATION_TESTS", "0") == "1"


class _FakeDateTime:
    """Test helper to control elapsed time in wait_for_job."""

    _current = datetime(2026, 1, 1, 0, 0, 0)

    @classmethod
    def now(cls) -> datetime:
        cls._current += timedelta(seconds=2)
        return cls._current


def test_wait_for_job_success_returns_completed_job(
    sdk_client,
    sample_job,
    completed_job,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wait_for_job should return once a completed job is observed."""
    processing = sample_job
    processing.status = JobStatus.PROCESSING

    fake_get_job = AsyncMock(side_effect=[processing, completed_job])
    monkeypatch.setattr(sdk_client, "get_job", fake_get_job)

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("model_download_sdk.client.asyncio.sleep", no_sleep)

    result = asyncio.run(sdk_client.wait_for_job("job-123", poll_interval=0.01, timeout=10))

    assert result.status == JobStatus.COMPLETED
    assert result.is_success is True


def test_wait_for_job_timeout_raises(
    sdk_client,
    sample_job,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wait_for_job should raise SDKTimeoutError when timeout is exceeded."""
    sample_job.status = JobStatus.PROCESSING
    monkeypatch.setattr(sdk_client, "get_job", AsyncMock(return_value=sample_job))
    monkeypatch.setattr("model_download_sdk.client.datetime", _FakeDateTime)

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("model_download_sdk.client.asyncio.sleep", no_sleep)

    with pytest.raises(SDKTimeoutError):
        asyncio.run(sdk_client.wait_for_job("job-123", poll_interval=0.01, timeout=1))


def test_wait_for_job_propagates_job_error(
    sdk_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wait_for_job should wrap unexpected polling errors as SDKConnectionError."""

    async def raising_get_job(_: str):
        raise ValueError("boom")

    monkeypatch.setattr(sdk_client, "get_job", raising_get_job)

    with pytest.raises(SDKConnectionError):
        asyncio.run(sdk_client.wait_for_job("job-123", poll_interval=0.01, timeout=2))


@pytest.mark.integration
@pytest.mark.skipif(
    not INTEGRATION_ENABLED,
    reason="Set RUN_INTEGRATION_TESTS=1 to run live service integration examples.",
)
def test_integration_wait_for_job_example() -> None:
    """Example integration test for job polling with a real service."""
    from model_download_sdk.client import ModelDownloadSDK

    client = ModelDownloadSDK()
    try:
        # Replace with a real job ID when enabling integration tests.
        asyncio.run(client.wait_for_job(job_id="replace-with-real-job-id", timeout=5))
    finally:
        asyncio.run(client.close())
