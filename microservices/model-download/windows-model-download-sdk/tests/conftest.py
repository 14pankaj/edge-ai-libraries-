"""Shared pytest fixtures for Model Download SDK tests."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from model_download_sdk.client import ModelDownloadSDK, SDKConfig
from model_download_sdk.models import Job, JobOperationType, JobStatus


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: marks tests requiring live model-download service")


@pytest.fixture
def sdk_config() -> SDKConfig:
    """Return default test SDK configuration."""
    return SDKConfig(base_url="http://localhost:8200", timeout=5.0)


@pytest.fixture
def sdk_client(sdk_config: SDKConfig) -> ModelDownloadSDK:
    """Return SDK client instance for unit tests."""
    return ModelDownloadSDK(config=sdk_config)


@pytest.fixture
def sample_job() -> Job:
    """Return a processing job object for polling-related tests."""
    return Job(
        id="job-123",
        operation=JobOperationType.DOWNLOAD,
        model_name="resnet50",
        status=JobStatus.PROCESSING,
        created_at=datetime.now(timezone.utc),
        hub="huggingface",
    )


@pytest.fixture
def completed_job(tmp_path: Path) -> Job:
    """Return a completed successful job object."""
    output_dir = tmp_path / "models" / "resnet50"
    output_dir.mkdir(parents=True, exist_ok=True)
    return Job(
        id="job-123",
        operation=JobOperationType.DOWNLOAD,
        model_name="resnet50",
        status=JobStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        hub="huggingface",
        output_directory=str(output_dir),
    )


@pytest.fixture
def integration_enabled() -> bool:
    """Whether integration tests are enabled via environment variable."""
    return os.getenv("RUN_INTEGRATION_TESTS", "0") == "1"
