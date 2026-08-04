"""DLStreamer helper workflow tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from model_download_sdk.dlstreamer import DLStreamerClient, DLStreamerConfig
from model_download_sdk.exceptions import NotFoundError, SDKError, ValidationError
from model_download_sdk.models import DownloadResult


def test_dlstreamer_config_validation() -> None:
    """DLStreamerConfig should validate supported devices and precisions."""
    cfg = DLStreamerConfig(device="CPU", model_precision="FP16")
    assert cfg.device == "CPU"
    assert cfg.model_precision == "FP16"

    with pytest.raises(ValueError):
        DLStreamerConfig(device="INVALID", model_precision="FP16")

    with pytest.raises(ValueError):
        DLStreamerConfig(device="CPU", model_precision="BF16")


def test_pull_for_dlstreamer_requires_bound_sdk() -> None:
    """Calling pull_for_dlstreamer without bound SDK should fail."""
    helper = DLStreamerClient()

    with pytest.raises(SDKError):
        asyncio.run(helper.pull_for_dlstreamer(model_name="resnet50", hub="huggingface"))


def test_pull_for_dlstreamer_validates_input(sdk_client) -> None:
    """Input validation errors should be raised for invalid values."""
    helper = DLStreamerClient(sdk_client)

    with pytest.raises(ValidationError):
        asyncio.run(helper.pull_for_dlstreamer(model_name="", hub="huggingface"))

    with pytest.raises(ValidationError):
        asyncio.run(helper.pull_for_dlstreamer(model_name="resnet50", hub=""))

    with pytest.raises(ValidationError):
        asyncio.run(helper.pull_for_dlstreamer(model_name="resnet50", hub="huggingface", poll_interval=0))


def test_pull_for_dlstreamer_success_prefers_existing_output(tmp_path: Path, sdk_client, completed_job, monkeypatch: pytest.MonkeyPatch) -> None:
    """Workflow should return first existing resolved output path."""
    helper = DLStreamerClient(sdk_client)

    output_dir = tmp_path / "models" / "resnet50"
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_job.output_directory = str(output_dir)
    download_result = DownloadResult(
        job_ids=["job-123"],
        message="submitted",
        successful_jobs=[],
        failed_jobs=[],
    )
    download_result.output_directory = str(tmp_path / "models")

    monkeypatch.setattr(sdk_client, "download_model", AsyncMock(return_value=download_result))
    monkeypatch.setattr(sdk_client, "wait_for_job", AsyncMock(return_value=completed_job))

    resolved = asyncio.run(helper.pull_for_dlstreamer(model_name="resnet50", hub="huggingface", download_path=str(tmp_path / "models")))

    assert isinstance(resolved, Path)
    assert resolved == output_dir.resolve(strict=False)


def test_pull_for_dlstreamer_missing_output_raises(tmp_path: Path, sdk_client, completed_job, monkeypatch: pytest.MonkeyPatch) -> None:
    """Workflow should raise NotFoundError when no candidate output path exists."""
    helper = DLStreamerClient(sdk_client)

    completed_job.output_directory = str(tmp_path / "does-not-exist")
    download_result = DownloadResult(
        job_ids=["job-123"],
        message="submitted",
        successful_jobs=[],
        failed_jobs=[],
    )
    download_result.output_directory = str(tmp_path / "also-missing")

    monkeypatch.setattr(sdk_client, "download_model", AsyncMock(return_value=download_result))
    monkeypatch.setattr(sdk_client, "wait_for_job", AsyncMock(return_value=completed_job))

    with pytest.raises(NotFoundError):
        asyncio.run(helper.pull_for_dlstreamer(model_name="resnet50", hub="huggingface", download_path=str(tmp_path / "models")))
