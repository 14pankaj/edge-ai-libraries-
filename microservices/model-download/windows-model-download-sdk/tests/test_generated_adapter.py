"""Tests for generated adapter enum conversion and validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from model_download_sdk._generated_adapter import GeneratedClientWrapper
from model_download_sdk._generated_adapter import generated_job_to_sdk
from model_download_sdk.exceptions import ValidationError


@pytest.mark.parametrize(
    ("hub", "type_"),
    [
        ("huggingface", "llm"),
        ("ollama", "vision"),
        ("geti", "embeddings"),
        ("ultralytics", None),
    ],
)
def test_call_download_models_converts_hub_and_type_to_generated_enums(hub: str, type_: str | None) -> None:
    """Adapter should convert hub/type_ strings to generated enum objects."""
    wrapper = GeneratedClientWrapper(client=object())

    models = [{"name": "bert-base-uncased", "hub": hub, "type_": type_, "is_ovms": False}]

    with patch("model_download_service_api_client.api.models.download_models.sync") as sync_mock:
        sync_mock.return_value = {"ok": True}

        result = wrapper.call_download_models(models, download_path="/tmp/models")

        assert result == {"ok": True}
        kwargs = sync_mock.call_args.kwargs
        body = kwargs["body"]
        model = body.models[0]

        assert hasattr(model.hub, "value")
        assert model.hub.value == hub

        if type_ is None:
            # Generated model uses UNSET by default when type_ omitted.
            assert not hasattr(model.type_, "value")
        else:
            assert hasattr(model.type_, "value")
            assert model.type_.value == type_


def test_call_download_models_raises_validation_error_for_unknown_hub() -> None:
    """Adapter should raise clear ValidationError for unsupported hub values."""
    wrapper = GeneratedClientWrapper(client=object())
    models = [{"name": "bert-base-uncased", "hub": "unknown-hub", "type_": "llm"}]

    with pytest.raises(ValidationError, match="Unknown hub"):
        wrapper.call_download_models(models, download_path="/tmp/models")


def test_call_download_models_raises_validation_error_for_unknown_type() -> None:
    """Adapter should raise clear ValidationError for unsupported type values."""
    wrapper = GeneratedClientWrapper(client=object())
    models = [{"name": "bert-base-uncased", "hub": "huggingface", "type_": "unknown-type"}]

    with pytest.raises(ValidationError, match="Unknown type"):
        wrapper.call_download_models(models, download_path="/tmp/models")


def test_generated_job_to_sdk_handles_generated_unset_values() -> None:
    """Generated Unset sentinel values should not leak into SDK Job fields."""
    from model_download_service_api_client.types import UNSET as GENERATED_UNSET

    class _FakeGeneratedJob:
        job_id = "job-123"
        operation_type = "download"
        model_name = "bert-base-uncased"
        status = "pending"
        hub = "huggingface"
        creation_time = "2026-01-01T00:00:00+00:00"
        completion_time = GENERATED_UNSET
        error = GENERATED_UNSET
        output_dir = GENERATED_UNSET
        plugin_name = GENERATED_UNSET

    job = generated_job_to_sdk(_FakeGeneratedJob())

    assert job.id == "job-123"
    assert job.output_directory is None
    assert job.error is None
    assert job.plugin is None
    assert job.completed_at is None
