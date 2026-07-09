"""Tests for download request/model request construction and serialization."""

from __future__ import annotations

from model_download_service_api_client.models.model_download_request import ModelDownloadRequest
from model_download_service_api_client.models.model_hub import ModelHub as GeneratedModelHub
from model_download_service_api_client.models.model_request import ModelRequest
from model_download_service_api_client.models.model_type import ModelType as GeneratedModelType

from model_download_sdk.models import ModelHub, ModelSpec, ModelType


def test_model_spec_creation_from_strings() -> None:
    """SDK ModelSpec should normalize string values to enums."""
    spec = ModelSpec(name="resnet50", hub="huggingface", type_="vision")

    assert spec.name == "resnet50"
    assert spec.hub == ModelHub.HUGGINGFACE
    assert spec.type_ == ModelType.VISION


def test_model_request_to_dict_round_trip() -> None:
    """Generated ModelRequest should serialize and deserialize correctly."""
    request = ModelRequest(
        name="microsoft/Phi-3.5-mini-instruct",
        hub=GeneratedModelHub.HUGGINGFACE,
        type_=GeneratedModelType.LLM,
        is_ovms=True,
        revision="main",
    )

    serialized = request.to_dict()
    restored = ModelRequest.from_dict(serialized)

    assert serialized["name"] == "microsoft/Phi-3.5-mini-instruct"
    assert serialized["hub"] == "huggingface"
    assert serialized["type"] == "llm"
    assert serialized["is_ovms"] is True
    assert serialized["revision"] == "main"

    assert restored.name == request.name
    assert restored.hub == request.hub
    assert restored.type_ == request.type_
    assert restored.is_ovms == request.is_ovms


def test_model_download_request_creation() -> None:
    """Generated ModelDownloadRequest should contain nested model requests."""
    model_a = ModelRequest(name="resnet50", hub=GeneratedModelHub.HUGGINGFACE)
    model_b = ModelRequest(name="yolov8n", hub=GeneratedModelHub.ULTRALYTICS)

    request = ModelDownloadRequest(models=[model_a, model_b], parallel_downloads=False)
    payload = request.to_dict()

    assert "models" in payload
    assert len(payload["models"]) == 2
    assert payload["models"][0]["name"] == "resnet50"
    assert payload["models"][1]["name"] == "yolov8n"
    assert payload["parallel_downloads"] is False


def test_model_download_request_round_trip() -> None:
    """ModelDownloadRequest should round-trip through dict serialization."""
    request = ModelDownloadRequest(
        models=[
            ModelRequest(name="resnet50", hub=GeneratedModelHub.HUGGINGFACE),
        ],
        parallel_downloads=True,
    )

    restored = ModelDownloadRequest.from_dict(request.to_dict())

    assert len(restored.models) == 1
    assert restored.models[0].name == "resnet50"
    assert restored.parallel_downloads is True
