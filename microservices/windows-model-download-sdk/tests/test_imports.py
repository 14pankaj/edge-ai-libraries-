"""Import and public API exposure tests."""

from __future__ import annotations

import importlib


def test_sdk_package_imports() -> None:
    """SDK root package should import successfully."""
    module = importlib.import_module("model_download_sdk")
    assert module is not None


def test_public_api_symbols_present() -> None:
    """Public API symbols should be exposed from package root."""
    sdk = importlib.import_module("model_download_sdk")

    expected_symbols = {
        "ModelDownloadSDK",
        "ModelSpec",
        "Job",
        "DownloadResult",
        "UploadResult",
        "SDKError",
        "ConnectionError",
        "ValidationError",
        "JobError",
        "TimeoutError",
        "NotFoundError",
        "AuthenticationError",
    }

    for symbol in expected_symbols:
        assert hasattr(sdk, symbol), f"Missing public symbol: {symbol}"


def test_generated_client_models_importable() -> None:
    """Generated OpenAPI client model types should be importable."""
    model_request = importlib.import_module(
        "model_download_service_api_client.models.model_request"
    )
    model_download_request = importlib.import_module(
        "model_download_service_api_client.models.model_download_request"
    )

    assert hasattr(model_request, "ModelRequest")
    assert hasattr(model_download_request, "ModelDownloadRequest")
