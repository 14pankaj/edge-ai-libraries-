"""Verify the generated OpenAPI client is importable after `pip install -e .`.

These tests intentionally do NOT add anything to sys.path and do NOT use any
PYTHONPATH environment variable.  If the package is correctly installed (even
in editable mode) all imports must resolve on their own.
"""

from __future__ import annotations

import importlib


def test_generated_client_root_importable() -> None:
    """Root generated package must import cleanly after editable install."""
    mod = importlib.import_module("model_download_service_api_client")
    assert mod is not None


def test_generated_client_class_importable() -> None:
    """Client class must be accessible from the generated package."""
    from model_download_service_api_client.client import AuthenticatedClient, Client

    assert Client is not None
    assert AuthenticatedClient is not None


def test_generated_models_importable() -> None:
    """Core generated model types must be importable."""
    from model_download_service_api_client.models.model_download_request import ModelDownloadRequest
    from model_download_service_api_client.models.model_request import ModelRequest

    assert ModelRequest is not None
    assert ModelDownloadRequest is not None


def test_generated_api_download_models_importable() -> None:
    """API endpoint module required by SDK adapter must be importable."""
    from model_download_service_api_client.api.models.download_models import sync

    assert sync is not None


def test_sdk_client_importable_alongside_generated() -> None:
    """Both SDK and generated packages must coexist without sys.path tricks."""
    from model_download_sdk.client import ModelDownloadSDK
    from model_download_service_api_client.client import Client

    assert ModelDownloadSDK is not None
    assert Client is not None
