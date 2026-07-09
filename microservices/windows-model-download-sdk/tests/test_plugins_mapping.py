"""Tests for plugin response mapping from generated /plugins payloads."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import Mock

import pytest

from model_download_sdk._http_client import ModelDownloadClient
from model_download_sdk.client import ModelDownloadSDK
from model_download_service_api_client.models.plugins_response import PluginsResponse


INTEGRATION_ENABLED = os.getenv("RUN_INTEGRATION_TESTS", "0") == "1"


def _build_http_client_with_response(response_obj: object) -> ModelDownloadClient:
    """Create a ModelDownloadClient instance with a mocked wrapper response."""
    client = ModelDownloadClient.__new__(ModelDownloadClient)
    wrapper = Mock()
    wrapper.call_list_plugins.return_value = response_obj
    client._wrapper = wrapper
    return client


def test_list_plugins_flattens_available_plugins_groups() -> None:
    """Grouped available_plugins should be flattened with preserved group type."""
    raw_response = {
        "available_plugins": {
            "downloader": [
                {"name": "huggingface", "description": "HF plugin", "available": True},
                {"name": "ollama", "description": "Ollama plugin", "available": True},
            ],
            "converter": [
                {"name": "openvino", "description": "OpenVINO converter", "available": True},
            ],
        }
    }
    generated_response = PluginsResponse.from_dict(raw_response)

    http_client = _build_http_client_with_response(generated_response)
    plugins = http_client.list_plugins()

    assert any(p.get("name") == "huggingface" and p.get("type") == "downloader" for p in plugins)
    assert any(p.get("name") == "ollama" and p.get("type") == "downloader" for p in plugins)
    assert any(p.get("name") == "openvino" and p.get("type") == "converter" for p in plugins)


def test_list_plugins_preserves_type_when_plugin_has_explicit_type() -> None:
    """Plugin-level type value should be preserved when already present."""
    raw_response = {
        "available_plugins": {
            "downloader": [
                {"name": "huggingface", "type": "downloader", "available": True},
            ]
        }
    }
    generated_response = PluginsResponse.from_dict(raw_response)

    http_client = _build_http_client_with_response(generated_response)
    plugins = http_client.list_plugins()

    assert plugins[0]["type"] == "downloader"


@pytest.mark.integration
@pytest.mark.skipif(
    not INTEGRATION_ENABLED,
    reason="Set RUN_INTEGRATION_TESTS=1 to run live service integration tests.",
)
def test_integration_list_plugins_includes_huggingface_downloader() -> None:
    """Live service should expose huggingface plugin in downloader group."""
    client = ModelDownloadSDK()
    try:
        plugins = asyncio.run(client.list_plugins())
        assert any(
            p.get("name") == "huggingface" and p.get("type") == "downloader"
            for p in plugins
        )
    finally:
        asyncio.run(client.close())
