"""
Model Download SDK - Production-grade wrapper around OpenAPI-generated Model Download Service client.

This SDK provides a stable, ergonomic interface for downloading and managing AI models from various
sources (HuggingFace, Ollama, Geti, etc.) with optional OpenVINO conversion.

Key Features:
- Async-first design with sync wrapper support
- Windows/Linux/WSL compatible path handling
- Comprehensive error handling
- Job tracking and polling
- Automatic path normalization

Example:
    >>> from model_download_sdk import ModelDownloadSDK, ModelSpec, ModelHub
    >>> client = ModelDownloadSDK(base_url="http://localhost:8200")
    >>> spec = ModelSpec(
    ...     name="microsoft/Phi-3.5-mini-instruct",
    ...     hub=ModelHub.HUGGINGFACE,
    ...     type_="llm",
    ... )
    >>> # Download implementation now available in Phase 2

Version:
    0.2.0 (Phase 2: ModelDownloadClient wrapper implemented)

Author:
    Intel AI Team
"""

from model_download_sdk.client import ModelDownloadSDK
from model_download_sdk.exceptions import (
    SDKError,
    ConnectionError,
    ValidationError,
    JobError,
    TimeoutError,
    NotFoundError,
    AuthenticationError,
)
from model_download_sdk.models import (
    ModelSpec,
    Job,
    DownloadResult,
    UploadResult,
)

__version__ = "0.2.0"
__author__ = "Intel AI Team"
__all__ = [
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
]
