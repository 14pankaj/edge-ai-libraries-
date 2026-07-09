"""
Intel DLStreamer integration for the Model Download SDK.

Provides high-level workflows for preparing models for DLStreamer usage:
1. Download model
2. Wait for completion
3. Verify output path exists locally
4. Return local model path
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

from model_download_sdk.exceptions import (
    SDKError,
    ValidationError,
    NotFoundError,
    JobError,
)

if TYPE_CHECKING:
    from model_download_sdk.client import ModelDownloadSDK

import logging

logger = logging.getLogger(__name__)


@dataclass
class DLStreamerConfig:
    """
    Configuration for DLStreamer model conversion and inference.
    
    Attributes:
        device: Target device for inference (CPU, GPU, VPU)
        batch_size: Batch size for inference
        num_streams: Number of parallel streams
        model_precision: Model precision (FP32, FP16, INT8)
        
    Note:
        This is a skeleton. Full implementation will be added in Phase 2+.
    """

    device: str = "CPU"
    """Target inference device"""

    batch_size: int = 1
    """Batch size for inference"""

    num_streams: int = 1
    """Number of parallel inference streams"""

    model_precision: str = "FP32"
    """Model precision (FP32, FP16, INT8)"""

    additional_config: Dict[str, Any] = None
    """Additional DLStreamer configuration parameters"""

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.additional_config is None:
            self.additional_config = {}

        # Validate device
        valid_devices = {"CPU", "GPU", "VPU", "HDDL"}
        if self.device not in valid_devices:
            raise ValueError(
                f"Invalid device '{self.device}'. Must be one of: {', '.join(valid_devices)}"
            )

        # Validate precision
        valid_precisions = {"FP32", "FP16", "INT8"}
        if self.model_precision not in valid_precisions:
            raise ValueError(
                f"Invalid precision '{self.model_precision}'. Must be one of: {', '.join(valid_precisions)}"
            )


class DLStreamerClient:
    """
    Client for DLStreamer model operations.
    
    Provides utilities for:
    - Converting models for DLStreamer inference
    - Managing DLStreamer configurations
    - Querying DLStreamer capabilities
    
    Note:
        This is a skeleton. Endpoint integration will be added in Phase 2+.
        
    Example:
        >>> # Future implementation
        >>> dlstreamer = DLStreamerClient()
        >>> config = DLStreamerConfig(device="GPU", model_precision="FP16")
        >>> # ... convert model ...
    """

    def __init__(self, sdk_client: Optional["ModelDownloadSDK"] = None) -> None:
        """Initialize DLStreamer client."""
        self.config = DLStreamerConfig()
        self._sdk_client = sdk_client

    def bind_sdk(self, sdk_client: "ModelDownloadSDK") -> None:
        """Bind a ModelDownloadSDK instance to this DLStreamer client."""
        self._sdk_client = sdk_client

    def set_config(self, config: DLStreamerConfig) -> None:
        """
        Set the DLStreamer configuration.
        
        Args:
            config: DLStreamerConfig instance
            
        Note:
            Placeholder for Phase 2+ implementation
        """
        self.config = config

    def get_supported_devices(self) -> list[str]:
        """
        Get list of supported inference devices.
        
        Returns:
            List of device names (CPU, GPU, VPU, HDDL)
            
        Note:
            Placeholder for Phase 2+ implementation
        """
        return ["CPU", "GPU", "VPU", "HDDL"]

    def get_supported_precisions(self) -> list[str]:
        """
        Get list of supported model precisions.
        
        Returns:
            List of precision names (FP32, FP16, INT8)
            
        Note:
            Placeholder for Phase 2+ implementation
        """
        return ["FP32", "FP16", "INT8"]

    async def pull_for_dlstreamer(
        self,
        model_name: str,
        hub: str,
        download_path: str = "models",
        model_type: Optional[str] = None,
        revision: Optional[str] = None,
        convert_to_openvino: bool = True,
        poll_interval: float = 5,
        timeout: int = 3600,
    ) -> Path:
        """
        Download and prepare a model for DLStreamer workflows.

        Workflow:
        1. Download model
        2. Wait for completion
        3. Verify output path exists
        4. Return local model path

        Args:
            model_name: Model identifier (e.g., "resnet50", "google/vit-base-patch16-224")
            hub: Model source hub (e.g., "huggingface")
            download_path: Local directory where model should be downloaded
            model_type: Optional model type
            revision: Optional model revision
            convert_to_openvino: Whether to convert to OpenVINO IR for DLStreamer
            poll_interval: Initial poll interval in seconds
            timeout: Maximum time to wait in seconds

        Returns:
            Local model output path as pathlib.Path

        Raises:
            ValidationError: Invalid input parameters
            SDKError: SDK is not bound
            JobError: Download job failed
            NotFoundError: Output path not found after completion
        """
        if self._sdk_client is None:
            raise SDKError(
                "DLStreamer client is not bound to ModelDownloadSDK. "
                "Create it via ModelDownloadSDK()."
            )

        if not model_name:
            raise ValidationError("model_name cannot be empty", field="model_name")

        if not hub:
            raise ValidationError("hub cannot be empty", field="hub")

        if poll_interval <= 0:
            raise ValidationError(
                "poll_interval must be > 0",
                field="poll_interval",
                value=poll_interval,
            )

        if timeout <= 0:
            raise ValidationError(
                "timeout must be > 0",
                field="timeout",
                value=timeout,
            )

        logger.info(
            "DLStreamer pull started for model=%s hub=%s download_path=%s",
            model_name,
            hub,
            download_path,
        )

        # Step 1: Download model (async job submission)
        result = await self._sdk_client.download_model(
            model_name=model_name,
            hub=hub,
            download_path=download_path,
            model_type=model_type,
            convert_to_openvino=convert_to_openvino,
            revision=revision,
            wait=False,
        )

        if not result.job_ids:
            raise SDKError("Download request did not return a job ID")

        job_id = result.job_ids[0]
        logger.info("DLStreamer pull job submitted: job_id=%s", job_id)

        # Step 2: Wait for completion
        job = await self._sdk_client.wait_for_job(
            job_id=job_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

        # wait_for_job/get_job raises JobError on failed state, but keep a safety check.
        if not job.is_success:
            raise JobError(
                job_id=job.id,
                message=job.error or "Model pull failed",
                status=job.status.value,
                error_details="DLStreamer pull workflow",
            )

        # Step 3: Verify explicit output paths exist using pathlib (cross-platform).
        # Do not infer or create fallback directories here.
        candidate_paths: list[Path] = []

        if job.output_directory:
            candidate_paths.append(Path(job.output_directory).expanduser())

        if result.output_directory:
            candidate_paths.append(Path(result.output_directory).expanduser())

        resolved_existing: Optional[Path] = None
        for path in candidate_paths:
            try:
                resolved = path.resolve(strict=False)
            except OSError:
                resolved = path

            if resolved.exists():
                resolved_existing = resolved
                break

        if resolved_existing is None:
            checked = ", ".join(str(p) for p in candidate_paths)
            raise NotFoundError(
                f"DLStreamer pull completed but no local model path was found. Checked: {checked}",
                resource_id=job.id,
                resource_type="local_model_path",
            )

        # Step 4: Return local model path
        logger.info(
            "DLStreamer pull complete: model=%s path=%s",
            model_name,
            resolved_existing,
        )
        return resolved_existing
