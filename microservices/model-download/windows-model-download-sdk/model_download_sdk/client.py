"""
Main SDK client for the Model Download Service.

This module provides the primary entry point for interacting with the Model Download Service.
The client supports both synchronous and asynchronous operations.

Key Class:
    ModelDownloadSDK: Main async client (async-first design)

Features:
    - Async-first API design with sync wrapper support
    - Job tracking and polling
    - Comprehensive error handling
    - Windows/Linux/WSL path compatibility
    - Lazy client initialization

Implementation Status:
    Phase 2 Complete: All endpoints implemented with full error handling and logging.

Example:
    >>> from model_download_sdk import ModelDownloadSDK, ModelSpec, ModelHub
    >>> client = ModelDownloadSDK(base_url="http://localhost:8200")
    >>> spec = ModelSpec(name="model-name", hub=ModelHub.HUGGINGFACE)
    >>> result = await client.download_models([spec], output_directory="~/models")
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import ssl
from datetime import datetime, timezone

from model_download_sdk.models import (
    ModelSpec,
    ModelHub,
    ModelType,
    Job,
    JobStatus,
    JobOperationType,
    DownloadResult,
    UploadResult,
)
from model_download_sdk.exceptions import (
    SDKError,
    ConnectionError as SDKConnectionError,
    ValidationError,
    TimeoutError as SDKTimeoutError,
    NotFoundError,
    JobError,
)
from model_download_sdk.filesystem import normalize_path
from model_download_sdk.dlstreamer import DLStreamerClient, DLStreamerConfig
from model_download_sdk._http_client import ModelDownloadClient

logger = logging.getLogger(__name__)


class SDKConfig:
    """
    Configuration for the Model Download SDK client.
    
    Attributes:
        base_url: Base URL of the Model Download Service
        timeout: Request timeout in seconds
        verify_ssl: Whether to verify SSL certificates
        headers: Additional HTTP headers
        cookies: HTTP cookies
        proxy_url: Proxy URL for API requests
        job_poll_interval: Poll interval for job status checks (seconds)
        job_max_timeout: Maximum timeout for job operations (seconds)
        normalize_paths: Whether to normalize paths automatically
        validate_paths: Whether to validate paths before use
        
    Example:
        >>> config = SDKConfig(
        ...     base_url="https://api.example.com:8200",
        ...     timeout=60,
        ...     verify_ssl=True,
        ... )
        >>> client = ModelDownloadSDK(config=config)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8200",
        timeout: float = 30.0,
        verify_ssl: bool | str | ssl.SSLContext = True,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        proxy_url: Optional[str] = None,
        job_poll_interval: float = 2.0,
        job_max_timeout: int = 3600,
        normalize_paths: bool = True,
        validate_paths: bool = True,
    ) -> None:
        """
        Initialize SDK configuration.
        
        Args:
            base_url: Base URL of the service (default: http://localhost:8200)
            timeout: HTTP request timeout in seconds
            verify_ssl: SSL verification (True, False, path to CA bundle, or SSLContext)
            headers: Additional HTTP headers to send with all requests
            cookies: HTTP cookies to send with all requests
            proxy_url: Proxy URL for all HTTP requests
            job_poll_interval: Interval between job status checks (seconds)
            job_max_timeout: Maximum time to wait for job completion (seconds)
            normalize_paths: Auto-normalize paths (expand ~, resolve symlinks, handle long paths)
            validate_paths: Validate paths before passing to API
        """
        self.base_url = base_url
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.proxy_url = proxy_url
        self.job_poll_interval = job_poll_interval
        self.job_max_timeout = job_max_timeout
        self.normalize_paths = normalize_paths
        self.validate_paths = validate_paths


class ModelDownloadSDK:
    """
    Main async client for the Model Download Service.
    
    Provides async methods for downloading, uploading, and managing AI models
    from various sources (HuggingFace, Ollama, Geti, etc.) with optional
    conversion to OpenVINO IR format.
    
    Design:
        - Async-first: All operations are async
        - Lazy initialization: Generated client not created until first use
        - Error handling: All errors normalized to SDK exception hierarchy
        - Path handling: Automatic Windows/Linux/WSL compatibility
    
    Attributes:
        config: SDK configuration
        dlstreamer: DLStreamer integration client
        
    Example:
        >>> client = ModelDownloadSDK()
        >>> # Check service health
        >>> # health = await client.health_check()
        
        >>> # Download a model
        >>> # spec = ModelSpec(
        >>> #     name="microsoft/Phi-3.5-mini-instruct",
        >>> #     hub=ModelHub.HUGGINGFACE,
        >>> #     type_=ModelType.LLM,
        >>> # )
        >>> # result = await client.download_models([spec], output_directory="~/models")
        
        >>> # List jobs
        >>> # jobs = await client.list_jobs()
        
    Note:
        This is a skeleton. Endpoint implementations will be added in Phase 2+.
    """

    def __init__(self, config: Optional[SDKConfig] = None) -> None:
        """
        Initialize the Model Download SDK client.
        
        Args:
            config: Optional SDKConfig instance. If not provided, default config is used.
            
        Example:
            >>> # Use defaults
            >>> client = ModelDownloadSDK()
            
            >>> # Use custom config
            >>> config = SDKConfig(base_url="https://api.example.com:8200")
            >>> client = ModelDownloadSDK(config=config)
        """
        self.config = config or SDKConfig()
        self._http_client: Optional[ModelDownloadClient] = None
        self._dlstreamer = DLStreamerClient(self)
        logger.debug(f"ModelDownloadSDK initialized with base_url={self.config.base_url}")

    @property
    def dlstreamer(self) -> DLStreamerClient:
        """Get DLStreamer integration client."""
        return self._dlstreamer

    def _get_http_client(self) -> ModelDownloadClient:
        """
        Get or create the HTTP client (lazy initialization).
        
        Returns:
            Initialized ModelDownloadClient
            
        Raises:
            ValidationError: If client initialization fails
        """
        if self._http_client is None:
            logger.debug("Initializing HTTP client (lazy initialization)")
            self._http_client = ModelDownloadClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                verify_ssl=self.config.verify_ssl,
                headers=self.config.headers,
                cookies=self.config.cookies,
                proxy_url=self.config.proxy_url,
            )
        return self._http_client

    async def _run_in_executor(self, func, *args) -> Any:
        """
        Run a sync function in thread executor (for wrapping sync HTTP calls).
        
        Args:
            func: Sync function to call
            *args: Arguments to pass
            
        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the Model Download Service.
        
        Returns:
            Dictionary with health status information:
                - status: Health status (e.g., "healthy", "degraded")
                - message: Status message
                - timestamp: Server timestamp
                
        Raises:
            SDKConnectionError: If unable to connect to service
            
        Example:
            >>> health = await client.health_check()
            >>> print(f"Service status: {health['status']}")
        """
        logger.info("Checking service health")
        try:
            client = self._get_http_client()
            result = await self._run_in_executor(client.health_check)
            logger.info(f"Health check passed: {result.get('status', 'unknown')}")
            return result
        except SDKConnectionError as e:
            logger.error(f"Health check failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in health_check: {e}")
            raise SDKConnectionError(f"Health check failed: {e}")

    async def download_models(
        self,
        models: List[ModelSpec],
        output_directory: str,
        wait: bool = False,
        timeout: Optional[int] = None,
    ) -> DownloadResult:
        """
        Download one or more models from various sources.
        
        Asynchronously downloads models from HuggingFace, Ollama, Geti, etc.,
        with optional conversion to OpenVINO IR format.
        
        Args:
            models: List of ModelSpec objects specifying models to download
            output_directory: Directory where models will be saved
            wait: If True, wait for jobs to complete before returning
            timeout: Maximum time to wait for job completion (seconds)
            
        Returns:
            DownloadResult containing:
                - job_ids: IDs of created jobs
                - message: Server status message
                - successful_jobs: Completed successful jobs (if wait=True)
                - failed_jobs: Failed jobs (if wait=True)
                
        Raises:
            ValidationError: If inputs are invalid
            SDKConnectionError: If unable to connect to service
            JobError: If job fails on server (if wait=True)
            TimeoutError: If operation exceeds timeout
            
        Example:
            >>> spec = ModelSpec(
            ...     name="meta-llama/Llama-2-7b",
            ...     hub=ModelHub.HUGGINGFACE,
            ...     type_=ModelType.LLM,
            ... )
            >>> result = await client.download_models(
            ...     models=[spec],
            ...     output_directory="~/models",
            ...     wait=True,
            ...     timeout=3600,
            ... )
            >>> print(f"Downloaded: {result.successful_jobs[0].output_directory}")
        """
        logger.info(f"Downloading {len(models)} model(s) to {output_directory}")
        
        # Validate inputs
        if not models:
            raise ValidationError("At least one model must be specified", field="models")
        
        # Normalize output path
        try:
            if self.config.normalize_paths:
                output_dir = normalize_path(output_directory)
            else:
                output_dir = Path(output_directory).expanduser().resolve(strict=False)
                
            if self.config.validate_paths:
                from model_download_sdk.filesystem import validate_writable
                validate_writable(output_dir)
        except ValidationError as e:
            logger.error(f"Path validation failed: {e}")
            raise
        
        try:
            # Call HTTP client to initiate download
            client = self._get_http_client()
            job_ids = await self._run_in_executor(
                client.download_models,
                models,
                str(output_dir),
            )
            
            logger.info(f"Download initiated: {len(job_ids)} job(s) created")
            
            # Create result
            result = DownloadResult(
                job_ids=job_ids,
                message=f"Started processing {len(models)} model(s)",
                output_directory=str(output_dir),
            )
            
            # Wait for completion if requested
            if wait:
                timeout_val = timeout or self.config.job_max_timeout
                logger.info(f"Waiting for downloads to complete (timeout={timeout_val}s)")
                
                successful = []
                failed = []

                # Preserve the original request metadata for each job when possible.
                model_by_job_id: Dict[str, Optional[ModelSpec]] = {
                    job_id: (models[idx] if idx < len(models) else None)
                    for idx, job_id in enumerate(job_ids)
                }

                async def _build_failed_job(job_id: str, error: Exception) -> Job:
                    model_name = "unknown"
                    hub = "unknown"
                    created_at = datetime.now(timezone.utc)
                    output_directory = None
                    plugin = None

                    source_spec = model_by_job_id.get(job_id)
                    if source_spec is not None:
                        model_name = source_spec.name or model_name
                        hub = (
                            str(source_spec.hub.value)
                            if hasattr(source_spec.hub, "value")
                            else str(source_spec.hub)
                        )

                    # Best-effort: fetch server-side metadata directly from low-level client
                    # without raising JobError for failed states.
                    try:
                        latest_job = await self._run_in_executor(client.get_job_status, job_id)
                        model_name = latest_job.model_name or model_name
                        hub = latest_job.hub or hub
                        created_at = latest_job.created_at or created_at
                        output_directory = latest_job.output_directory
                        plugin = latest_job.plugin
                    except Exception:
                        pass

                    return Job(
                        id=job_id,
                        operation=JobOperationType.DOWNLOAD,
                        model_name=model_name,
                        status=JobStatus.FAILED,
                        created_at=created_at,
                        completed_at=datetime.now(timezone.utc),
                        hub=hub,
                        error=str(error),
                        output_directory=output_directory,
                        plugin=plugin,
                    )
                
                start_time = datetime.now()
                for job_id in job_ids:
                    try:
                        job = await self.wait_for_job(job_id, timeout=timeout_val)
                        if job.is_success:
                            successful.append(job)
                        else:
                            failed.append(job)
                    except SDKTimeoutError as e:
                        logger.error(f"Job {job_id} timed out: {e}")
                        failed.append(await _build_failed_job(job_id, e))
                    except Exception as e:
                        logger.error(f"Error waiting for job {job_id}: {e}")
                        failed.append(await _build_failed_job(job_id, e))
                
                result.successful_jobs = successful
                result.failed_jobs = failed
                
                logger.info(
                    f"Download complete: {len(successful)} successful, "
                    f"{len(failed)} failed"
                )
            
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error downloading models: {e}")
            raise SDKConnectionError(f"Model download failed: {e}")

    async def download_model(
        self,
        model_name: str,
        hub: str,
        download_path: str = ".",
        model_type: Optional[str] = None,
        convert_to_openvino: bool = False,
        revision: Optional[str] = None,
        wait: bool = False,
        timeout: Optional[int] = None,
    ) -> DownloadResult:
        """
        Download a single model with simple string-based API.
        
        This is a convenience method that wraps download_models() with a simpler
        interface for downloading a single model. Internally converts string inputs
        to strongly typed SDK objects.
        
        Args:
            model_name: Model identifier (e.g., "microsoft/Phi-3.5-mini-instruct")
            hub: Model repository source (e.g., "huggingface", "ollama", "ultralytics")
            download_path: Directory where model will be saved (uses pathlib.Path)
            model_type: Optional model type (e.g., "llm", "vision", "embeddings")
            convert_to_openvino: Whether to convert to OpenVINO IR format
            revision: Optional model revision/branch/version to download
            wait: If True, wait for job to complete before returning
            timeout: Maximum time to wait for job completion (seconds)
            
        Returns:
            DownloadResult containing:
                - job_ids: IDs of created jobs (single item for single model)
                - message: Server status message
                - successful_jobs: Completed successful jobs (if wait=True)
                - failed_jobs: Failed jobs (if wait=True)
                
        Raises:
            ValidationError: If inputs are invalid or hub/type unknown
            SDKConnectionError: If unable to connect to service
            TimeoutError: If operation exceeds timeout
            
        Example:
            >>> result = await client.download_model(
            ...     model_name="microsoft/Phi-3.5-mini-instruct",
            ...     hub="huggingface",
            ...     download_path="models",
            ...     model_type="llm",
            ...     convert_to_openvino=True,
            ...     wait=True,
            ...     timeout=3600,
            ... )
            >>> print(f"Downloaded to: {result.output_directory}")
            
        Note:
            This method is a convenience wrapper. For downloading multiple models
            in a single operation, use download_models() with a list of ModelSpec.
        """
        logger.info(
            f"Downloading single model: {model_name} from {hub} to {download_path}"
        )
        
        # Validate model_name
        if not model_name:
            raise ValidationError("model_name cannot be empty", field="model_name")
        
        if not isinstance(model_name, str):
            raise ValidationError(
                f"model_name must be string, got {type(model_name).__name__}",
                field="model_name",
            )
        
        # Validate hub
        if not hub:
            raise ValidationError("hub cannot be empty", field="hub")
        
        if not isinstance(hub, str):
            raise ValidationError(
                f"hub must be string, got {type(hub).__name__}",
                field="hub",
            )
        
        # Convert hub string to ModelHub enum (validation happens in ModelSpec)
        try:
            hub_lower = hub.lower()
            # Map common aliases to enum values
            hub_aliases = {
                "huggingface": ModelHub.HUGGINGFACE,
                "hf": ModelHub.HUGGINGFACE,
                "ollama": ModelHub.OLLAMA,
                "ultralytics": ModelHub.ULTRALYTICS,
                "yolo": ModelHub.ULTRALYTICS,
                "geti": ModelHub.GETI,
                "openvino": ModelHub.OPENVINO,
                "hls": ModelHub.HLS,
                "pipeline-zoo": ModelHub.PIPELINE_ZOO,
            }
            
            if hub_lower in hub_aliases:
                hub_enum = hub_aliases[hub_lower]
            else:
                # Try direct enum lookup
                hub_enum = ModelHub(hub_lower)
                
        except (ValueError, KeyError) as e:
            valid_hubs = [h.value for h in ModelHub]
            raise ValidationError(
                f"Unknown hub '{hub}'. Must be one of: {', '.join(valid_hubs)}",
                field="hub",
                value=hub,
            )
        
        # Convert model_type string to ModelType enum if provided
        model_type_enum = None
        if model_type is not None:
            if not isinstance(model_type, str):
                raise ValidationError(
                    f"model_type must be string, got {type(model_type).__name__}",
                    field="model_type",
                )
            
            try:
                model_type_lower = model_type.lower()
                # Map common aliases
                type_aliases = {
                    "llm": ModelType.LLM,
                    "embeddings": ModelType.EMBEDDINGS,
                    "embed": ModelType.EMBEDDINGS,
                    "reranker": ModelType.RERANKER,
                    "rerank": ModelType.RERANKER,
                    "vision": ModelType.VISION,
                    "image": ModelType.VISION,
                    "vlm": ModelType.VLM,
                }
                
                if model_type_lower in type_aliases:
                    model_type_enum = type_aliases[model_type_lower]
                else:
                    model_type_enum = ModelType(model_type_lower)
                    
            except (ValueError, KeyError) as e:
                valid_types = [t.value for t in ModelType]
                raise ValidationError(
                    f"Unknown model type '{model_type}'. Must be one of: {', '.join(valid_types)}",
                    field="model_type",
                    value=model_type,
                )
        
        # Use pathlib for path handling
        from pathlib import Path
        
        download_path_obj = Path(download_path)
        logger.debug(f"Using pathlib.Path for download_path: {download_path_obj}")
        
        # Convert download_path to string for API (handles ~ expansion via normalize_path)
        download_path_str = str(download_path_obj)
        
        # Create ModelSpec with strongly typed objects
        spec = ModelSpec(
            name=model_name,
            hub=hub_enum,
            type_=model_type_enum,
            convert_to_openvino=convert_to_openvino,
            revision=revision,
        )
        
        logger.debug(
            f"Created ModelSpec: name={model_name}, hub={hub_enum}, "
            f"type_={model_type_enum}, convert_to_openvino={convert_to_openvino}"
        )
        
        # Call download_models with single model
        result = await self.download_models(
            models=[spec],
            output_directory=download_path_str,
            wait=wait,
            timeout=timeout,
        )
        
        logger.info(
            f"download_model completed: {len(result.job_ids)} job(s) created"
        )
        
        return result

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
        High-level workflow to prepare a model for DLStreamer.

        Workflow:
        1. Download model
        2. Wait for completion
        3. Verify output exists
        4. Return local model path

        Args:
            model_name: Model identifier (e.g., "resnet50")
            hub: Model source hub (e.g., "huggingface")
            download_path: Local directory for download
            model_type: Optional model type
            revision: Optional model revision
            convert_to_openvino: Convert to OpenVINO IR for DLStreamer
            poll_interval: Initial polling interval in seconds
            timeout: Max wait time in seconds

        Returns:
            Local model path as pathlib.Path

        Raises:
            ValidationError: Invalid input
            JobError: Model pull job failed
            TimeoutError: Job exceeded timeout
            NotFoundError: Local output path not found
            SDKConnectionError: Connection failure

        Example:
            >>> path = await client.pull_for_dlstreamer(
            ...     model_name="resnet50",
            ...     hub="huggingface",
            ... )
            >>> print(path)
        """
        return await self.dlstreamer.pull_for_dlstreamer(
            model_name=model_name,
            hub=hub,
            download_path=download_path,
            model_type=model_type,
            revision=revision,
            convert_to_openvino=convert_to_openvino,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    async def upload_model(
        self,
        file_path: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UploadResult:
        """
        Upload a custom model to the service.
        
        Args:
            file_path: Path to model file to upload
            model_name: Name to give the uploaded model
            metadata: Optional metadata about the model
            
        Returns:
            UploadResult containing:
                - job_id: ID of the upload job
                - message: Status message
                - model_name: Name of uploaded model
                
        Raises:
            ValidationError: If inputs are invalid
            SDKConnectionError: If unable to connect to service
            JobError: If upload fails on server
            
        Example:
            >>> result = await client.upload_model(
            ...     file_path="~/my_model.bin",
            ...     model_name="my-custom-model",
            ... )
            >>> print(f"Upload job: {result.job_id}")
        """
        logger.info(f"Uploading model: {model_name} from {file_path}")
        
        # Validate inputs
        if not file_path:
            raise ValidationError("file_path cannot be empty", field="file_path")
        if not model_name:
            raise ValidationError("model_name cannot be empty", field="model_name")
        
        # Check file exists
        import os
        if not os.path.exists(file_path):
            raise ValidationError(
                f"File not found: {file_path}",
                field="file_path",
                value=file_path,
            )
        
        logger.debug(f"Model upload not yet implemented")
        raise NotImplementedError(
            "upload_model endpoint implementation pending (Phase 3+)"
        )

    async def get_job(self, job_id: str) -> Job:
        """
        Get status and details of a specific job.
        
        Fetches the current status of a job including progress information,
        completion status, and error details if applicable.
        
        Args:
            job_id: ID of the job to retrieve
            
        Returns:
            Job object with current status and details
            
        Raises:
            ValidationError: If job_id is empty
            NotFoundError: If job not found on server
            JobError: If job has failed
            SDKConnectionError: If unable to connect to service
            
        Example:
            >>> job = await client.get_job("job-12345")
            >>> print(f"Status: {job.status}")
            >>> print(f"Progress: {job.created_at} -> {job.completed_at}")
            >>> if job.is_complete:
            ...     if job.is_success:
            ...         print(f"✓ Model saved to: {job.output_directory}")
            ...     else:
            ...         print(f"✗ Job failed: {job.error}")
        """
        logger.debug(f"Getting job status for {job_id}")
        
        if not job_id:
            raise ValidationError("job_id cannot be empty", field="job_id")
        
        try:
            client = self._get_http_client()
            job = await self._run_in_executor(client.get_job_status, job_id)
            
            logger.debug(f"Job {job_id} status: {job.status} (operation: {job.operation})")
            
            # If job has failed, raise JobError
            if job.status == JobStatus.FAILED:
                logger.error(f"Job {job_id} failed: {job.error}")
                raise JobError(
                    job_id=job_id,
                    message=job.error or "Job failed",
                    status=job.status.value if job.status else None,
                    error_details=f"Model: {job.model_name}, Operation: {job.operation}",
                )
            
            return job
        except NotFoundError:
            logger.warning(f"Job {job_id} not found")
            raise
        except JobError:
            raise
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            raise SDKConnectionError(f"Failed to get job status: {e}")

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
        """
        List all jobs with optional filtering.
        
        Args:
            status: Filter by job status (pending/processing/completed/failed)
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip (for pagination)
            
        Returns:
            List of Job objects
            
        Raises:
            SDKConnectionError: If unable to connect to service
            
        Example:
            >>> jobs = await client.list_jobs(status=JobStatus.PROCESSING)
            >>> print(f"Processing {len(jobs)} jobs")
        """
        logger.debug(f"Listing jobs (status={status})")
        
        try:
            client = self._get_http_client()
            jobs = await self._run_in_executor(client.list_jobs)
            
            # Filter by status if requested
            if status:
                jobs = [j for j in jobs if j.status == status]
            
            # Apply limit and offset
            jobs = jobs[offset:offset + limit]
            
            logger.info(f"Retrieved {len(jobs)} jobs")
            return jobs
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            raise SDKConnectionError(f"Failed to list jobs: {e}")

    async def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 5,
        timeout: Optional[int] = None,
    ) -> Job:
        """
        Wait for a job to complete with polling.
        
        Polls the job status at regular intervals until completion. Uses exponential
        backoff to reduce server load: intervals are multiplied by 1.5 (capped at 60s).
        
        Args:
            job_id: ID of the job to wait for
            poll_interval: Initial poll interval in seconds (default: 5). Increases exponentially.
            timeout: Maximum time to wait in seconds (default: config.job_max_timeout)
            
        Returns:
            Job object at completion (status: completed or failed)
            
        Raises:
            ValidationError: If job_id is empty
            TimeoutError: If timeout exceeded before completion
            NotFoundError: If job not found during polling
            JobError: If job fails
            SDKConnectionError: If connection error during polling
            
        Example:
            >>> # Simple usage with defaults (5s initial interval, 3600s timeout)
            >>> job = await client.wait_for_job("job-12345")
            >>> if job.is_success:
            ...     print(f"✓ Downloaded to {job.output_directory}")
            >>> else:
            ...     print(f"✗ Failed: {job.error}")
            
            >>> # Custom polling interval and timeout
            >>> job = await client.wait_for_job(
            ...     job_id="job-12345",
            ...     poll_interval=2,  # Check every 2s initially
            ...     timeout=7200,     # Wait up to 2 hours
            ... )
        """
        logger.info(f"Waiting for job {job_id} (timeout={timeout}s, initial_poll={poll_interval}s)")
        
        if not job_id:
            raise ValidationError("job_id cannot be empty", field="job_id")
        
        timeout_val = timeout or self.config.job_max_timeout
        current_poll_interval = poll_interval
        max_poll_interval = 60  # Cap exponential backoff at 60 seconds
        backoff_multiplier = 1.5
        start_time = datetime.now()
        poll_count = 0
        
        logger.info(f"Job polling started (timeout: {timeout_val}s, initial interval: {poll_interval}s)")
        
        while True:
            poll_count += 1
            
            try:
                # Get current job status
                job = await self.get_job(job_id)
                
                # Check if completed
                if job.is_complete:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if job.is_success:
                        logger.info(
                            f"✓ Job {job_id} completed successfully "
                            f"(elapsed: {elapsed:.1f}s, polls: {poll_count})"
                        )
                    else:
                        logger.warning(
                            f"✗ Job {job_id} completed with failure "
                            f"(elapsed: {elapsed:.1f}s, polls: {poll_count}): {job.error}"
                        )
                    return job
                
                # Check timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout_val:
                    logger.error(
                        f"Job {job_id} timeout after {elapsed:.1f}s "
                        f"(status: {job.status}, polls: {poll_count})"
                    )
                    raise SDKTimeoutError(
                        f"Job {job_id} did not complete within {timeout_val}s "
                        f"(status: {job.status}, polls: {poll_count})",
                        timeout_seconds=timeout_val,
                        operation=f"wait_for_job({job_id})",
                    )
                
                # Exponential backoff
                logger.debug(
                    f"Job {job_id} still {job.status.value} "
                    f"(elapsed: {elapsed:.1f}s, next poll in {current_poll_interval:.1f}s)"
                )
                await asyncio.sleep(current_poll_interval)
                
                # Increase poll interval with exponential backoff (capped at max)
                current_poll_interval = min(
                    current_poll_interval * backoff_multiplier,
                    max_poll_interval,
                )
                
            except (NotFoundError, JobError):
                # These exceptions should propagate immediately
                raise
            except SDKTimeoutError:
                # Timeout exception should propagate immediately
                raise
            except Exception as e:
                logger.error(f"Error waiting for job {job_id}: {e}")
                raise SDKConnectionError(
                    f"Error polling job {job_id}: {e}",
                    original_error=e,
                )

    async def get_model_results(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
        """
        Get completed model operations.
        
        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            
        Returns:
            List of completed Job objects
            
        Raises:
            SDKConnectionError: If unable to connect to service
            
        Example:
            >>> results = await client.get_model_results()
            >>> for result in results:
            ...     print(f"{result.model_name}: {result.output_directory}")
        """
        logger.debug("Getting model results")
        
        try:
            client = self._get_http_client()
            jobs = await self._run_in_executor(client.get_model_results)
            
            # Apply limit and offset
            jobs = jobs[offset:offset + limit]
            
            logger.info(f"Retrieved {len(jobs)} model results")
            return jobs
        except Exception as e:
            logger.error(f"Error getting model results: {e}")
            raise SDKConnectionError(f"Failed to get model results: {e}")

    async def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List available plugins and their capabilities.
        
        Returns:
            List of plugin information dictionaries
            
        Raises:
            SDKConnectionError: If unable to connect to service
            
        Example:
            >>> plugins = await client.list_plugins()
            >>> for plugin in plugins:
            ...     print(f"Plugin: {plugin['name']} v{plugin['version']}")
        """
        logger.debug("Listing plugins")
        
        try:
            client = self._get_http_client()
            plugins = await self._run_in_executor(client.list_plugins)
            logger.info(f"Retrieved {len(plugins)} plugins")
            return plugins
        except Exception as e:
            logger.error(f"Error listing plugins: {e}")
            raise SDKConnectionError(f"Failed to list plugins: {e}")

    async def download_and_wait(
        self,
        models: List[ModelSpec],
        output_directory: str,
        timeout: Optional[int] = None,
    ) -> DownloadResult:
        """
        Convenience method: Download models and wait for completion.
        
        Equivalent to download_models(..., wait=True).
        
        Args:
            models: List of ModelSpec objects
            output_directory: Directory to save models
            timeout: Maximum wait time (seconds)
            
        Returns:
            DownloadResult with job information
            
        Raises:
            ValidationError: If inputs are invalid
            SDKConnectionError: If unable to connect
            JobError: If job fails
            TimeoutError: If operation exceeds timeout
            
        Note:
            Endpoint implementation pending (Phase 2+)
            
        Example:
            >>> result = await client.download_and_wait(
            ...     models=[spec1, spec2],
            ...     output_directory="~/models",
            ...     timeout=3600,
            ... )
            >>> for job in result.successful_jobs:
            ...     print(f"✓ {job.model_name}")
        """
        return await self.download_models(
            models=models,
            output_directory=output_directory,
            wait=True,
            timeout=timeout,
        )

    async def close(self) -> None:
        """
        Close client connections and clean up resources.
        
        Must be called when done using the client to ensure proper cleanup.
        
        Example:
            >>> try:
            ...     # Use client
            ...     result = await client.download_models(...)
            ... finally:
            ...     await client.close()
            
            >>> # Or use as context manager (future feature)
            >>> async with ModelDownloadSDK() as client:
            ...     result = await client.download_models(...)
        """
        logger.debug("Closing SDK client")
        if self._http_client:
            self._http_client.close()
            self._http_client = None
        logger.info("SDK client closed")

    async def __aenter__(self) -> "ModelDownloadSDK":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()