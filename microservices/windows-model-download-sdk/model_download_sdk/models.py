"""
Data models for the Model Download SDK.

This module defines SDK-level models that are intentionally separate from the generated
API client models. This isolation ensures:
1. SDK API stability even if generated models change
2. User-friendly field names and defaults
3. Clear separation of concerns

Key Models:
    ModelSpec: Specification for a model to download/convert
    Job: Status and details of an async operation
    DownloadResult: Result of a download operation
    UploadResult: Result of an upload operation

Note:
    These models are converted to/from generated models via an adapter layer
    (_generated_adapter.py) that users should never import directly.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Any


class ModelHub(str, Enum):
    """
    Supported model repository sources.
    
    Enum Values:
        HUGGINGFACE: HuggingFace model hub
        OLLAMA: Ollama model repository
        GETI: Intel Geti platform
        HLS: HuggingFace-compatible server
        OPENVINO: OpenVINO model repository
        ULTRALYTICS: Ultralytics YOLOv8 models
        PIPELINE_ZOO: Intel Pipeline Zoo models
    """

    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    GETI = "geti"
    HLS = "hls"
    OPENVINO = "openvino"
    ULTRALYTICS = "ultralytics"
    PIPELINE_ZOO = "pipeline-zoo-models"

    def __str__(self) -> str:
        """Return the enum value as string."""
        return self.value


class ModelType(str, Enum):
    """
    Supported model types for conversion/optimization.
    
    The model type determines how conversion to OpenVINO IR format is handled.
    
    Enum Values:
        LLM: Large Language Model
        EMBEDDINGS: Text embedding model
        RERANKER: Reranking model
        VISION: Vision model
        VLM: Vision Language Model
    """

    LLM = "llm"
    EMBEDDINGS = "embeddings"
    RERANKER = "reranker"
    VISION = "vision"
    VLM = "vlm"

    def __str__(self) -> str:
        """Return the enum value as string."""
        return self.value


class JobStatus(str, Enum):
    """
    Status of an async job operation.
    
    Enum Values:
        PENDING: Waiting for processing
        DOWNLOADING: Currently downloading model artifacts
        PROCESSING: Currently processing
        COMPLETED: Successfully completed
        FAILED: Failed or cancelled
    """

    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self) -> str:
        """Return the enum value as string."""
        return self.value


class JobOperationType(str, Enum):
    """
    Type of async operation.
    
    Enum Values:
        DOWNLOAD: Model download operation
        CONVERT: Model conversion operation
        UPLOAD: Model upload operation
    """

    DOWNLOAD = "download"
    CONVERT = "convert"
    UPLOAD = "upload"

    def __str__(self) -> str:
        """Return the enum value as string."""
        return self.value


@dataclass
class ModelSpec:
    """
    User-friendly specification for a model to download.
    
    Attributes:
        name: Model identifier (e.g., "microsoft/Phi-3.5-mini-instruct")
        hub: Model repository source (ModelHub enum or string)
        type_: Optional model type for conversion behavior (ModelType enum or string)
        convert_to_openvino: Whether to convert to OpenVINO IR format
        revision: Optional model revision/version/branch to download
        
    Example:
        >>> spec = ModelSpec(
        ...     name="meta-llama/Llama-2-7b",
        ...     hub=ModelHub.HUGGINGFACE,
        ...     type_=ModelType.LLM,
        ...     convert_to_openvino=True,
        ...     revision="main",
        ... )
    """

    name: str
    """Model identifier (e.g., 'microsoft/Phi-3.5-mini-instruct')"""

    hub: ModelHub | str
    """Model repository source"""

    type_: Optional[ModelType | str] = None
    """Optional model type for conversion behavior"""

    convert_to_openvino: bool = False
    """Whether to convert to OpenVINO IR format"""

    revision: Optional[str] = None
    """Optional model revision/version to download"""

    def __post_init__(self) -> None:
        """Validate ModelSpec after initialization."""
        # Convert hub to enum if string
        if isinstance(self.hub, str):
            try:
                self.hub = ModelHub(self.hub)
            except ValueError:
                valid_hubs = [h.value for h in ModelHub]
                raise ValueError(
                    f"Invalid hub '{self.hub}'. Must be one of: {', '.join(valid_hubs)}"
                )

        # Convert type_ to enum if string
        if self.type_ is not None and isinstance(self.type_, str):
            try:
                self.type_ = ModelType(self.type_)
            except ValueError:
                valid_types = [t.value for t in ModelType]
                raise ValueError(
                    f"Invalid type '{self.type_}'. Must be one of: {', '.join(valid_types)}"
                )


@dataclass
class Job:
    """
    Status and details of an async job operation.
    
    Represents the current state of a download, conversion, or upload operation.
    
    Attributes:
        id: Unique job identifier
        operation: Type of operation (download/convert/upload)
        model_name: Name of the model involved
        status: Current job status (pending/processing/completed/failed)
        created_at: Timestamp when job was created
        completed_at: Timestamp when job completed (None if still running)
        error: Error message if job failed
        output_directory: Directory where model was saved (if successful)
        plugin: Name of the plugin handling this job
        hub: Model hub source
        
    Example:
        >>> job = sdk.get_job("job-12345")
        >>> if job.is_complete:
        ...     if job.is_success:
        ...         print(f"Model saved to {job.output_directory}")
        ...     else:
        ...         print(f"Job failed: {job.error}")
    """

    id: str
    """Unique job identifier"""

    operation: JobOperationType
    """Type of operation (download/convert/upload)"""

    model_name: str
    """Name of the model involved"""

    status: JobStatus
    """Current job status"""

    created_at: datetime
    """Timestamp when job was created"""

    hub: str
    """Model hub source"""

    completed_at: Optional[datetime] = None
    """Timestamp when job completed (None if still running)"""

    error: Optional[str] = None
    """Error message if job failed"""

    output_directory: Optional[str] = None
    """Directory where model was saved (if successful)"""

    plugin: Optional[str] = None
    """Name of the plugin handling this job"""

    @property
    def is_complete(self) -> bool:
        """Check if job has reached a terminal state."""
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED)

    @property
    def is_success(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if job failed."""
        return self.status == JobStatus.FAILED

    @property
    def is_pending(self) -> bool:
        """Check if job is waiting to start."""
        return self.status == JobStatus.PENDING

    @property
    def is_processing(self) -> bool:
        """Check if job is currently processing."""
        return self.status in (JobStatus.PROCESSING, JobStatus.DOWNLOADING)


@dataclass
class DownloadResult:
    """
    Result of a download operation.
    
    Contains information about all jobs created by a download request.
    
    Attributes:
        job_ids: List of created job IDs
        message: Status message from server
        successful_jobs: Jobs that completed successfully
        failed_jobs: Jobs that failed
        
    Example:
        >>> result = await sdk.download_models(
        ...     models=[spec1, spec2],
        ...     output_directory="~/models",
        ...     wait=True,
        ... )
        >>> for job in result.successful_jobs:
        ...     print(f"✓ {job.model_name}: {job.output_directory}")
        >>> for job in result.failed_jobs:
        ...     print(f"✗ {job.model_name}: {job.error}")
    """

    job_ids: List[str]
    """List of created job IDs"""

    message: str
    """Status message from server"""

    output_directory: Optional[str] = None
    """Directory where requested models are written"""

    successful_jobs: List[Job] = field(default_factory=list)
    """Jobs that completed successfully (only if wait=True)"""

    failed_jobs: List[Job] = field(default_factory=list)
    """Jobs that failed (only if wait=True)"""

    @property
    def total_jobs(self) -> int:
        """Get total number of jobs."""
        return len(self.job_ids)

    @property
    def successful_count(self) -> int:
        """Get count of successful jobs."""
        return len(self.successful_jobs)

    @property
    def failed_count(self) -> int:
        """Get count of failed jobs."""
        return len(self.failed_jobs)

    @property
    def all_succeeded(self) -> bool:
        """Check if all jobs succeeded."""
        return self.failed_count == 0 and self.successful_count > 0


@dataclass
class UploadResult:
    """
    Result of an upload operation.
    
    Attributes:
        job_id: ID of the upload job
        message: Status message from server
        model_name: Name given to the uploaded model
        status: Status of the upload operation
        
    Example:
        >>> result = await sdk.upload_model(
        ...     file_path="~/my_model.bin",
        ...     model_name="my-custom-model",
        ... )
        >>> print(f"Upload job: {result.job_id}")
    """

    job_id: str
    """ID of the upload job"""

    message: str
    """Status message from server"""

    model_name: str
    """Name given to the uploaded model"""

    status: str = "pending"
    """Status of the upload operation"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata from server response"""
