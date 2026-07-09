"""
Adapter layer for converting between SDK models and generated API models.

This module is the ONLY place that imports from the generated client package.
All conversions are centralized here to ensure:
1. SDK models remain stable even if generated code changes
2. Single point of maintenance for API changes
3. Easy to test and mock

The adapter should NOT be imported by SDK users directly.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from model_download_sdk.models import (
    ModelSpec,
    Job,
    JobStatus,
    JobOperationType,
)
from model_download_sdk.exceptions import ValidationError

logger = logging.getLogger(__name__)


def sdk_model_spec_to_generated(spec: ModelSpec) -> Dict[str, Any]:
    """
    Convert SDK ModelSpec to generated ModelRequest dict format.
    
    Args:
        spec: SDK ModelSpec
        
    Returns:
        Dictionary suitable for generated ModelRequest.to_dict()
        
    Raises:
        ValidationError: If conversion fails
    """
    try:
        return {
            "name": spec.name,
            "hub": str(spec.hub.value) if hasattr(spec.hub, 'value') else str(spec.hub),
            "type_": str(spec.type_.value) if spec.type_ and hasattr(spec.type_, 'value') else (str(spec.type_) if spec.type_ else None),
            "is_ovms": spec.convert_to_openvino,
            "revision": spec.revision,
        }
    except Exception as e:
        raise ValidationError(
            f"Failed to convert ModelSpec to generated format: {e}",
            field="model_spec",
        )


def generated_job_to_sdk(gen_job: Any) -> Job:
    """
    Convert generated Job to SDK Job.
    
    Args:
        gen_job: Generated API Job object
        
    Returns:
        SDK Job object
        
    Note:
        Handles None/UNSET fields gracefully
    """
    from model_download_sdk.types import UNSET
    
    def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
        """Safely get attribute, handling UNSET values."""
        val = getattr(obj, attr, default)
        if val is UNSET:
            return default
        # Generated client uses its own Unset sentinel class, distinct from SDK UNSET.
        if val is not None and val.__class__.__name__ == "Unset":
            return default
        return val
    
    # Convert status string to enum
    status_str = safe_get(gen_job, 'status', 'pending')
    try:
        status = JobStatus(status_str) if isinstance(status_str, str) else status_str
    except ValueError:
        logger.warning(f"Unknown job status: {status_str}, defaulting to pending")
        status = JobStatus.PENDING
    
    # Convert operation type
    op_type_str = safe_get(gen_job, 'operation_type', 'download')
    try:
        operation = JobOperationType(op_type_str) if isinstance(op_type_str, str) else op_type_str
    except ValueError:
        logger.warning(f"Unknown operation type: {op_type_str}, defaulting to download")
        operation = JobOperationType.DOWNLOAD
    
    # Handle datetime fields
    created_at = safe_get(gen_job, 'creation_time')
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
    elif created_at is None:
        created_at = datetime.now()
    
    completed_at = safe_get(gen_job, 'completion_time')
    if isinstance(completed_at, str):
        completed_at = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
    
    return Job(
        id=safe_get(gen_job, 'job_id', 'unknown'),
        operation=operation,
        model_name=safe_get(gen_job, 'model_name', 'unknown'),
        status=status,
        created_at=created_at,
        hub=safe_get(gen_job, 'hub', 'unknown'),
        completed_at=completed_at,
        error=safe_get(gen_job, 'error'),
        output_directory=safe_get(gen_job, 'output_dir'),
        plugin=safe_get(gen_job, 'plugin_name'),
    )


def generated_download_response_to_job_ids(response: Any) -> List[str]:
    """
    Extract job IDs from generated DownloadResponse.
    
    Args:
        response: Generated DownloadResponse object
        
    Returns:
        List of job IDs
    """
    from model_download_sdk.types import UNSET
    
    job_ids = getattr(response, 'job_ids', None)
    if job_ids is UNSET or job_ids is None:
        return []
    return list(job_ids) if job_ids else []


def generated_error_response_to_message(response: Any) -> str:
    """
    Extract error message from generated error response.
    
    Args:
        response: Generated error response object
        
    Returns:
        Error message string
    """
    # Try common error message fields
    for field in ['detail', 'message', 'error', 'description']:
        value = getattr(response, field, None)
        if value:
            return str(value)
    
    return str(response)


class GeneratedClientWrapper:
    """
    Wrapper to provide simplified access to generated client functions.
    
    This allows for easier mocking and testing.
    """

    def __init__(self, client: Any) -> None:
        """
        Initialize wrapper with generated client.
        
        Args:
            client: Generated API client instance
        """
        self.client = client

    def call_health_check(self) -> Any:
        """Call health_check endpoint."""
        from model_download_service_api_client.api.health.health_check import sync as health_check_sync
        return health_check_sync(client=self.client)

    def call_download_models(self, models_dict: List[Dict[str, Any]], download_path: str) -> Any:
        """Call download_models endpoint."""
        from model_download_service_api_client.models.model_download_request import ModelDownloadRequest
        from model_download_service_api_client.models.model_hub import ModelHub
        from model_download_service_api_client.models.model_request import ModelRequest
        from model_download_service_api_client.models.model_type import ModelType
        from model_download_service_api_client.api.models.download_models import sync as download_models_sync

        valid_hubs = {item.value: item for item in ModelHub}
        valid_types = {item.value: item for item in ModelType}

        def parse_hub(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, ModelHub):
                return value
            hub_str = str(value)
            parsed = valid_hubs.get(hub_str)
            if parsed is None:
                allowed = ", ".join(sorted(valid_hubs.keys()))
                raise ValidationError(
                    f"Unknown hub '{hub_str}'. Must be one of: {allowed}",
                    field="hub",
                    value=hub_str,
                )
            return parsed

        def parse_type(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, ModelType):
                return value
            type_str = str(value)
            parsed = valid_types.get(type_str)
            if parsed is None:
                allowed = ", ".join(sorted(valid_types.keys()))
                raise ValidationError(
                    f"Unknown type '{type_str}'. Must be one of: {allowed}",
                    field="type_",
                    value=type_str,
                )
            return parsed
        
        # Convert dict to generated models
        model_requests = []
        for model_dict in models_dict:
            payload = {k: v for k, v in model_dict.items() if v is not None}
            if "hub" in payload:
                payload["hub"] = parse_hub(payload["hub"])
            if "type_" in payload:
                payload["type_"] = parse_type(payload["type_"])
            model_requests.append(ModelRequest(**payload))
        
        request = ModelDownloadRequest(models=model_requests)
        return download_models_sync(client=self.client, body=request, download_path=download_path)

    def call_get_job_status(self, job_id: str) -> Any:
        """Call get_job_status endpoint."""
        from model_download_service_api_client.api.jobs.get_job_status import sync as get_job_status_sync
        return get_job_status_sync(client=self.client, job_id=job_id)

    def call_list_jobs(self) -> Any:
        """Call list_jobs endpoint."""
        from model_download_service_api_client.api.jobs.list_jobs import sync as list_jobs_sync
        return list_jobs_sync(client=self.client)

    def call_get_model_results(self) -> Any:
        """Call get_model_results endpoint."""
        from model_download_service_api_client.api.models.get_model_results import sync as get_model_results_sync
        return get_model_results_sync(client=self.client)

    def call_list_plugins(self) -> Any:
        """Call list_plugins endpoint."""
        from model_download_service_api_client.api.plugins.list_plugins import sync as list_plugins_sync
        return list_plugins_sync(client=self.client)
