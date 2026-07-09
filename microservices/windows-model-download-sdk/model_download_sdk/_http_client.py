"""
Low-level HTTP client for the Model Download Service.

This module wraps the generated API client, handling errors, logging, and
providing a clean interface for SDK operations.

This is the internal client used by ModelDownloadSDK.
Do NOT import from this module in user code.
"""

import logging
from typing import Optional, List, Dict, Any

import httpx

from model_download_sdk.models import (
    ModelSpec,
    Job,
    DownloadResult,
    UploadResult,
)
from model_download_sdk.exceptions import (
    SDKError,
    ConnectionError as SDKConnectionError,
    ValidationError,
    JobError,
    NotFoundError,
)
from model_download_sdk._generated_adapter import (
    GeneratedClientWrapper,
    sdk_model_spec_to_generated,
    generated_job_to_sdk,
    generated_download_response_to_job_ids,
    generated_error_response_to_message,
)
from model_download_sdk.error_mapper import (
    map_http_error,
    map_network_error,
    map_generated_error,
)

logger = logging.getLogger(__name__)


class ModelDownloadClient:
    """
    Low-level HTTP client for the Model Download Service.
    
    Wraps the generated API client with error handling, logging, and retry logic.
    
    Attributes:
        base_url: Base URL of the service
        timeout: Request timeout in seconds
        
    Note:
        This class should not be imported directly by SDK users.
        Use ModelDownloadSDK instead.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8200",
        timeout: float = 30.0,
        verify_ssl: bool | str = True,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        proxy_url: Optional[str] = None,
    ) -> None:
        """
        Initialize ModelDownloadClient.
        
        Args:
            base_url: Base URL of the service
            timeout: Request timeout in seconds
            verify_ssl: SSL verification (True, False, or path to CA bundle)
            headers: Additional HTTP headers
            cookies: HTTP cookies
            proxy_url: Proxy URL for requests
            
        Raises:
            ValidationError: If base_url is invalid
        """
        self.base_url = self._validate_base_url(base_url)
        self.timeout = timeout
        
        logger.debug(
            f"Initializing ModelDownloadClient: base_url={self.base_url}, "
            f"timeout={timeout}s, verify_ssl={verify_ssl}"
        )
        
        # Import and initialize generated client
        try:
            from model_download_service_api_client.client import Client
            
            # Map SSL parameter to httpx format
            ssl_context = None
            if isinstance(verify_ssl, str):
                # Path to CA bundle
                ssl_context = verify_ssl
            elif verify_ssl is True:
                ssl_context = True
            elif verify_ssl is False:
                ssl_context = False
            
            self._gen_client = Client(
                base_url=self.base_url,
                timeout=timeout,
                verify_ssl=ssl_context,
                headers=headers or {},
                cookies=cookies or {},
                httpx_args={"proxy": proxy_url} if proxy_url else {},
            )
            
            self._wrapper = GeneratedClientWrapper(self._gen_client)
            
            logger.info(f"ModelDownloadClient initialized successfully")
            
        except ImportError as e:
            raise ValidationError(
                f"Failed to import generated client: {e}",
                field="generated_client",
            )
        except Exception as e:
            raise ValidationError(
                f"Failed to initialize generated client: {e}",
                field="initialization",
            )

    def _validate_base_url(self, url: str) -> str:
        """
        Validate and normalize base URL.
        
        Args:
            url: Base URL
            
        Returns:
            Validated base URL
            
        Raises:
            ValidationError: If URL is invalid
        """
        if not url:
            raise ValidationError("base_url cannot be empty", field="base_url")
        
        if not isinstance(url, str):
            raise ValidationError(
                f"base_url must be string, got {type(url).__name__}",
                field="base_url",
            )
        
        # Ensure it has a scheme
        if not url.startswith(("http://", "https://")):
            raise ValidationError(
                "base_url must start with http:// or https://",
                field="base_url",
                value=url,
            )
        
        # Remove trailing slash
        return url.rstrip("/")

    def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the service.
        
        Returns:
            Health status information:
                - status: Health status
                - message: Status message
                
        Raises:
            SDKConnectionError: If unable to connect
        """
        logger.debug("Calling health_check endpoint")
        
        try:
            response = self._wrapper.call_health_check()
            
            if response is None:
                raise SDKConnectionError("Health check returned no response")
            
            # Convert response to dict
            result = {
                "status": getattr(response, "status", "unknown"),
                "message": getattr(response, "message", ""),
            }
            
            logger.debug(f"Health check result: {result}")
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in health_check: {e}")
            raise map_network_error(e, "health_check")
        except Exception as e:
            logger.error(f"Error in health_check: {e}")
            raise map_generated_error(e, "health_check")

    def download_models(
        self,
        models: List[ModelSpec],
        download_path: str,
    ) -> List[str]:
        """
        Request model download.
        
        Args:
            models: List of models to download
            download_path: Directory to save models
            
        Returns:
            List of job IDs created
            
        Raises:
            ValidationError: If inputs invalid
            SDKConnectionError: If connection fails
            JobError: If server rejects request
        """
        logger.debug(
            f"Calling download_models: {len(models)} models, "
            f"path={download_path}"
        )
        
        if not models:
            raise ValidationError(
                "At least one model must be specified",
                field="models",
            )
        
        try:
            # Convert SDK models to generated format
            models_dict = []
            for spec in models:
                model_dict = sdk_model_spec_to_generated(spec)
                models_dict.append(model_dict)
            
            logger.debug(f"Converted {len(models_dict)} models to API format")
            
            # Call generated endpoint
            response = self._wrapper.call_download_models(models_dict, download_path)
            
            if response is None:
                raise SDKConnectionError("Download request returned no response")
            
            # Extract job IDs
            job_ids = generated_download_response_to_job_ids(response)
            
            if not job_ids:
                logger.warning("Download endpoint returned no job IDs")
            
            logger.info(f"Download initiated: {len(job_ids)} jobs created")
            return job_ids
            
        except ValidationError:
            raise
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in download_models: {e}")
            raise map_network_error(e, "download_models")
        except Exception as e:
            logger.error(f"Error in download_models: {e}")
            raise map_generated_error(e, "download_models")

    def get_job_status(self, job_id: str) -> Job:
        """
        Get the status of a download/upload job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Job with current status
            
        Raises:
            NotFoundError: If job not found
            SDKConnectionError: If connection fails
        """
        logger.debug(f"Calling get_job_status for job {job_id}")
        
        if not job_id:
            raise ValidationError("job_id cannot be empty", field="job_id")
        
        try:
            response = self._wrapper.call_get_job_status(job_id)
            
            if response is None:
                logger.error(f"Job {job_id} not found")
                raise NotFoundError(
                    f"Job {job_id} not found",
                    resource_id=job_id,
                    resource_type="job",
                )
            
            job = generated_job_to_sdk(response)
            logger.debug(f"Job {job_id} status: {job.status}")
            return job
            
        except NotFoundError:
            raise
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in get_job_status: {e}")
            raise map_network_error(e, f"get_job_status({job_id})")
        except Exception as e:
            logger.error(f"Error in get_job_status: {e}")
            raise map_generated_error(e, f"get_job_status({job_id})")

    def list_jobs(self) -> List[Job]:
        """
        List all jobs.
        
        Returns:
            List of Job objects
            
        Raises:
            SDKConnectionError: If connection fails
        """
        logger.debug("Calling list_jobs")
        
        try:
            response = self._wrapper.call_list_jobs()
            
            if response is None:
                logger.warning("list_jobs returned no response")
                return []
            
            # Handle different response formats
            if isinstance(response, list):
                jobs = [generated_job_to_sdk(j) for j in response]
            elif hasattr(response, "jobs"):
                # Response might wrap jobs in a property
                jobs = [generated_job_to_sdk(j) for j in response.jobs]
            elif hasattr(response, "__iter__"):
                jobs = [generated_job_to_sdk(j) for j in response]
            else:
                logger.warning(f"Unexpected list_jobs response format: {type(response)}")
                return []
            
            logger.info(f"Retrieved {len(jobs)} jobs")
            return jobs
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in list_jobs: {e}")
            raise map_network_error(e, "list_jobs")
        except Exception as e:
            logger.error(f"Error in list_jobs: {e}")
            raise map_generated_error(e, "list_jobs")

    def get_model_results(self) -> List[Job]:
        """
        Get completed model download/upload results.
        
        Returns:
            List of completed Job objects
            
        Raises:
            SDKConnectionError: If connection fails
        """
        logger.debug("Calling get_model_results")
        
        try:
            response = self._wrapper.call_get_model_results()
            
            if response is None:
                logger.warning("get_model_results returned no response")
                return []
            
            # Handle different response formats
            if isinstance(response, list):
                jobs = [generated_job_to_sdk(j) for j in response]
            elif hasattr(response, "results"):
                jobs = [generated_job_to_sdk(j) for j in response.results]
            elif hasattr(response, "jobs"):
                jobs = [generated_job_to_sdk(j) for j in response.jobs]
            elif hasattr(response, "__iter__"):
                jobs = [generated_job_to_sdk(j) for j in response]
            else:
                logger.warning(f"Unexpected get_model_results response format: {type(response)}")
                return []
            
            logger.info(f"Retrieved {len(jobs)} completed results")
            return jobs
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in get_model_results: {e}")
            raise map_network_error(e, "get_model_results")
        except Exception as e:
            logger.error(f"Error in get_model_results: {e}")
            raise map_generated_error(e, "get_model_results")

    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List available plugins.
        
        Returns:
            List of plugin information
            
        Raises:
            SDKConnectionError: If connection fails
        """
        logger.debug("Calling list_plugins")
        
        try:
            response = self._wrapper.call_list_plugins()
            
            if response is None:
                logger.warning("list_plugins returned no response")
                return []
            
            # Normalize /plugins response into a flat list while preserving group type
            # (e.g. downloader/converter) from available_plugins mapping.
            result: List[Dict[str, Any]] = []

            def is_unset(value: Any) -> bool:
                return value is not None and value.__class__.__name__ == "Unset"

            def plugin_to_dict(plugin: Any) -> Dict[str, Any]:
                if isinstance(plugin, dict):
                    item = dict(plugin)
                elif hasattr(plugin, "to_dict"):
                    item = dict(plugin.to_dict())
                else:
                    item = {
                        "name": getattr(plugin, "name", "unknown"),
                        "version": getattr(plugin, "version", "unknown"),
                        "description": getattr(plugin, "description", ""),
                    }

                # Generated model field is type_ in attrs but serialized as "type".
                if "type" not in item and "type_" in item:
                    item["type"] = item.pop("type_")
                return item

            # Preferred shape: {"available_plugins": {"downloader": [...], "converter": [...]}}
            available_plugins = None
            if isinstance(response, dict):
                available_plugins = response.get("available_plugins")
            elif hasattr(response, "available_plugins"):
                available_plugins = getattr(response, "available_plugins")

            if available_plugins is not None and not is_unset(available_plugins):
                grouped_plugins: Dict[str, Any] = {}
                if isinstance(available_plugins, dict):
                    grouped_plugins = available_plugins
                elif hasattr(available_plugins, "additional_properties"):
                    grouped_plugins = getattr(available_plugins, "additional_properties") or {}
                elif hasattr(available_plugins, "to_dict"):
                    grouped_plugins = available_plugins.to_dict()

                for group_name, plugins in grouped_plugins.items():
                    if plugins is None:
                        continue
                    for plugin in plugins:
                        plugin_dict = plugin_to_dict(plugin)
                        if not plugin_dict.get("type"):
                            plugin_dict["type"] = group_name
                        result.append(plugin_dict)
            else:
                # Legacy fallback shapes
                plugins = []
                if isinstance(response, list):
                    plugins = response
                elif hasattr(response, "plugins"):
                    resp_plugins = getattr(response, "plugins")
                    if not is_unset(resp_plugins) and resp_plugins is not None:
                        plugins = resp_plugins
                elif hasattr(response, "__iter__") and not isinstance(response, dict):
                    plugins = list(response)

                for plugin in plugins:
                    result.append(plugin_to_dict(plugin))
            
            logger.info(f"Retrieved {len(result)} plugins")
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in list_plugins: {e}")
            raise map_network_error(e, "list_plugins")
        except Exception as e:
            logger.error(f"Error in list_plugins: {e}")
            raise map_generated_error(e, "list_plugins")

    def close(self) -> None:
        """Close the underlying HTTP client."""
        try:
            if hasattr(self._gen_client, "close"):
                self._gen_client.close()
            logger.debug("ModelDownloadClient closed")
        except Exception as e:
            logger.warning(f"Error closing client: {e}")
