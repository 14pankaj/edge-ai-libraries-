"""
Error mapping and handling for SDK client operations.

Maps HTTP errors and generated client errors to SDK exception hierarchy.
"""

import logging
from typing import Any, Optional

from model_download_sdk.exceptions import (
    SDKError,
    ConnectionError as SDKConnectionError,
    ValidationError,
    JobError,
    NotFoundError,
    AuthenticationError,
)

logger = logging.getLogger(__name__)


def map_http_error(
    status_code: int,
    error_response: Any,
    operation: str = "API call",
) -> SDKError:
    """
    Map HTTP status code and error response to SDK exception.
    
    Args:
        status_code: HTTP status code
        error_response: Error response from API
        operation: Description of the operation that failed
        
    Returns:
        Appropriate SDK exception
    """
    error_message = extract_error_message(error_response)

    if status_code == 400:
        return ValidationError(
            f"{operation}: Bad request - {error_message}",
            field="request",
        )

    elif status_code == 401:
        return AuthenticationError(
            f"{operation}: Authentication failed - {error_message}",
            auth_type="bearer",
        )

    elif status_code == 403:
        return AuthenticationError(
            f"{operation}: Access denied - {error_message}",
            auth_type="bearer",
        )

    elif status_code == 404:
        return NotFoundError(
            f"{operation}: Resource not found - {error_message}",
            resource_type="model_or_job",
        )

    elif status_code == 422:
        return ValidationError(
            f"{operation}: Invalid request parameters - {error_message}",
            field="request_body",
        )

    elif status_code >= 500:
        return SDKConnectionError(
            f"{operation}: Server error ({status_code}) - {error_message}",
        )

    else:
        return SDKConnectionError(
            f"{operation}: HTTP {status_code} - {error_message}",
        )


def map_network_error(exception: Exception, operation: str = "API call") -> SDKConnectionError:
    """
    Map network/connection exceptions to SDK ConnectionError.
    
    Args:
        exception: The exception that was raised
        operation: Description of the operation
        
    Returns:
        SDKConnectionError with context
    """
    import httpx

    error_message = str(exception)

    if isinstance(exception, httpx.TimeoutException):
        return SDKConnectionError(
            f"{operation}: Request timeout - {error_message}",
            original_error=exception,
        )

    elif isinstance(exception, httpx.ConnectError):
        return SDKConnectionError(
            f"{operation}: Connection failed - {error_message}",
            original_error=exception,
        )

    elif isinstance(exception, httpx.HTTPError):
        return SDKConnectionError(
            f"{operation}: HTTP error - {error_message}",
            original_error=exception,
        )

    else:
        return SDKConnectionError(
            f"{operation}: {type(exception).__name__} - {error_message}",
            original_error=exception,
        )


def map_generated_error(exception: Exception, operation: str = "API call") -> SDKError:
    """
    Map generated client exceptions to SDK exceptions.
    
    Args:
        exception: Exception from generated client
        operation: Description of the operation
        
    Returns:
        Appropriate SDK exception
    """
    error_message = str(exception)
    exc_type = type(exception).__name__

    logger.debug(f"Mapping generated error: {exc_type} - {error_message}")

    # Handle UnexpectedStatus from generated client
    if exc_type == "UnexpectedStatus":
        status_code = getattr(exception, "status_code", 500)
        response_content = getattr(exception, "content", b"")
        return map_http_error(status_code, response_content, operation)

    # Handle timeout errors
    elif "timeout" in error_message.lower():
        return SDKConnectionError(
            f"{operation}: Timeout - {error_message}",
            original_error=exception,
        )

    # Default to generic SDK error
    else:
        return SDKConnectionError(
            f"{operation}: {error_message}",
            original_error=exception,
        )


def extract_error_message(error_response: Any) -> str:
    """
    Extract error message from various error response formats.
    
    Args:
        error_response: Error response object or string
        
    Returns:
        Error message string
    """
    if isinstance(error_response, str):
        return error_response

    if isinstance(error_response, bytes):
        try:
            return error_response.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary error response: {len(error_response)} bytes>"

    if isinstance(error_response, dict):
        for field in ["detail", "message", "error", "description", "msg"]:
            if field in error_response:
                return str(error_response[field])
        return str(error_response)

    # Try common attributes
    for field in ["detail", "message", "error", "description"]:
        value = getattr(error_response, field, None)
        if value:
            return str(value)

    return str(error_response)
