"""
Exception classes for the Model Download SDK.

This module defines a comprehensive exception hierarchy that normalizes errors from:
- HTTP client (httpx)
- Generated API client
- SDK validation logic
- Job tracking operations

Users should catch these SDK exceptions, not underlying library exceptions.

Exception Hierarchy:
    Exception
    └── SDKError (base for all SDK exceptions)
        ├── ConnectionError (network/connection issues)
        ├── ValidationError (invalid input)
        ├── JobError (server-side job failure)
        ├── TimeoutError (operation timeout)
        ├── NotFoundError (resource not found)
        └── AuthenticationError (auth failure)
"""

from typing import Optional


class SDKError(Exception):
    """
    Base exception for all SDK errors.
    
    All exceptions raised by the SDK inherit from this class, allowing users to
    catch all SDK errors with a single except clause.
    
    Attributes:
        message (str): Human-readable error message
        error_code (Optional[str]): Error code for programmatic handling
    """

    def __init__(self, message: str, error_code: Optional[str] = None) -> None:
        """
        Initialize SDKError.
        
        Args:
            message: Descriptive error message
            error_code: Optional error code for identification
        """
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class ConnectionError(SDKError):
    """
    Raised when unable to connect to the Model Download Service.
    
    This includes network errors, timeouts, SSL/TLS issues, and service unavailability.
    
    Attributes:
        base_url (Optional[str]): The service URL that failed to connect
        original_error (Optional[Exception]): The underlying exception
    """

    def __init__(
        self,
        message: str,
        base_url: Optional[str] = None,
        original_error: Optional[Exception] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """
        Initialize ConnectionError.
        
        Args:
            message: Descriptive error message
            base_url: The service URL connection attempted to
            original_error: The underlying httpx or network exception
            error_code: Optional error code
        """
        self.base_url = base_url
        self.original_error = original_error
        super().__init__(message, error_code or "CONNECTION_ERROR")


class ValidationError(SDKError):
    """
    Raised when user input validation fails.
    
    This includes invalid model names, unsupported hubs, invalid paths, etc.
    
    Attributes:
        field (Optional[str]): The field that failed validation
        value (Optional[str]): The invalid value (may be None for sensitive data)
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """
        Initialize ValidationError.
        
        Args:
            message: Descriptive error message
            field: The field that failed validation
            value: The invalid value
            error_code: Optional error code
        """
        self.field = field
        self.value = value
        super().__init__(message, error_code or "VALIDATION_ERROR")


class JobError(SDKError):
    """
    Raised when a server-side job fails.
    
    This is raised when an async operation (download, upload, conversion) fails on the
    server side. The job completed but with a failure status.
    
    Attributes:
        job_id (str): The ID of the failed job
        status (Optional[str]): Job status (e.g., "failed", "cancelled")
        error_details (Optional[str]): Detailed error from server
    """

    def __init__(
        self,
        job_id: str,
        message: str,
        status: Optional[str] = None,
        error_details: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """
        Initialize JobError.
        
        Args:
            job_id: The ID of the failed job
            message: Descriptive error message
            status: Job status from server
            error_details: Detailed error information from server
            error_code: Optional error code
        """
        self.job_id = job_id
        self.status = status
        self.error_details = error_details
        full_message = f"Job {job_id} failed: {message}"
        if error_details:
            full_message += f"\nDetails: {error_details}"
        super().__init__(full_message, error_code or "JOB_ERROR")


class TimeoutError(SDKError):
    """
    Raised when an operation exceeds its time limit.
    
    This can occur when:
    - Waiting for a job to complete exceeds the timeout
    - An API call takes longer than configured
    - Connection establishment takes too long
    
    Attributes:
        timeout_seconds (float): The timeout that was exceeded
        operation (Optional[str]): The operation that timed out
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        operation: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """
        Initialize TimeoutError.
        
        Args:
            message: Descriptive error message
            timeout_seconds: The timeout duration that was exceeded
            operation: Description of the operation that timed out
            error_code: Optional error code
        """
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        super().__init__(message, error_code or "TIMEOUT_ERROR")


class NotFoundError(SDKError):
    """
    Raised when a requested resource is not found.
    
    This includes job IDs, models, or other resources not found on the server.
    
    Attributes:
        resource_id (Optional[str]): The ID of the resource not found
        resource_type (Optional[str]): The type of resource (e.g., "job", "model")
    """

    def __init__(
        self,
        message: str,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """
        Initialize NotFoundError.
        
        Args:
            message: Descriptive error message
            resource_id: The ID of the missing resource
            resource_type: The type of resource
            error_code: Optional error code
        """
        self.resource_id = resource_id
        self.resource_type = resource_type
        super().__init__(message, error_code or "NOT_FOUND_ERROR")


class AuthenticationError(SDKError):
    """
    Raised when authentication or authorization fails.
    
    This includes invalid credentials, expired tokens, insufficient permissions, etc.
    
    Attributes:
        auth_type (Optional[str]): Type of authentication (e.g., "bearer", "api_key")
    """

    def __init__(
        self,
        message: str,
        auth_type: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """
        Initialize AuthenticationError.
        
        Args:
            message: Descriptive error message
            auth_type: Type of authentication that failed
            error_code: Optional error code
        """
        self.auth_type = auth_type
        super().__init__(message, error_code or "AUTHENTICATION_ERROR")
