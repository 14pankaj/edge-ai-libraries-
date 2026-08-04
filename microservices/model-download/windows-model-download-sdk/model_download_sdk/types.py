"""
Type definitions and utilities for the SDK.

This module provides type markers and utilities that are used throughout the SDK.
"""

from typing import TypeVar, Generic, Optional, Any


class _Unset:
    """Marker for unset/missing values from generated API responses."""

    def __repr__(self) -> str:
        return "UNSET"


# Singleton instance for unset values
UNSET = _Unset()

T = TypeVar("T")


class Response(Generic[T]):
    """
    Generic response wrapper (compatible with generated client Response).
    
    Attributes:
        status_code: HTTP status code
        content: Raw response content
        headers: Response headers
        parsed: Parsed response data
    """

    def __init__(
        self,
        status_code: int,
        content: bytes,
        headers: dict,
        parsed: Optional[T] = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers
        self.parsed = parsed
