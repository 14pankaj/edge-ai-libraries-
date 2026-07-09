"""Filesystem utilities implemented with pathlib only.

This module provides cross-platform path helpers for Windows, Linux, and WSL
without embedding OS-specific path string logic.

All public functions return pathlib.Path objects.
"""

from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

from model_download_sdk.exceptions import ValidationError


def normalize_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Normalize a user path into an absolute pathlib.Path.

    Rules:
    - expand user home
    - if relative, resolve against base_dir (or cwd)
    - resolve without requiring the target to exist
    """
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            base = Path(base_dir).expanduser() if base_dir is not None else Path.cwd()
            p = base / p
        return p.resolve(strict=False)
    except Exception as exc:
        raise ValidationError(
            f"Failed to normalize path '{path}': {exc}",
            field="path",
            value=str(path),
        )


def validate_path(
    path: str | Path,
    *,
    must_exist: bool = False,
    must_be_dir: bool = False,
    writable: bool = False,
    create: bool = False,
) -> Path:
    """Validate a path and return a normalized Path.

    Args:
        path: Path to validate
        must_exist: Require that path already exists
        must_be_dir: Require directory path (and not file)
        writable: Verify write access by creating/removing a probe file
        create: Create directory when missing (directory validation flow)
    """
    p = normalize_path(path)

    try:
        if create:
            p.mkdir(parents=True, exist_ok=True)

        if must_exist and not p.exists():
            raise ValidationError(
                f"Path does not exist: {p}",
                field="path",
                value=str(p),
            )

        if must_be_dir and p.exists() and not p.is_dir():
            raise ValidationError(
                f"Path is not a directory: {p}",
                field="path",
                value=str(p),
            )

        if writable:
            parent = p if p.is_dir() else p.parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
            probe = parent / f".sdk_write_probe_{uuid4().hex}"
            probe.touch(exist_ok=False)
            probe.unlink(missing_ok=True)

        return p
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(
            f"Path validation failed for '{p}': {exc}",
            field="path",
            value=str(p),
        )


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return normalized Path."""
    return validate_path(path, must_be_dir=True, writable=True, create=True)


def validate_writable(path: str | Path, create: bool = True) -> Path:
    """Validate that a directory is writable and return normalized Path."""
    return validate_path(path, must_be_dir=True, writable=True, create=create)


def get_safe_path(path: str | Path, create: bool = True) -> Path:
    """Normalize and validate a directory path, returning a Path object."""
    return validate_writable(path, create=create)


def get_cache_directory(app_name: str = "model-download-sdk") -> Path:
    """Create and return SDK cache directory.

    Uses a user-home scoped cache root and works across Windows/Linux/WSL.
    """
    cache_dir = Path.home() / ".cache" / app_name
    return ensure_directory(cache_dir)


def get_download_directory(subdir: str = "models") -> Path:
    """Create and return default SDK download directory as Path."""
    download_dir = Path.home() / "Downloads" / subdir
    return ensure_directory(download_dir)


def create_temp_working_directory(
    prefix: str = "model-download-",
    base_dir: str | Path | None = None,
) -> Path:
    """Create and return a temporary working directory as Path."""
    tmp_root = ensure_directory(base_dir) if base_dir is not None else None
    created = mkdtemp(prefix=prefix, dir=str(tmp_root) if tmp_root is not None else None)
    return normalize_path(created)


class FileSystem:
    """Legacy compatibility wrapper around pathlib-based helpers."""

    @staticmethod
    def default_model_dir() -> Path:
        """Get default model directory."""
        return get_download_directory("model-download")

    @staticmethod
    def ensure_directory(path: Path) -> Path:
        """Ensure directory exists and return it."""
        return ensure_directory(path)