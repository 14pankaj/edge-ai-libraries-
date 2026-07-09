"""Filesystem helper tests for Windows/Linux/WSL-compatible behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_download_sdk.exceptions import ValidationError
from model_download_sdk.filesystem import (
    create_temp_working_directory,
    ensure_directory,
    get_cache_directory,
    get_download_directory,
    get_safe_path,
    normalize_path,
    validate_path,
    validate_writable,
)


def test_normalize_path_returns_absolute_path(tmp_path: Path) -> None:
    """normalize_path should return absolute pathlib.Path objects."""
    relative = Path(".") / "subdir"
    normalized = normalize_path(relative, base_dir=tmp_path)

    assert isinstance(normalized, Path)
    assert normalized.is_absolute()
    assert normalized == (tmp_path / "subdir").resolve(strict=False)


def test_ensure_directory_creates_directory(tmp_path: Path) -> None:
    """ensure_directory should create missing directories and return Path."""
    target = tmp_path / "new" / "folder"

    returned = ensure_directory(target)

    assert returned == target.resolve(strict=False)
    assert target.exists()
    assert target.is_dir()


def test_validate_path_requires_existing_path(tmp_path: Path) -> None:
    """validate_path should raise ValidationError if must_exist is true and path is missing."""
    missing = tmp_path / "missing"

    with pytest.raises(ValidationError):
        validate_path(missing, must_exist=True)


def test_validate_writable_returns_path(tmp_path: Path) -> None:
    """validate_writable should return normalized Path for writable directory."""
    path = validate_writable(tmp_path)
    assert isinstance(path, Path)
    assert path == tmp_path.resolve(strict=False)


def test_get_safe_path_creates_directory(tmp_path: Path) -> None:
    """get_safe_path should create directory when create is true."""
    target = tmp_path / "safe"

    returned = get_safe_path(target, create=True)

    assert returned.exists()
    assert returned.is_dir()


def test_get_cache_and_download_directory_use_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache and download directory helpers should be home-relative and created."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    cache_dir = get_cache_directory("sdk-test")
    download_dir = get_download_directory("models-test")

    assert cache_dir == (tmp_path / ".cache" / "sdk-test").resolve(strict=False)
    assert download_dir == (tmp_path / "Downloads" / "models-test").resolve(strict=False)
    assert cache_dir.exists() and cache_dir.is_dir()
    assert download_dir.exists() and download_dir.is_dir()


def test_create_temp_working_directory(tmp_path: Path) -> None:
    """Temporary working directories should be created and returned as Path."""
    temp_dir = create_temp_working_directory(prefix="sdk-", base_dir=tmp_path)

    assert isinstance(temp_dir, Path)
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    assert temp_dir.parent == tmp_path.resolve(strict=False)
