"""Tests for OpenAPI fixer required-field normalization."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_fix_openapi_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "fix_openapi.py"
    spec = importlib.util.spec_from_file_location("fix_openapi", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_malformed_required_boolean_arrays_are_fixed_recursively() -> None:
    mod = _load_fix_openapi_module()
    stats = mod.FixStats()
    doc = {
        "paths": {
            "/models/download": {
                "post": {
                    "parameters": [{"name": "x", "in": "query", "required": ["True"]}],
                    "requestBody": {"required": [False]},
                }
            }
        }
    }

    fixed = mod._normalize_schema_node(doc, stats)
    param_required = fixed["paths"]["/models/download"]["post"]["parameters"][0]["required"]
    body_required = fixed["paths"]["/models/download"]["post"]["requestBody"]["required"]

    assert param_required is True
    assert body_required is False
    assert stats.malformed_boolean_required_fixed == 2


def test_schema_required_arrays_are_preserved() -> None:
    mod = _load_fix_openapi_module()
    stats = mod.FixStats()
    doc = {
        "components": {
            "schemas": {
                "ModelRequest": {
                    "type": "object",
                    "required": ["name", "hub"],
                    "properties": {"name": {"type": "string"}, "hub": {"type": "string"}},
                }
            }
        }
    }

    fixed = mod._normalize_schema_node(doc, stats)
    required_fields = fixed["components"]["schemas"]["ModelRequest"]["required"]

    assert required_fields == ["name", "hub"]
    assert stats.malformed_boolean_required_fixed == 0


def test_valid_required_boolean_is_not_modified() -> None:
    mod = _load_fix_openapi_module()
    stats = mod.FixStats()
    doc = {
        "paths": {
            "/plugins": {
                "get": {
                    "requestBody": {"required": True}
                }
            }
        }
    }

    fixed = mod._normalize_schema_node(doc, stats)
    required_flag = fixed["paths"]["/plugins"]["get"]["requestBody"]["required"]

    assert required_flag is True
    assert stats.malformed_boolean_required_fixed == 0


def test_fixer_adds_missing_downloading_job_status() -> None:
    mod = _load_fix_openapi_module()
    stats = mod.FixStats()
    doc = {
        "components": {
            "schemas": {
                "JobStatus": {
                    "type": "string",
                    "enum": ["pending", "processing", "completed", "failed"],
                }
            }
        }
    }

    mod._ensure_job_status_states(doc, stats)

    values = doc["components"]["schemas"]["JobStatus"]["enum"]
    assert "downloading" in values
    assert stats.job_status_states_added == 1


def test_fixer_keeps_job_status_when_downloading_present() -> None:
    mod = _load_fix_openapi_module()
    stats = mod.FixStats()
    doc = {
        "components": {
            "schemas": {
                "JobStatus": {
                    "type": "string",
                    "enum": ["pending", "downloading", "processing", "completed", "failed"],
                }
            }
        }
    }

    mod._ensure_job_status_states(doc, stats)

    values = doc["components"]["schemas"]["JobStatus"]["enum"]
    assert values.count("downloading") == 1
    assert stats.job_status_states_added == 0
