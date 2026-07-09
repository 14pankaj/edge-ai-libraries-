#!/usr/bin/env python3
"""Automatically fix common OpenAPI schema issues for client generation.

Usage:
    python scripts/fix_openapi.py \
      --input /path/to/openapi.yaml \
      --output /path/to/openapi.fixed.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}


class FixStats:
    """Collect transformation counts for reporting."""

    def __init__(self) -> None:
        self.server_defaults_stringified = 0
        self.type_arrays_normalized = 0
        self.const_to_enum = 0
        self.job_status_states_added = 0
        self.required_normalized = 0
        self.malformed_boolean_required_fixed = 0
        self.bool_flags_normalized = 0
        self.operation_ids_added = 0

    def summary(self) -> str:
        return (
            "OpenAPI fix summary:\n"
            f"  - server variable defaults stringified: {self.server_defaults_stringified}\n"
            f"  - schema type arrays normalized: {self.type_arrays_normalized}\n"
            f"  - const converted to enum: {self.const_to_enum}\n"
            f"  - job status enum states added: {self.job_status_states_added}\n"
            f"  - required normalized to list: {self.required_normalized}\n"
            f"  - malformed boolean required fields fixed: {self.malformed_boolean_required_fixed}\n"
            f"  - boolean flags normalized: {self.bool_flags_normalized}\n"
            f"  - operationId values added: {self.operation_ids_added}"
        )


def _to_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return value


def _normalize_malformed_boolean_required(value: Any) -> bool | None:
    """Normalize malformed one-item boolean-like required arrays.

    OpenAPI Parameter/RequestBody objects expect `required` as boolean.
    Some upstream specs incorrectly produce:
      - [True], ["True"], [False], ["False"]
    This converts only those malformed shapes to boolean.
    """
    if not isinstance(value, list) or len(value) != 1:
        return None

    item = value[0]
    if isinstance(item, bool):
        return item

    if isinstance(item, str):
        lowered = item.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False

    return None


def _normalize_schema_node(node: Any, stats: FixStats) -> Any:
    """Recursively normalize OpenAPI nodes for generator compatibility."""
    if isinstance(node, dict):
        # OAS 3.1 -> 3.0 style: type can be ["string", "null"].
        if "type" in node and isinstance(node["type"], list):
            types = [t for t in node["type"] if t is not None]
            has_null = "null" in types
            types = [t for t in types if t != "null"]
            if has_null:
                node["nullable"] = True
            if len(types) == 1:
                node["type"] = types[0]
            elif len(types) == 0:
                node.pop("type", None)
            else:
                # Convert multi-type into anyOf for better generator compatibility.
                node.pop("type", None)
                if "anyOf" not in node:
                    node["anyOf"] = [{"type": t} for t in types]
            stats.type_arrays_normalized += 1

        # OAS 3.1 'const' -> OAS 3.0-compatible enum singleton.
        if "const" in node and "enum" not in node:
            node["enum"] = [node["const"]]
            node.pop("const", None)
            stats.const_to_enum += 1

        # Normalize malformed required values while preserving valid OpenAPI forms.
        if "required" in node:
            malformed_bool = _normalize_malformed_boolean_required(node["required"])
            if malformed_bool is not None:
                node["required"] = malformed_bool
                stats.malformed_boolean_required_fixed += 1
            elif not isinstance(node["required"], (list, bool)):
                # Keep legacy normalization only for clearly invalid non-list/non-bool values.
                if node["required"] is None:
                    node["required"] = []
                elif isinstance(node["required"], (tuple, set)):
                    node["required"] = list(node["required"])
                else:
                    node["required"] = [str(node["required"])]
                stats.required_normalized += 1

        # Normalize common boolean flags.
        for flag in ("nullable", "readOnly", "writeOnly", "deprecated"):
            if flag in node:
                normalized = _to_bool(node[flag])
                if normalized is not node[flag]:
                    node[flag] = normalized
                    stats.bool_flags_normalized += 1

        # Recurse.
        for key, value in list(node.items()):
            node[key] = _normalize_schema_node(value, stats)
        return node

    if isinstance(node, list):
        return [_normalize_schema_node(item, stats) for item in node]

    return node


def _fix_server_defaults(spec: dict[str, Any], stats: FixStats) -> None:
    for server in spec.get("servers", []):
        variables = server.get("variables", {})
        if not isinstance(variables, dict):
            continue
        for value in variables.values():
            if isinstance(value, dict) and "default" in value and not isinstance(value["default"], str):
                value["default"] = str(value["default"])
                stats.server_defaults_stringified += 1


def _ensure_job_status_states(spec: dict[str, Any], stats: FixStats) -> None:
    """Ensure JobStatus enum contains all server states used at runtime.

    Some service deployments return `downloading` as a non-terminal in-progress
    job state. If missing from OpenAPI, generated clients fail to deserialize jobs.
    """
    components = spec.get("components")
    if not isinstance(components, dict):
        return

    schemas = components.get("schemas")
    if not isinstance(schemas, dict):
        return

    job_status = schemas.get("JobStatus")
    if not isinstance(job_status, dict):
        return

    enum_values = job_status.get("enum")
    if not isinstance(enum_values, list):
        return

    # Keep existing order stable while appending only missing valid states.
    required_states = ["pending", "downloading", "processing", "completed", "failed"]
    existing = {str(v) for v in enum_values}
    for state in required_states:
        if state not in existing:
            enum_values.append(state)
            existing.add(state)
            stats.job_status_states_added += 1


def _slugify_path(path: str) -> str:
    stripped = path.strip("/")
    if not stripped:
        return "root"
    stripped = stripped.replace("{", "by_").replace("}", "")
    stripped = re.sub(r"[^a-zA-Z0-9_]+", "_", stripped)
    stripped = re.sub(r"_+", "_", stripped).strip("_")
    return stripped or "root"


def _add_missing_operation_ids(spec: dict[str, Any], stats: FixStats) -> None:
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        return

    used_ids = set()
    for path_item in paths.values():
        if isinstance(path_item, dict):
            for method_item in path_item.values():
                if isinstance(method_item, dict) and "operationId" in method_item:
                    used_ids.add(str(method_item["operationId"]))

    for route, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method.lower() not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            if operation.get("operationId"):
                continue

            base = f"{method.lower()}_{_slugify_path(route)}"
            candidate = base
            suffix = 2
            while candidate in used_ids:
                candidate = f"{base}_{suffix}"
                suffix += 1

            operation["operationId"] = candidate
            used_ids.add(candidate)
            stats.operation_ids_added += 1


def main() -> int:
    try:
        import yaml
    except ModuleNotFoundError:
        print("ERROR: Missing dependency 'pyyaml'. Install with: pip install pyyaml")
        return 1

    parser = argparse.ArgumentParser(description="Fix OpenAPI schema issues for code generation.")
    parser.add_argument("--input", required=True, help="Input OpenAPI YAML file path.")
    parser.add_argument("--output", required=True, help="Output fixed OpenAPI YAML file path.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        print(f"ERROR: Input OpenAPI file does not exist: {input_path}")
        return 1

    with input_path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    if not isinstance(spec, dict):
        print("ERROR: OpenAPI document root must be a mapping/dictionary")
        return 1

    stats = FixStats()

    _fix_server_defaults(spec, stats)
    _ensure_job_status_states(spec, stats)
    _add_missing_operation_ids(spec, stats)
    fixed = _normalize_schema_node(spec, stats)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(fixed, handle, sort_keys=False, allow_unicode=True)

    print(f"Fixed OpenAPI written to: {output_path}")
    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())