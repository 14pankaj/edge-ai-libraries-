# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run-scoped capture of exact AQH-to-model messages and responses."""

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

log = logging.getLogger(__name__)

_active_trace: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "aqh_model_io_trace", default=None
)


@contextmanager
def capture_model_io(*, enabled: bool) -> Iterator[list[dict[str, Any]]]:
    """Collect model calls made in this run without sharing state across runs."""
    records: list[dict[str, Any]] = []
    token = _active_trace.set(records if enabled else None)
    try:
        yield records
    finally:
        _active_trace.reset(token)


def record_model_io(record: dict[str, Any]) -> None:
    """Attach a completed model exchange to the active run and service log."""
    records = _active_trace.get()
    if records is None:
        return
    record["sequence"] = len(records) + 1
    records.append(record)
    log.info("AQH_MODEL_IO %s", json.dumps(record, ensure_ascii=False, default=str))