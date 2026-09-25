# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for normalizing LLM-produced agent routes.

Both ``router.py`` (severity classification) and ``deep_agent_runner.py``
(execution planning) let an LLM choose which specialist agents run and in
what order. Neither the router's route nor the deep agent's plan is
guaranteed to be well-formed: the model may return unknown agent names,
duplicate an agent, or place ``ticketing`` before the ``policy``/``analysis``
results it depends on. ``normalize_route`` fixes all three cases while
otherwise preserving the model's intended ordering (e.g. running ``evidence``
before or after ``analysis`` is left untouched).
"""

from __future__ import annotations

KNOWN_AGENTS = frozenset({"policy", "analysis", "evidence", "ticketing"})

# Agents whose results ``ticketing`` reads (see ticketing_agent.run /
# deep_agent_runner's run_ticketing_agent tool). If any of these are present
# in a route, ticketing must execute after all of them.
_TICKETING_DEPENDENCIES = ("policy", "analysis")


def normalize_route(route: list[str]) -> list[str]:
    """Return a deduplicated, dependency-safe copy of ``route``.

    - Unknown agent names are dropped.
    - Duplicates are removed, keeping each agent's first occurrence.
    - ``ticketing`` is moved to the end (after every other agent still in the
      route) whenever it is present, since it depends on ``policy`` and
      ``analysis`` output and must not run before them.
    """
    deduped: list[str] = []
    seen: set[str] = set()
    for agent in route:
        if agent in KNOWN_AGENTS and agent not in seen:
            seen.add(agent)
            deduped.append(agent)

    if "ticketing" not in seen:
        return deduped

    if not any(dependency in seen for dependency in _TICKETING_DEPENDENCIES):
        return deduped

    others = [agent for agent in deduped if agent != "ticketing"]
    others.append("ticketing")
    return others
