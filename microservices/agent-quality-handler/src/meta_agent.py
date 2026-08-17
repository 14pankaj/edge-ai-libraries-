# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Meta-agent orchestration for the configured use case.

Supports two orchestration modes (controlled by ``AGENT_MODE``):

* **routing** (default) — A router node classifies severity from detection
  data, then conditionally dispatches to the subset of specialist agents
  required.  In LLM mode the deep-agent runner delegates via
  ``create_deep_agent()``; in fallback mode the same rule-based agents run
  but only when the route includes them.

* **sequential** — The legacy linear chain Policy → Analysis → Evidence →
  Ticketing that always runs every agent.
"""

import logging
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from langgraph.graph import StateGraph, END

from .agents import policy_agent, analysis_agent, evidence_agent, ticketing_agent
from .routing.router import RoutingDecision, Severity, classify
from .routing.deep_agent_runner import run_deep_agent
from .utility.config_loader import load_config, get_use_case_id
from .utility.llm_client import is_fallback_mode
from .utility.runtime_config import load_runtime_settings

log = logging.getLogger(__name__)


class AgentState(TypedDict):
    use_case_id: str
    config: dict
    prompts_dir: str | None
    min_id: int | None
    max_id: int | None
    routing_decision: dict
    policy_result: dict
    analysis_result: dict
    evidence_result: dict
    ticket_result: dict
    errors: list[dict[str, Any]]


def _failure(agent: str, exc: Exception) -> dict[str, Any]:
    return {
        "agent": agent,
        "status": "failed",
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _validated_result(agent: str, result: Any) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError(
            f"{agent.title()} agent returned {type(result).__name__}; expected a mapping"
        )
    return dict(result)


def _failed_dependencies(
    state: AgentState, agent: str, dependencies: tuple[str, ...]
) -> AgentState | None:
    failed = [
        dependency
        for dependency in dependencies
        if any(
            error["agent"] == dependency
            and error["status"] in {"failed", "skipped"}
            for error in state["errors"]
        )
    ]
    if not failed:
        return None

    detail = {
        "agent": agent,
        "status": "skipped",
        "type": "dependency_failure",
        "message": f"Skipped because prerequisites failed: {', '.join(failed)}",
        "dependencies": failed,
    }
    log.warning("%s agent skipped: failed prerequisites %s", agent.title(), failed)
    return {**state, "errors": [*state["errors"], detail]}


def _run_policy(state: AgentState) -> AgentState:
    try:
        result = policy_agent.run(
            state["use_case_id"],
            state["config"],
            state.get("prompts_dir"),
            state.get("min_id"),
            state.get("max_id"),
        )
        return {**state, "policy_result": _validated_result("policy", result)}
    except Exception as exc:
        log.error("Policy agent failed: %s", exc)
        return {**state, "errors": [*state["errors"], _failure("policy", exc)]}


def _run_analysis(state: AgentState) -> AgentState:
    try:
        result = analysis_agent.run(
            state["use_case_id"],
            state["config"],
            state.get("prompts_dir"),
            None,
            state.get("min_id"),
            state.get("max_id"),
        )
        return {**state, "analysis_result": _validated_result("analysis", result)}
    except Exception as exc:
        log.error("Analysis agent failed: %s", exc)
        return {**state, "errors": [*state["errors"], _failure("analysis", exc)]}


def _run_evidence(state: AgentState) -> AgentState:
    try:
        result = evidence_agent.run(
            state["use_case_id"],
            state["config"],
            state.get("prompts_dir"),
            state.get("min_id"),
            state.get("max_id"),
        )
        return {**state, "evidence_result": _validated_result("evidence", result)}
    except Exception as exc:
        log.error("Evidence agent failed: %s", exc)
        return {**state, "errors": [*state["errors"], _failure("evidence", exc)]}


def _run_ticketing(state: AgentState) -> AgentState:
    skipped = _failed_dependencies(state, "ticketing", ("policy", "analysis"))
    if skipped is not None:
        return skipped
    try:
        result = ticketing_agent.run(
            state["use_case_id"],
            state["config"],
            state["policy_result"],
            state["analysis_result"],
            state.get("prompts_dir"),
        )
        return {**state, "ticket_result": _validated_result("ticketing", result)}
    except Exception as exc:
        log.error("Ticketing agent failed: %s", exc)
        return {**state, "errors": [*state["errors"], _failure("ticketing", exc)]}


def _build_graph() -> Any:
    g = StateGraph(AgentState)
    g.add_node("policy",   _run_policy)
    g.add_node("analysis", _run_analysis)
    g.add_node("evidence", _run_evidence)
    g.add_node("ticketing", _run_ticketing)

    # Keep independent work running after failures; ticketing validates its
    # explicit prerequisites before it executes.
    g.set_entry_point("policy")
    g.add_edge("policy",   "analysis")
    g.add_edge("analysis", "evidence")
    g.add_edge("evidence", "ticketing")
    g.add_edge("ticketing", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Routing graph — severity-based conditional execution
# ---------------------------------------------------------------------------


def _run_router(state: AgentState) -> AgentState:
    """Classify severity and store the routing decision."""
    try:
        decision = classify(
            state["use_case_id"],
            state["config"],
            state.get("prompts_dir"),
            state.get("min_id"),
            state.get("max_id"),
        )
        log.info(
            "Router classified severity=%s route=%s",
            decision.severity.value,
            decision.route,
        )
        return {**state, "routing_decision": decision.to_dict()}
    except Exception as exc:
        log.error("Router failed: %s — defaulting to full pipeline", exc)
        fallback_decision = RoutingDecision(
            severity=Severity.HIGH,
            reason=f"Router error ({exc}); defaulting to HIGH",
            route=["policy", "analysis", "evidence", "ticketing"],
            summary={},
        )
        return {
            **state,
            "routing_decision": fallback_decision.to_dict(),
            "errors": [*state["errors"], _failure("router", exc)],
        }


def _route_after_router(state: AgentState) -> str:
    """Conditional edge: decide next node based on routing decision."""
    route = state.get("routing_decision", {}).get("route", [])
    if "policy" in route:
        return "policy"
    if "analysis" in route:
        return "analysis"
    if "evidence" in route:
        return "evidence"
    if "ticketing" in route:
        return "ticketing"
    return "ticketing"


def _route_after_policy(state: AgentState) -> str:
    route = state.get("routing_decision", {}).get("route", [])
    if "analysis" in route:
        return "analysis"
    if "evidence" in route:
        return "evidence"
    return "ticketing"


def _route_after_analysis(state: AgentState) -> str:
    route = state.get("routing_decision", {}).get("route", [])
    if "evidence" in route:
        return "evidence"
    return "ticketing"


def _build_routing_graph() -> Any:
    """Build a LangGraph with a router node that conditionally dispatches."""
    g = StateGraph(AgentState)
    g.add_node("router", _run_router)
    g.add_node("policy", _run_policy)
    g.add_node("analysis", _run_analysis)
    g.add_node("evidence", _run_evidence)
    g.add_node("ticketing", _run_ticketing)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        _route_after_router,
        {"policy": "policy", "analysis": "analysis", "evidence": "evidence", "ticketing": "ticketing"},
    )
    g.add_conditional_edges(
        "policy",
        _route_after_policy,
        {"analysis": "analysis", "evidence": "evidence", "ticketing": "ticketing"},
    )
    g.add_conditional_edges(
        "analysis",
        _route_after_analysis,
        {"evidence": "evidence", "ticketing": "ticketing"},
    )
    g.add_edge("evidence", "ticketing")
    g.add_edge("ticketing", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Deep-agent graph — delegates tool-calling to create_deep_agent()
# ---------------------------------------------------------------------------


def _run_deep_agent_node(state: AgentState) -> AgentState:
    """Run the deep agent with routing-aware tool invocation."""
    routing_dict = state.get("routing_decision", {})
    decision = RoutingDecision(
        severity=Severity(routing_dict.get("severity", "HIGH")),
        reason=routing_dict.get("reason", ""),
        route=routing_dict.get("route", ["policy", "analysis", "evidence", "ticketing"]),
        summary={},
    )
    try:
        results = run_deep_agent(
            decision,
            state["use_case_id"],
            state["config"],
            state.get("prompts_dir"),
            state.get("min_id"),
            state.get("max_id"),
        )
        return {
            **state,
            "policy_result": results.get("policy", state.get("policy_result", {})),
            "analysis_result": results.get("analysis", state.get("analysis_result", {})),
            "evidence_result": results.get("evidence", state.get("evidence_result", {})),
            "ticket_result": results.get("ticketing", state.get("ticket_result", {})),
        }
    except Exception as exc:
        log.error("Deep agent failed: %s", exc)
        return {**state, "errors": [*state["errors"], _failure("deep_agent", exc)]}


def _build_deep_agent_graph() -> Any:
    """Build a graph: router → deep_agent (single node that orchestrates tools)."""
    g = StateGraph(AgentState)
    g.add_node("router", _run_router)
    g.add_node("deep_agent", _run_deep_agent_node)

    g.set_entry_point("router")
    g.add_edge("router", "deep_agent")
    g.add_edge("deep_agent", END)
    return g.compile()


# Module-level compiled graphs — loaded once per mode at startup.
_graphs: dict[str, Any] = {}


def get_graph(mode: str | None = None):
    """Return the compiled graph for the requested orchestration mode."""
    if mode is None:
        settings = load_runtime_settings()
        mode = settings.agent_mode

    if mode not in _graphs:
        if mode == "sequential":
            _graphs[mode] = _build_graph()
        elif mode == "routing" and not is_fallback_mode():
            # LLM mode with routing uses the deep-agent graph
            _graphs[mode] = _build_deep_agent_graph()
        else:
            # Fallback routing uses conditional edges (no deep agent)
            _graphs[mode] = _build_routing_graph()
    return _graphs[mode]


def run_pipeline(
    config_path: str | None = None,
    prompts_dir: str | None = None,
    min_id: int | None = None,
    max_id: int | None = None,
) -> dict[str, Any]:
    """Run the full multi-agent pipeline and return all agent outputs."""
    config = load_config(config_path)
    use_case_id = get_use_case_id(config)

    initial_state: AgentState = {
        "use_case_id": use_case_id,
        "config": config,
        "prompts_dir": prompts_dir,
        "min_id": min_id,
        "max_id": max_id,
        "routing_decision": {},
        "policy_result": {},
        "analysis_result": {},
        "evidence_result": {},
        "ticket_result": {},
        "errors": [],
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)
    errors = final_state.get("errors", [])
    return {
        "use_case_id": use_case_id,
        "routing": final_state.get("routing_decision", {}),
        "policy":   final_state.get("policy_result", {}),
        "analysis": final_state.get("analysis_result", {}),
        "evidence": final_state.get("evidence_result", {}),
        "ticket":   final_state.get("ticket_result", {}),
        "errors": errors,
        # Retained as a compatibility alias; structured details live in errors.
        "error": errors[0]["message"] if errors else None,
    }
