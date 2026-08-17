# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Deep Agent runner — wraps specialist agents as LangChain tools and delegates
execution to ``create_deep_agent()`` for LLM-mode routing.

The deep agent receives a routing decision (severity + route) and executes
only the selected specialist agents via tool calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from ..agents import policy_agent, analysis_agent, evidence_agent, ticketing_agent
from .router import RoutingDecision

log = logging.getLogger(__name__)


def _make_policy_tool(
    use_case_id: str,
    config: dict,
    prompts_dir: str | None,
    min_id: int | None,
    max_id: int | None,
):
    """Create a bound policy-agent tool."""

    @tool
    def run_policy_agent(reason: str = "") -> str:
        """Run the policy agent to generate inspection policies from detection data.
        Call this when the routing decision includes 'policy' in the route.
        """
        try:
            result = policy_agent.run(
                use_case_id, config, prompts_dir, min_id, max_id
            )
            return json.dumps(result, default=str)
        except Exception as exc:
            log.error("Policy agent tool failed: %s", exc)
            return json.dumps({"error": str(exc), "agent": "policy"})

    return run_policy_agent


def _make_analysis_tool(
    use_case_id: str,
    config: dict,
    prompts_dir: str | None,
    min_id: int | None,
    max_id: int | None,
):
    """Create a bound analysis-agent tool."""

    @tool
    def run_analysis_agent(policy_result_json: str = "{}") -> str:
        """Run the analysis agent to produce a structured analysis report.
        Call this when the routing decision includes 'analysis' in the route.
        Pass the policy result JSON string from run_policy_agent if available.
        """
        try:
            policy_result = json.loads(policy_result_json) if policy_result_json else None
            result = analysis_agent.run(
                use_case_id, config, prompts_dir, policy_result, None, min_id, max_id
            )
            return json.dumps(result, default=str)
        except Exception as exc:
            log.error("Analysis agent tool failed: %s", exc)
            return json.dumps({"error": str(exc), "agent": "analysis"})

    return run_analysis_agent


def _make_evidence_tool(
    use_case_id: str,
    config: dict,
    prompts_dir: str | None,
    min_id: int | None,
    max_id: int | None,
):
    """Create a bound evidence-agent tool."""

    @tool
    def run_evidence_agent(reason: str = "") -> str:
        """Run the evidence agent to build an audit trail for compliance.
        Call this when the routing decision includes 'evidence' in the route.
        """
        try:
            result = evidence_agent.run(
                use_case_id, config, prompts_dir, min_id, max_id
            )
            return json.dumps(result, default=str)
        except Exception as exc:
            log.error("Evidence agent tool failed: %s", exc)
            return json.dumps({"error": str(exc), "agent": "evidence"})

    return run_evidence_agent


def _make_ticketing_tool(
    use_case_id: str,
    config: dict,
    prompts_dir: str | None,
):
    """Create a bound ticketing-agent tool."""

    @tool
    def run_ticketing_agent(
        policy_result_json: str = "{}",
        analysis_result_json: str = "{}",
    ) -> str:
        """Run the ticketing agent to generate a maintenance ticket.
        Call this when the routing decision includes 'ticketing' in the route.
        Pass policy and analysis results as JSON strings.
        """
        try:
            policy_result = json.loads(policy_result_json) if policy_result_json else {}
            analysis_result = (
                json.loads(analysis_result_json) if analysis_result_json else {}
            )
            result = ticketing_agent.run(
                use_case_id, config, policy_result, analysis_result, prompts_dir
            )
            return json.dumps(result, default=str)
        except Exception as exc:
            log.error("Ticketing agent tool failed: %s", exc)
            return json.dumps({"error": str(exc), "agent": "ticketing"})

    return run_ticketing_agent


def build_tools(
    use_case_id: str,
    config: dict,
    prompts_dir: str | None,
    min_id: int | None,
    max_id: int | None,
) -> list:
    """Build LangChain tool list for the specialist agents."""
    return [
        _make_policy_tool(use_case_id, config, prompts_dir, min_id, max_id),
        _make_analysis_tool(use_case_id, config, prompts_dir, min_id, max_id),
        _make_evidence_tool(use_case_id, config, prompts_dir, min_id, max_id),
        _make_ticketing_tool(use_case_id, config, prompts_dir),
    ]


def run_deep_agent(
    routing_decision: RoutingDecision,
    use_case_id: str,
    config: dict,
    prompts_dir: str | None,
    min_id: int | None,
    max_id: int | None,
) -> dict[str, Any]:
    """Execute the deep agent with routing-aware tool invocation.

    Uses ``create_deep_agent()`` from the deepagents library to orchestrate
    tool-calling.  The agent receives the routing decision and is instructed
    to call only the agents in the route.

    Falls back to direct sequential execution if deepagents is unavailable.
    """
    tools = build_tools(use_case_id, config, prompts_dir, min_id, max_id)

    try:
        return _run_with_deep_agent(routing_decision, tools, config)
    except ImportError:
        log.warning(
            "deepagents not available; falling back to direct tool execution"
        )
        return _run_tools_directly(routing_decision, tools)


def _run_with_deep_agent(
    routing_decision: RoutingDecision,
    tools: list,
    config: dict,
) -> dict[str, Any]:
    """Execute via create_deep_agent()."""
    from deepagents import create_deep_agent
    from ..utility.runtime_config import load_runtime_settings

    settings = load_runtime_settings()

    agent = create_deep_agent(
        model=f"openai:{settings.llm_model_name}",
        tools=tools,
        model_kwargs={
            "base_url": settings.llm_base_url,
            "api_key": settings.llm_api_key,
        },
    )

    route_list = ", ".join(routing_decision.route)
    prompt = (
        f"You are an agentic predictive maintenance system. "
        f"A detection batch has been classified as {routing_decision.severity.value} severity.\n"
        f"Reason: {routing_decision.reason}\n\n"
        f"Execute ONLY these agents in order: {route_list}.\n"
        f"Pass results between agents as needed (policy result to analysis, "
        f"policy+analysis results to ticketing).\n"
        f"Return a JSON summary of all agent outputs."
    )

    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    return _extract_results(response, routing_decision)


def _run_tools_directly(
    routing_decision: RoutingDecision,
    tools: list,
) -> dict[str, Any]:
    """Direct sequential execution as a fallback when deepagents is unavailable."""
    tool_map = {t.name: t for t in tools}
    results: dict[str, Any] = {}

    policy_json = "{}"
    analysis_json = "{}"

    for agent_name in routing_decision.route:
        tool_name = f"run_{agent_name}_agent"
        tool_fn = tool_map.get(tool_name)
        if tool_fn is None:
            continue

        if agent_name == "policy":
            raw = tool_fn.invoke({"reason": routing_decision.reason})
            policy_json = raw
        elif agent_name == "analysis":
            raw = tool_fn.invoke({"policy_result_json": policy_json})
            analysis_json = raw
        elif agent_name == "evidence":
            raw = tool_fn.invoke({"reason": routing_decision.reason})
        elif agent_name == "ticketing":
            raw = tool_fn.invoke(
                {
                    "policy_result_json": policy_json,
                    "analysis_result_json": analysis_json,
                }
            )
        else:
            continue

        try:
            results[agent_name] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            results[agent_name] = {"raw": raw}

    return results


def _extract_results(
    response: Any, routing_decision: RoutingDecision
) -> dict[str, Any]:
    """Extract structured results from the deep agent response."""
    results: dict[str, Any] = {
        "routing": routing_decision.to_dict(),
    }

    if hasattr(response, "get") and "messages" in response:
        messages = response["messages"]
        if messages:
            last = messages[-1]
            content = last.content if hasattr(last, "content") else str(last)
            try:
                parsed = json.loads(content)
                results.update(parsed)
            except (json.JSONDecodeError, TypeError):
                results["raw_output"] = content

    return results
