# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Deep Agent runner — wraps specialist agents as LangChain tools and uses an
OVMS structured-output plan (rather than native LLM tool-calling, which small
locally-served models often fail to emit) to drive LLM-mode routing.

The deep agent receives a routing decision (severity + route) and directly
invokes the selected specialist agents, guaranteeing they run.
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
            result = analysis_agent.run(
                use_case_id, config, prompts_dir, None, min_id, max_id
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

    Asks the LLM for a structured execution plan (via OVMS structured output)
    and then directly invokes the specialist-agent tools for every agent in
    the routing decision's route, guaranteeing they run even if the model's
    plan is incomplete or malformed.

    Falls back to plain direct sequential execution if anything unexpected
    (e.g. import errors) prevents the structured-plan path from running.
    """
    tools = build_tools(use_case_id, config, prompts_dir, min_id, max_id)

    try:
        return _run_with_deep_agent(routing_decision, tools, config)
    except ImportError:
        log.warning(
            "langchain_openai not available; falling back to direct tool execution"
        )
        return _run_tools_directly(routing_decision, tools)


def _run_with_deep_agent(
    routing_decision: RoutingDecision,
    tools: list,
    config: dict,
) -> dict[str, Any]:
    """Execute the deep agent's plan using OVMS structured output.

    Small, locally-served models frequently fail to emit OpenAI-style
    ``tool_calls`` (e.g. they answer conversationally or use a model-specific
    tag format the server's tool parser isn't configured for), so
    ``create_deep_agent()``'s native tool-calling loop can silently invoke no
    subagents at all.

    Instead, we ask the model for a structured execution plan using OVMS's
    guided/structured output (``response_format`` json-schema enforcement, see
    https://docs.openvino.ai/2025/model-server/ovms_structured_output.html),
    which is reliably honored even by small models. We then invoke the
    specialist-agent tools ourselves, guaranteeing every agent in the routing
    decision's route actually runs regardless of what the model returns.
    """
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    from ..utility.runtime_config import load_runtime_settings

    settings = load_runtime_settings()

    model = ChatOpenAI(
        model=settings.llm_model_name,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )

    class AgentInvocation(BaseModel):
        agent: str = Field(
            description="One of: policy, analysis, evidence, ticketing"
        )
        reason: str = Field(default="", description="Why this agent runs next")

    class AgentExecutionPlan(BaseModel):
        calls: list[AgentInvocation] = Field(
            description="Ordered list of agents to invoke"
        )

    route_list = ", ".join(routing_decision.route)
    prompt = (
        f"You are an agentic predictive maintenance system. "
        f"A detection batch has been classified as {routing_decision.severity.value} severity.\n"
        f"Reason: {routing_decision.reason}\n\n"
        f"The allowed agents to execute, in the required order, are: {route_list}.\n"
        f"Return the execution plan as an ordered list of agent calls."
    )

    ordered_agents: list[str] = []
    try:
        plan = model.with_structured_output(AgentExecutionPlan).invoke(prompt)
        allowed = set(routing_decision.route)
        seen: set[str] = set()
        for call in plan.calls:
            if call.agent in allowed and call.agent not in seen:
                seen.add(call.agent)
                ordered_agents.append(call.agent)
    except Exception as exc:
        log.warning(
            "Structured execution plan generation failed (%s); using routing order",
            exc,
        )

    # Guarantee every agent in the routing decision runs, even if the model's
    # plan omitted some or structured output generation failed entirely.
    for agent_name in routing_decision.route:
        if agent_name not in ordered_agents:
            ordered_agents.append(agent_name)

    forced_route = RoutingDecision(
        severity=routing_decision.severity,
        reason=routing_decision.reason,
        route=ordered_agents,
        summary=routing_decision.summary,
    )
    results = _run_tools_directly(forced_route, tools)
    return {"routing": routing_decision.to_dict(), **results}


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
