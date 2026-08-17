# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the severity router and routing graph."""

import json
import pytest

from src.routing.router import (
    Severity,
    RoutingDecision,
    SEVERITY_ROUTES,
    classify,
    _fallback_classify,
    _llm_classify,
)


# ── Fallback classification ──────────────────────────────────────────────────

_CONFIG = {
    "policy": {
        "critical_classes": ["Rupture", "Disconnect"],
    }
}


def _summary(by_class):
    return {"by_class": by_class}


def test_fallback_critical_high_confidence():
    summary = _summary([
        {"label": "Rupture", "count": 3, "avg_confidence": 0.85, "max_confidence": 0.92},
    ])
    decision = _fallback_classify(summary, _CONFIG)
    assert decision.severity == Severity.CRITICAL
    assert decision.route == SEVERITY_ROUTES[Severity.CRITICAL]


def test_fallback_critical_high_count():
    summary = _summary([
        {"label": "Obstacle", "count": 60, "avg_confidence": 0.4, "max_confidence": 0.5},
    ])
    decision = _fallback_classify(summary, _CONFIG)
    assert decision.severity == Severity.CRITICAL


def test_fallback_high_critical_class_present():
    summary = _summary([
        {"label": "Disconnect", "count": 2, "avg_confidence": 0.55, "max_confidence": 0.6},
    ])
    decision = _fallback_classify(summary, _CONFIG)
    assert decision.severity == Severity.HIGH
    assert "analysis" in decision.route


def test_fallback_medium():
    summary = _summary([
        {"label": "Deformation", "count": 8, "avg_confidence": 0.55, "max_confidence": 0.58},
    ])
    decision = _fallback_classify(summary, _CONFIG)
    assert decision.severity == Severity.MEDIUM
    assert decision.route == ["policy", "analysis", "ticketing"]


def test_fallback_low():
    summary = _summary([
        {"label": "Obstacle", "count": 2, "avg_confidence": 0.3, "max_confidence": 0.4},
    ])
    decision = _fallback_classify(summary, _CONFIG)
    assert decision.severity == Severity.LOW
    assert decision.route == ["policy", "ticketing"]


def test_fallback_empty_summary():
    decision = _fallback_classify(_summary([]), _CONFIG)
    assert decision.severity == Severity.LOW


def test_routing_decision_should_run():
    decision = RoutingDecision(
        severity=Severity.LOW,
        reason="test",
        route=["policy", "ticketing"],
        summary={},
    )
    assert decision.should_run("policy")
    assert decision.should_run("ticketing")
    assert not decision.should_run("analysis")
    assert not decision.should_run("evidence")


def test_routing_decision_to_dict():
    decision = RoutingDecision(
        severity=Severity.HIGH,
        reason="test reason",
        route=["policy", "analysis", "evidence", "ticketing"],
        summary={},
    )
    d = decision.to_dict()
    assert d["severity"] == "HIGH"
    assert d["reason"] == "test reason"
    assert d["route"] == ["policy", "analysis", "evidence", "ticketing"]


# ── LLM classification ──────────────────────────────────────────────────────

def test_llm_classify_parses_valid_json(monkeypatch):
    response = json.dumps({
        "severity": "CRITICAL",
        "reason": "Rupture detected",
        "route": ["policy", "analysis", "evidence", "ticketing"],
    })
    monkeypatch.setattr(
        "src.routing.router.llm_client.call_llm",
        lambda **kwargs: response,
    )
    monkeypatch.setattr(
        "src.routing.router.prompt_loader.get_section",
        lambda *args, **kwargs: "test prompt",
    )

    decision = _llm_classify("test-case", _summary([]), None)
    assert decision.severity == Severity.CRITICAL
    assert "policy" in decision.route


def test_llm_classify_handles_malformed_response(monkeypatch):
    monkeypatch.setattr(
        "src.routing.router.llm_client.call_llm",
        lambda **kwargs: "This is not JSON at all",
    )
    monkeypatch.setattr(
        "src.routing.router.prompt_loader.get_section",
        lambda *args, **kwargs: "test prompt",
    )

    decision = _llm_classify("test-case", _summary([]), None)
    assert decision.severity == Severity.HIGH  # safe default


def test_llm_classify_extracts_json_from_code_fence(monkeypatch):
    response = '```json\n{"severity": "LOW", "reason": "minor", "route": ["policy", "ticketing"]}\n```'
    monkeypatch.setattr(
        "src.routing.router.llm_client.call_llm",
        lambda **kwargs: response,
    )
    monkeypatch.setattr(
        "src.routing.router.prompt_loader.get_section",
        lambda *args, **kwargs: "test prompt",
    )

    decision = _llm_classify("test-case", _summary([]), None)
    assert decision.severity == Severity.LOW


# ── Routing graph integration ────────────────────────────────────────────────

def test_routing_graph_low_severity_skips_analysis_and_evidence(monkeypatch):
    from src import meta_agent

    monkeypatch.setattr(meta_agent, "_graphs", {})
    monkeypatch.setattr(meta_agent, "load_config", lambda _path: {
        "use_case_id": "case",
        "policy": {"critical_classes": ["Rupture"]},
    })
    monkeypatch.setenv("AGENT_MODE", "routing")
    monkeypatch.setenv("LLM_MODE", "fallback")

    # Router returns LOW → only policy + ticketing
    monkeypatch.setattr(
        "src.routing.router.storage_client.get_summary",
        lambda **kwargs: {"by_class": [
            {"label": "Obstacle", "count": 1, "avg_confidence": 0.3, "max_confidence": 0.35},
        ]},
    )
    monkeypatch.setattr(meta_agent.policy_agent, "run", lambda *args: {"policy": True})
    monkeypatch.setattr(
        meta_agent.analysis_agent, "run",
        lambda *args: pytest.fail("analysis should not run for LOW severity"),
    )
    monkeypatch.setattr(
        meta_agent.evidence_agent, "run",
        lambda *args: pytest.fail("evidence should not run for LOW severity"),
    )
    monkeypatch.setattr(meta_agent.ticketing_agent, "run", lambda *args: {"ticket": True})

    result = meta_agent.run_pipeline()

    assert result["routing"]["severity"] == "LOW"
    assert result["policy"] == {"policy": True}
    assert result["analysis"] == {}
    assert result["evidence"] == {}
    assert result["ticket"] == {"ticket": True}


def test_routing_graph_high_severity_runs_all_agents(monkeypatch):
    from src import meta_agent

    monkeypatch.setattr(meta_agent, "_graphs", {})
    monkeypatch.setattr(meta_agent, "load_config", lambda _path: {
        "use_case_id": "case",
        "policy": {"critical_classes": ["Rupture"]},
    })
    monkeypatch.setenv("AGENT_MODE", "routing")
    monkeypatch.setenv("LLM_MODE", "fallback")

    monkeypatch.setattr(
        "src.routing.router.storage_client.get_summary",
        lambda **kwargs: {"by_class": [
            {"label": "Rupture", "count": 5, "avg_confidence": 0.7, "max_confidence": 0.75},
        ]},
    )
    calls = []
    monkeypatch.setattr(meta_agent.policy_agent, "run", lambda *args: calls.append("policy") or {"p": 1})
    monkeypatch.setattr(meta_agent.analysis_agent, "run", lambda *args: calls.append("analysis") or {"a": 1})
    monkeypatch.setattr(meta_agent.evidence_agent, "run", lambda *args: calls.append("evidence") or {"e": 1})
    monkeypatch.setattr(meta_agent.ticketing_agent, "run", lambda *args: calls.append("ticketing") or {"t": 1})

    result = meta_agent.run_pipeline()

    assert result["routing"]["severity"] == "HIGH"
    assert calls == ["policy", "analysis", "evidence", "ticketing"]
