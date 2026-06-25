"""MCP-Atlas benchmark adapter.

Reference: https://arxiv.org/html/2602.00933v1
MCP-Atlas: 36 real MCP servers, 220 tools, 1000 tasks.
Tasks require 3-6 cross-server tool calls; graded by claims-based rubric.

This adapter converts MCP-Atlas task results to AEP records for
WasmAgent evidence scoring and training export.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPAtlasToolCall:
    """One tool call step in an MCP-Atlas trajectory."""
    step_index: int
    server_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: str | None = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class MCPAtlasClaim:
    """One grading claim in an MCP-Atlas task."""
    claim_id: str
    description: str
    passed: bool
    evidence: str = ""


@dataclass
class MCPAtlasTask:
    """One MCP-Atlas evaluation task."""
    task_id: str
    user_query: str
    required_servers: list[str]
    claims: list[MCPAtlasClaim] = field(default_factory=list)
    min_tool_calls: int = 3
    max_tool_calls: int = 6


@dataclass
class MCPAtlasResult:
    """One model response trajectory for an MCP-Atlas task."""
    task_id: str
    model_id: str
    trajectory: list[MCPAtlasToolCall]
    final_answer: str = ""
    claims_passed: int = 0
    claims_total: int = 0
    session_id: str = ""


def mcp_atlas_to_aep(result: MCPAtlasResult, task: MCPAtlasTask) -> dict[str, Any]:
    """Convert an MCPAtlasResult to an AEP record dict (aep/v0.1).

    Captures tool calls as ActionEvidence entries; claims as VerifierResult entries.
    """
    actions = []
    for step in result.trajectory:
        actions.append({
            "action_id": f"step-{step.step_index}",
            "tool_name": step.tool_name,
            "state_changing": step.result is not None and step.error is None,
            "result_digest": None,
            "evidence_refs": [],
            "timestamp_ms": step.latency_ms,
        })

    verifier_results = []
    if task.claims:
        for claim in task.claims:
            verifier_results.append({
                "verifier_id": f"mcp-atlas-claim/{claim.claim_id}",
                "passed": claim.passed,
                "score": 1.0 if claim.passed else 0.0,
                "claim_ids": [claim.claim_id],
            })
    elif result.claims_total > 0:
        verifier_results.append({
            "verifier_id": "mcp-atlas-claims",
            "passed": result.claims_passed == result.claims_total,
            "score": result.claims_passed / result.claims_total,
            "claim_ids": [],
        })

    return {
        "schema_version": "aep/v0.1",
        "run_id": f"mcp-atlas/{result.task_id}/{result.model_id}",
        "model_id": result.model_id,
        "model_provider": "benchmark",
        "input_refs": [{"uri": f"mcp-atlas/task/{result.task_id}"}],
        "output_refs": [{"uri": f"mcp-atlas/result/{result.session_id}"}],
        "capability_decisions": [],
        "actions": actions,
        "verifier_results": verifier_results,
        "created_at_ms": 0,
    }


def mcp_atlas_to_rollout(result: MCPAtlasResult, task: MCPAtlasTask) -> dict[str, Any]:
    """Convert MCPAtlasResult to rollout-wire/v1 for training export."""
    tool_calls = []
    for step in result.trajectory:
        tool_calls.append({
            "event": "tool_call",
            "data": {
                "name": f"{step.server_id}/{step.tool_name}",
                "arguments": step.arguments,
            },
        })
        obs = step.result or step.error or ""
        tool_calls.append({"event": "tool_result", "data": {"result": obs}})

    pass_rate = result.claims_passed / result.claims_total if result.claims_total else 0.0
    return {
        "schema_version": "rollout-wire/v1",
        "rollout_id": f"mcp-atlas/{result.task_id}",
        "task": task.user_query,
        "branch_index": 0,
        "temperature": 0.0,
        "session_id": result.session_id or f"mcp-atlas-{result.task_id}",
        "tool_call_sequence": tool_calls,
        "final_answer": result.final_answer,
        "build_result": None,
        "objective_score": int(pass_rate == 1.0),
        "objective_status": "pass" if pass_rate == 1.0 else "fail" if pass_rate == 0.0 else "unknown",
        "rank": 0,
        "total_score": pass_rate,
        "provenance": {
            "source": "mcp-atlas",
            "session_id": result.session_id,
            "job_id": result.task_id,
            "exported_at_ms": 0,
            "schema_version": "rollout-wire/v1",
            "evidence_source": "benchmark_graded",
            "redaction_version": "none",
        },
    }


class MCPAtlasAdapter:
    """Adapter for MCP-Atlas data import.

    Usage:
        adapter = MCPAtlasAdapter()
        records = adapter.load_jsonl("mcp_atlas_results.jsonl")
        aep_records = adapter.to_aep(records)
        rollouts = adapter.to_rollouts(records)
    """

    def load_jsonl(self, path: str) -> list[tuple[MCPAtlasResult, MCPAtlasTask]]:
        import json
        pairs = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                obj = json.loads(line)
                task = MCPAtlasTask(
                    task_id=obj["task_id"],
                    user_query=obj["user_query"],
                    required_servers=obj.get("required_servers", []),
                    claims=[MCPAtlasClaim(**c) for c in obj.get("claims", [])],
                    min_tool_calls=obj.get("min_tool_calls", 3),
                    max_tool_calls=obj.get("max_tool_calls", 6),
                )
                result = MCPAtlasResult(
                    task_id=obj["task_id"],
                    model_id=obj.get("model_id", "unknown"),
                    trajectory=[MCPAtlasToolCall(**s) for s in obj.get("trajectory", [])],
                    final_answer=obj.get("final_answer", ""),
                    claims_passed=obj.get("claims_passed", 0),
                    claims_total=obj.get("claims_total", 0),
                    session_id=obj.get("session_id", ""),
                )
                pairs.append((result, task))
        return pairs

    def to_aep(self, pairs: list[tuple[MCPAtlasResult, MCPAtlasTask]]) -> list[dict]:
        return [mcp_atlas_to_aep(r, t) for r, t in pairs]

    def to_rollouts(self, pairs: list[tuple[MCPAtlasResult, MCPAtlasTask]]) -> list[dict]:
        return [mcp_atlas_to_rollout(r, t) for r, t in pairs]
