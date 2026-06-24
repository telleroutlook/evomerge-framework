"""Rollout wire format — matches wasmagent-js rollout-wire/v1 JSON Schema."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallEntry(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None


class BuildResult(BaseModel):
    status: Literal["pass", "fail", "skip"]
    exit_code: int | None = None
    stderr: str | None = None


class RolloutBranchRecord(BaseModel):
    """One branch produced by wasmagent-js RolloutForkRunner.

    schema_version is always "rollout-wire/v1" to match the SSOT in
    wasmagent-js/packages/core/src/ranking/schemas/rollout-wire.schema.json.
    """

    schema_version: Literal["rollout-wire/v1"] = "rollout-wire/v1"
    rollout_id: str
    task: str
    branch_index: int
    temperature: float
    session_id: str
    tool_call_sequence: list[ToolCallEntry] = Field(default_factory=list)
    final_answer: str
    build_result: BuildResult | None = None
    # RolloutRanker-enriched fields
    objective_score: Literal[0, 1] = 0
    objective_status: Literal["pass", "fail", "unknown"] = "unknown"
    rank: int = 0
    total_score: float = 0.0


__all__ = ["BuildResult", "RolloutBranchRecord", "ToolCallEntry"]
