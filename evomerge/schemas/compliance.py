"""Compliance schemas — mirrors wasmagent-js/packages/compliance TypeScript interfaces."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ConstraintLevel(str, Enum):
    hard = "hard"
    soft = "soft"


class ConstraintCategory(str, Enum):
    format = "format"
    content = "content"
    style = "style"
    tool = "tool"
    state = "state"
    security = "security"
    semantic = "semantic"


class RepairStrategy(str, Enum):
    patch = "patch"
    insert_section = "insert_section"
    regenerate_region = "regenerate_region"
    full = "full"


class ViolationStage(str, Enum):
    pre_decode = "pre_decode"
    post_decode = "post_decode"
    post_tool_call = "post_tool_call"


class RunMode(str, Enum):
    direct = "direct"
    prompt_retry = "prompt_retry"
    full_pcl = "full_pcl"


class EvidenceSpan(BaseModel):
    region_id: str | None = None
    json_pointer: str | None = None
    # half-open [start, end) character range
    char_range: tuple[int, int] | None = None
    # inclusive [start, end] line range, 1-indexed
    line_range: tuple[int, int] | None = None


class RepairPolicy(BaseModel):
    strategy: RepairStrategy
    target_region: str | None = None
    max_rounds: int | None = None


class ConstraintIR(BaseModel):
    id: str
    description: str
    verify_method: str
    arg: Any = None
    path: str | None = None
    level: ConstraintLevel
    priority: int = 50
    category: ConstraintCategory
    repair: RepairPolicy | None = None


class ConstraintViolation(BaseModel):
    constraint_id: str
    level: ConstraintLevel
    category: ConstraintCategory
    hint: str
    evidence_span: EvidenceSpan | None = None
    detected_at: ViolationStage


class RepairTraceEntry(BaseModel):
    round: int
    violation_ids: list[str]
    strategy: RepairStrategy
    target_region: str | None = None
    ok: bool
    rolled_back: bool = False
    remaining_violation_ids: list[str] = Field(default_factory=list)
    token_cost: dict[str, int] | None = None
    latency_ms: float | None = None


class ToolPolicy(BaseModel):
    allowed: list[str]
    denied: list[str] = Field(default_factory=list)


class TaskSpecRepairConfig(BaseModel):
    max_rounds: int = 3
    default_strategy: RepairStrategy = RepairStrategy.patch


class TaskSpecTraceConfig(BaseModel):
    record_constraint_eval: bool = True
    record_tool_calls: bool = True
    record_repairs: bool = True


class TaskSpec(BaseModel):
    id: str
    intent: str
    language: str = "en"
    audience: str | None = None
    constraints: list[ConstraintIR] = Field(default_factory=list)
    priority_hierarchy: list[str] = Field(default_factory=list)
    tools: ToolPolicy | None = None
    repair: TaskSpecRepairConfig = Field(default_factory=TaskSpecRepairConfig)
    trace: TaskSpecTraceConfig = Field(default_factory=TaskSpecTraceConfig)


class TokenCost(BaseModel):
    prompt: int | None = None
    generation: int | None = None
    repair: int | None = None


class ComplianceError(BaseModel):
    kind: Literal[
        "model_error", "verifier_error", "repair_error", "workspace_error", "unknown"
    ]
    message: str
    stage: Literal["generate", "verify", "repair", "write"]


class ComplianceEvalRecord(BaseModel):
    task_id: str
    task_spec_hash: str
    model: str
    mode: RunMode
    violations: list[ConstraintViolation] = Field(default_factory=list)
    repair_trace: list[RepairTraceEntry] = Field(default_factory=list)
    repair_rounds: int = 0
    final_pass: bool
    token_cost: TokenCost = Field(default_factory=TokenCost)
    latency_ms: float = 0.0
    artifact: str
    error: ComplianceError | None = None


__all__ = [
    "ComplianceError",
    "ComplianceEvalRecord",
    "ConstraintCategory",
    "ConstraintIR",
    "ConstraintLevel",
    "ConstraintViolation",
    "EvidenceSpan",
    "RepairPolicy",
    "RepairStrategy",
    "RepairTraceEntry",
    "RunMode",
    "TaskSpec",
    "TaskSpecRepairConfig",
    "TaskSpecTraceConfig",
    "TokenCost",
    "ToolPolicy",
    "ViolationStage",
]
