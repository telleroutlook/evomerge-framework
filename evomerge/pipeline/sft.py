"""Convert RolloutBranchRecord list → SFT training records.

One SFT record per branch (all branches, not just winners). The training target
is the final_answer; the conversation prefix reconstructs the tool call
sequence so the model sees exactly what the agent saw.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

from evomerge.schemas.rollout import RolloutBranchRecord
from evomerge.schemas.training import Message, Provenance, SftTrainingRecord


def _task_hash(task: str) -> str:
    return hashlib.sha256(task.encode()).hexdigest()[:16]


def _build_messages(record: RolloutBranchRecord) -> list[Message]:
    msgs: list[Message] = [Message(role="user", content=record.task)]
    for entry in record.tool_call_sequence:
        if entry.tool_name:
            msgs.append(
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {"name": entry.tool_name, "arguments": entry.arguments}
                    ],
                )
            )
            result_content = (
                str(entry.result) if entry.result is not None else (entry.error or "")
            )
            msgs.append(
                Message(role="tool", content=result_content)
            )
    msgs.append(Message(role="assistant", content=record.final_answer))
    return msgs


def to_sft_records(
    rollouts: Sequence[RolloutBranchRecord],
    *,
    only_passing: bool = True,
) -> list[SftTrainingRecord]:
    """Convert rollout branches to SFT records.

    Args:
        rollouts: branches from RolloutForkRunner.
        only_passing: when True (default) only include branches with
            objective_score == 1.  Set False to include all branches.

    Returns:
        List of SftTrainingRecord, one per qualifying branch.
    """
    records: list[SftTrainingRecord] = []
    for r in rollouts:
        if only_passing and r.objective_score != 1:
            continue
        prov = Provenance(
            source="wasmagent-rollout",
            rollout_id=r.rollout_id,
            task_hash=_task_hash(r.task),
        )
        records.append(
            SftTrainingRecord(
                messages=_build_messages(r),
                output_type="final_answer",
                provenance=prov,
            )
        )
    return records
