"""Convert RolloutBranchRecord list → DPO preference pairs.

Pairing strategy: within each rollout_id, pair the highest-ranked branch
(chosen) against the lowest-ranked branch (rejected).  Multiple pairs are
generated if there are more than two branches.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Sequence

from evomerge.schemas.rollout import RolloutBranchRecord
from evomerge.schemas.training import DpoTrainingRecord, Message, Provenance, SftTrainingRecord
from evomerge.pipeline.sft import _build_messages, _task_hash


def _ngram_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def to_dpo_records(
    rollouts: Sequence[RolloutBranchRecord],
) -> list[DpoTrainingRecord]:
    """Pair highest vs lowest objective_score branches per rollout.

    Branches with objective_status == "unknown" are skipped unless all
    branches in the rollout are unknown (in which case none are paired).

    Returns:
        List of DpoTrainingRecord.
    """
    by_rollout: dict[str, list[RolloutBranchRecord]] = defaultdict(list)
    for r in rollouts:
        by_rollout[r.rollout_id].append(r)

    records: list[DpoTrainingRecord] = []
    for rollout_id, branches in by_rollout.items():
        known = [b for b in branches if b.objective_status != "unknown"]
        if len(known) < 2:
            continue
        chosen_branch = max(known, key=lambda b: (b.objective_score, b.total_score))
        rejected_branch = min(known, key=lambda b: (b.objective_score, b.total_score))
        if chosen_branch.branch_index == rejected_branch.branch_index:
            continue

        prompt_msgs = _build_messages(chosen_branch)[:-1]  # strip final assistant turn
        prov = Provenance(
            source="wasmagent-rollout",
            rollout_id=rollout_id,
            task_hash=_task_hash(chosen_branch.task),
            n_gram_hash=_ngram_hash(chosen_branch.final_answer),
        )
        all_msgs = _build_messages(chosen_branch)
        records.append(
            DpoTrainingRecord(
                messages=all_msgs,
                prompt_messages=prompt_msgs,
                chosen=chosen_branch.final_answer,
                rejected=rejected_branch.final_answer,
                provenance=prov,
            )
        )
    return records
