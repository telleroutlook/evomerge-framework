"""Tests for evomerge.eval.stat_bridge."""
from __future__ import annotations

import pytest

from evomerge.eval.metrics import EvalRecord
from evomerge.eval.stat_bridge import compare_all_groups, paired_significance


def _rec(task_id: str, group: str, final_pass: bool,
         tool_calls=0, valid_calls=0) -> EvalRecord:
    return EvalRecord(
        task_id=task_id,
        group=group,
        final_pass=final_pass,
        tool_calls_total=tool_calls,
        tool_calls_valid=valid_calls,
    )


def _group(label: str, pass_ids: set[str], all_ids: list[str]) -> list[EvalRecord]:
    return [_rec(tid, label, tid in pass_ids) for tid in all_ids]


class TestPairedSignificance:
    def setup_method(self):
        self.task_ids = [f"t{i}" for i in range(20)]

    def test_no_difference_high_p(self):
        pass_set = {f"t{i}" for i in range(10)}
        a = _group("A", pass_set, self.task_ids)
        b = _group("C", pass_set, self.task_ids)
        r = paired_significance(a, b, label_a="A", label_b="C")
        assert r.mcnemar_p == 1.0
        assert r.pass_rate_delta == 0.0

    def test_clear_improvement_low_p(self):
        pass_a = {f"t{i}" for i in range(5)}
        pass_c = {f"t{i}" for i in range(18)}
        a = _group("A", pass_a, self.task_ids)
        c = _group("C", pass_c, self.task_ids)
        r = paired_significance(a, c)
        assert r.mcnemar_p < 0.05
        assert r.pass_rate_delta > 0
        assert r.significant_at_05

    def test_n_common_correct(self):
        a = [_rec("t1", "A", True), _rec("t2", "A", False)]
        b = [_rec("t1", "B", True), _rec("t3", "B", True)]
        r = paired_significance(a, b)
        assert r.n_common == 1

    def test_no_common_raises(self):
        a = [_rec("t1", "A", True)]
        b = [_rec("t2", "B", True)]
        with pytest.raises(ValueError, match="No common task_ids"):
            paired_significance(a, b)

    def test_wilson_ci_bounds_valid(self):
        a = _group("A", set(), self.task_ids)
        b = _group("B", set(self.task_ids), self.task_ids)
        r = paired_significance(a, b)
        assert 0.0 <= r.pass_ci_a[0] <= r.pass_ci_a[1] <= 1.0
        assert 0.0 <= r.pass_ci_b[0] <= r.pass_ci_b[1] <= 1.0

    def test_tool_mcnemar_computed_when_both_have_calls(self):
        a = [_rec("t1", "A", True, tool_calls=2, valid_calls=1)]
        b = [_rec("t1", "B", True, tool_calls=2, valid_calls=2)]
        r = paired_significance(a, b)
        assert r.tool_mcnemar_p is not None

    def test_tool_mcnemar_none_when_no_calls(self):
        a = [_rec("t1", "A", True, tool_calls=0)]
        b = [_rec("t1", "B", True, tool_calls=0)]
        r = paired_significance(a, b)
        assert r.tool_mcnemar_p is None

    def test_bootstrap_keys_present(self):
        a = _group("A", {"t0", "t1"}, self.task_ids)
        b = _group("B", {"t0", "t1", "t2", "t3"}, self.task_ids)
        r = paired_significance(a, b)
        assert "delta_acc" in r.bootstrap
        assert "ci_lo" in r.bootstrap
        assert "ci_hi" in r.bootstrap

    def test_to_dict_serialisable(self):
        import json
        a = _group("A", {"t0"}, self.task_ids)
        b = _group("B", {"t0", "t1"}, self.task_ids)
        r = paired_significance(a, b)
        json.dumps(r.to_dict())

    def test_significant_at_01_flag(self):
        pass_a = {f"t{i}" for i in range(2)}
        pass_c = {f"t{i}" for i in range(19)}
        a = _group("A", pass_a, self.task_ids)
        c = _group("C", pass_c, self.task_ids)
        r = paired_significance(a, c)
        assert r.significant_at_01


class TestCompareAllGroups:
    def test_all_pairs_returned(self):
        ids = [f"t{i}" for i in range(10)]
        groups = {
            "A": _group("A", set(), ids),
            "B": _group("B", {f"t{i}" for i in range(5)}, ids),
            "C": _group("C", {f"t{i}" for i in range(8)}, ids),
        }
        results = compare_all_groups(groups, reference="A")
        assert "A_vs_B" in results
        assert "A_vs_C" in results
        assert "A_vs_A" not in results

    def test_missing_reference_raises(self):
        ids = ["t1"]
        groups = {"B": _group("B", set(), ids)}
        with pytest.raises(ValueError, match="Reference group"):
            compare_all_groups(groups, reference="A")
