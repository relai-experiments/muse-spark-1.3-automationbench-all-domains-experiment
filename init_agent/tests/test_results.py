from __future__ import annotations

import pytest

from automationbench_experiments.run import to_relai_result


def _entry() -> dict:
    return {
        "evaluators": [
            {"id": "assertion-001", "description": '{"type":"first"}'},
            {"id": "assertion-002", "description": '{"type":"second"}'},
            {"id": "assertion-003", "description": '{"type":"third"}'},
        ]
    }


def test_scored_assertions_preserve_native_partial_credit() -> None:
    output = {
        "reward": 0.5,
        "metrics": {"partial_credit": 0.5, "task_completed_correctly": 0.0},
        "_assertion_results": [
            {"type": "first", "passed": True, "excluded": False, "params": {"x": 1}},
            {"type": "second", "passed": False, "excluded": False, "params": {"x": 2}},
            {"type": "third", "passed": True, "excluded": True, "params": {"x": 3}},
        ],
        "completion": [{"role": "assistant", "content": "done"}],
    }
    result = to_relai_result("finance.example", output, _entry())
    assert [e["evaluator_id"] for e in result["evaluations"]] == ["assertion-001", "assertion-002", "assertion-003"]
    assert [e["score"] for e in result["evaluations"]] == [1.0, 0.0, 1.0]
    assert result["evaluations"][2]["feedback"].startswith("EXCLUDED (PASS)")
    assert result["evaluations"][2]["result"]["assertion"]["excluded"] is True
    assert result["report"]["partial_credit"] == 0.5
    assert result["report"]["task_completed_correctly"] == 0.0
    assert result["final_output"] == "done"


def test_mismatch_fails_instead_of_misreporting_a_score() -> None:
    output = {
        "reward": 1.0,
        "metrics": {"partial_credit": 1.0},
        "_assertion_results": [
            {"type": "first", "passed": False, "excluded": False},
            {"type": "second", "passed": False, "excluded": True},
            {"type": "third", "passed": False, "excluded": True},
        ],
    }
    with pytest.raises(ValueError, match="scored assertion mean"):
        to_relai_result("finance.example", output, _entry())


def test_no_scored_assertions_still_reports_every_assertion() -> None:
    output = {
        "reward": 0.0,
        "metrics": {"partial_credit": 0.0, "task_completed_correctly": 0.0},
        "_assertion_results": [
            {"type": kind, "passed": index != 1, "excluded": True}
            for index, kind in enumerate(("first", "second", "third"))
        ],
    }
    result = to_relai_result("finance.example", output, _entry())
    assert [e["evaluator_id"] for e in result["evaluations"]] == [
        "assertion-001", "assertion-002", "assertion-003"
    ]
    assert [e["score"] for e in result["evaluations"]] == [1.0, 0.0, 1.0]
    assert result["evaluations"][1]["feedback"].startswith("EXCLUDED (FAIL)")
    assert result["report"]["partial_credit"] == 0.0
