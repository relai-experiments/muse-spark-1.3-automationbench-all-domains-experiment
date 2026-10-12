"""Cohort reporting must survive out-of-order completion and grading failures."""

from __future__ import annotations

import json

import pytest

from automationbench_experiments.catalog import load_catalog
from automationbench_experiments.evaluate import evaluate_cohort
from automationbench_experiments.runtime import AutomationBenchHarness


async def test_parallel_results_are_saved_by_task_id_and_failures_are_excluded(
    monkeypatch, tmp_path
) -> None:
    catalog = load_catalog()
    entries = [item for item in catalog["tasks"] if item["metadata"]["domain"] == "simple"][:2]
    ids = [item["id"] for item in entries]
    outputs = []
    for entry in entries:
        assertions = [json.loads(spec["description"]) for spec in entry["evaluators"]]
        outputs.append({
            "info": {"task_name": entry["id"]},
            "reward": 1.0,
            "metrics": {"partial_credit": 1.0, "task_completed_correctly": 1.0},
            "_assertion_results": [
                {"type": assertion["type"], "params": assertion.get("params", {}), "passed": True}
                for assertion in assertions
            ],
            "_usage": {"input_tokens": 3, "output_tokens": 2},
            "completion": [{"role": "assistant", "content": "Done"}],
        })
    # Native completion order need not match catalog order. The first task also
    # has a broken assertion payload, which must not poison the second score.
    outputs[0]["_assertion_results"][0]["type"] = "wrong-type"
    outputs[1]["_assertion_results"][0]["excluded"] = True
    outputs[1]["reward"] = 0.0
    outputs[1]["metrics"] = {"partial_credit": 0.0, "task_completed_correctly": 0.0}
    calls = []

    async def fake_evaluate(self, **kwargs):
        calls.append((kwargs["max_concurrent"], kwargs["num_examples"]))
        kwargs["on_start"]([], [])
        for output in reversed(outputs):
            kwargs["on_progress"]([], [output], {})
        return {"outputs": list(reversed(outputs)), "metadata": {}}

    monkeypatch.setattr(AutomationBenchHarness, "evaluate", fake_evaluate)
    summary = await evaluate_cohort(
        ids, catalog=catalog, output_dir=tmp_path, max_concurrent=7, client=object()
    )
    assert calls == [(7, 2)]
    assert summary["selected_task_ids"] == ids
    assert summary["completed_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["mean_partial_credit"] == 0.0
    assert summary["strict_pass_rate"] == 0.0
    assert summary["domains"]["simple"]["task_count"] == 2
    assert summary["domains"]["simple"]["completed_count"] == 1
    assert summary["usage"]["total_tokens"] == 10
    assert summary["usage"]["uncached_input_tokens"] == 6
    assert summary["usage"]["cached_input_tokens"] == 0
    assert summary["usage"]["estimated_cost_usd"] == pytest.approx(0.0000245)
    first = json.loads((tmp_path / "task-0001.json").read_text())
    second = json.loads((tmp_path / "task-0002.json").read_text())
    assert first["task_id"] == ids[0]
    assert first["failure"]["error_type"] == "ValueError"
    assert second["task_id"] == ids[1]
    assert second["trace"]["steps"]
    assert second["evaluations"][0]["score"] == 1.0
    assert second["evaluations"][0]["feedback"].startswith("EXCLUDED (PASS)")
    assert second["report"]["partial_credit"] == 0.0
    assert second["report"]["token_breakdown"]["estimated_cost_usd"] == pytest.approx(0.00001225)
    assert json.loads((tmp_path / "summary.json").read_text())["tasks"][0]["task_id"] == ids[0]


async def test_mixed_domains_have_separate_native_scores_and_failure_counts(monkeypatch, tmp_path) -> None:
    catalog = load_catalog()
    entries = {
        domain: next(item for item in catalog["tasks"] if item["metadata"]["domain"] == domain)
        for domain in ("sales", "finance", "hr")
    }
    ids = [entries[domain]["id"] for domain in entries]
    outputs = []
    for domain in ("sales", "finance"):
        entry = entries[domain]
        assertions = [json.loads(spec["description"]) for spec in entry["evaluators"]]
        excluded = domain == "finance"
        outputs.append({
            "info": {"task_name": entry["id"]},
            "reward": 0.0 if excluded else 1.0,
            "metrics": {
                "partial_credit": 0.0 if excluded else 1.0,
                "task_completed_correctly": 0.0 if excluded else 1.0,
            },
            "_assertion_results": [
                {"type": assertion["type"], "passed": True, "excluded": excluded}
                for assertion in assertions
            ],
        })

    async def fake_evaluate(self, **kwargs):
        kwargs["on_start"]([], [])
        for output in reversed(outputs):
            kwargs["on_progress"]([], [output], {})
        return {"outputs": list(reversed(outputs)), "metadata": {}}

    monkeypatch.setattr(AutomationBenchHarness, "evaluate", fake_evaluate)
    summary = await evaluate_cohort(ids, catalog=catalog, output_dir=tmp_path, client=object())
    assert summary["task_count"] == 3
    assert summary["completed_count"] == 2
    assert summary["mean_partial_credit"] == 0.5
    assert summary["strict_pass_rate"] == 0.5
    assert summary["domains"]["sales"]["mean_partial_credit"] == 1.0
    assert summary["domains"]["finance"]["mean_partial_credit"] == 0.0
    assert summary["domains"]["hr"] == {
        "task_count": 1,
        "completed_count": 0,
        "failed_count": 1,
        "mean_partial_credit": None,
        "strict_pass_rate": None,
    }
    assert json.loads((tmp_path / "summary.json").read_text())["domains"] == summary["domains"]
