from __future__ import annotations

import pytest

from automationbench_experiments.muse_pricing import PRICING, task_usage, total_usage


def test_cache_aware_cost_and_totals() -> None:
    first = task_usage({
        "_usage": {"input_tokens": 1000, "output_tokens": 200},
        "_perf": {"cached_input_tokens": 600},
    })
    second = task_usage({
        "_usage": {"input_tokens": 100, "output_tokens": 50},
        "_perf": {"cached_input_tokens": 200},
    })
    assert first["uncached_input_tokens"] == 400
    assert first["cached_input_tokens"] == 600
    assert first["estimated_cost_usd"] == pytest.approx((400 * 1.25 + 600 * 0.15 + 200 * 4.25) / 1e6)
    assert second["cached_input_tokens"] == 100
    assert second["uncached_input_tokens"] == 0
    total = total_usage([first, second])
    assert total["input_tokens"] == 1100
    assert total["cached_input_tokens"] == 700
    assert total["uncached_input_tokens"] == 400
    assert total["estimated_cost_usd"] == pytest.approx(
        first["estimated_cost_usd"] + second["estimated_cost_usd"]
    )
    assert total["pricing"] == PRICING


def test_missing_usage_does_not_look_free() -> None:
    unknown = task_usage({"_perf": {"cached_input_tokens": 10}})
    known = task_usage({"_usage": {"input_tokens": 0, "output_tokens": 0}})
    assert unknown["estimated_cost_usd"] is None
    assert known["estimated_cost_usd"] == 0.0
    total = total_usage([unknown, known])
    assert total["estimated_cost_usd"] is None
    assert total["priced_task_count"] == 1
