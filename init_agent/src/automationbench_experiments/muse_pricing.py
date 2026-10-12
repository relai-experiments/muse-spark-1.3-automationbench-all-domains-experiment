"""Reproducible OpenRouter Muse Spark 1.3 token cost estimates."""

from __future__ import annotations

from typing import Any

PRICING = {
    "source": "OpenRouter meta/muse-spark-1.3",
    "url": "https://openrouter.ai/meta/muse-spark-1.3",
    "verified_date": "2026-10-01",
    "unit": "USD per 1M tokens",
    "uncached_input": 1.25,
    "cached_input": 0.15,
    "output": 4.25,
}


def task_usage(output: dict[str, Any]) -> dict[str, Any]:
    """Use the benchmark's per-rollout token and cache counters."""
    usage = output.get("_usage") or output.get("token_usage") or {}
    perf = output.get("_perf") or {}
    if "input_tokens" not in usage or "output_tokens" not in usage:
        return {
            "input_tokens": None,
            "cached_input_tokens": None,
            "uncached_input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "estimated_cost_usd": None,
        }

    input_tokens = max(int(usage["input_tokens"] or 0), 0)
    output_tokens = max(int(usage["output_tokens"] or 0), 0)
    cached_input_tokens = min(max(int(perf.get("cached_input_tokens", 0) or 0), 0), input_tokens)
    uncached_input_tokens = input_tokens - cached_input_tokens
    estimated_cost = (
        uncached_input_tokens * PRICING["uncached_input"]
        + cached_input_tokens * PRICING["cached_input"]
        + output_tokens * PRICING["output"]
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": estimated_cost,
    }


def total_usage(task_usages: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not task_usages:
        return None
    keys = ("input_tokens", "cached_input_tokens", "uncached_input_tokens",
            "output_tokens", "total_tokens")
    totals = {key: sum(item[key] or 0 for item in task_usages) for key in keys}
    totals["estimated_cost_usd"] = (
        sum(item["estimated_cost_usd"] for item in task_usages)
        if all(item["estimated_cost_usd"] is not None for item in task_usages)
        else None
    )
    totals["priced_task_count"] = sum(item["estimated_cost_usd"] is not None for item in task_usages)
    totals["pricing_source"] = PRICING["source"]
    totals["pricing"] = PRICING
    return totals
