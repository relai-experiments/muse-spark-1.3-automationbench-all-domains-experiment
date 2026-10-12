"""Parallel cohort evaluation of the same agent used by the RELAI adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from automationbench.domains import get_domain_dataset
from automationbench.rubric import create_rubric
from datasets import concatenate_datasets
from tqdm import tqdm

from automationbench_experiments.agent import BaselineAutomationAgent
from automationbench_experiments.catalog import task_name
from automationbench_experiments.config import AutomationAgentConfig
from automationbench_experiments.muse_pricing import task_usage, total_usage
from automationbench_experiments.run import (
    DEFAULT_MODEL,
    EFFORTS,
    add_run_config,
    failure_result,
    to_relai_result,
    write_result,
)
from automationbench_experiments.runtime import AutomationBenchHarness


def selected_dataset(ids: list[str], entries: dict[str, dict[str, Any]]):
    """Load each required domain once, preserving the requested task order."""
    selected_ids = set(ids)
    domains = list(dict.fromkeys(entries[identifier]["metadata"]["domain"] for identifier in ids))
    rows: dict[str, tuple[Any, int]] = {}
    for domain in domains:
        dataset = get_domain_dataset(domain)
        for index, row in enumerate(dataset):
            identifier = task_name(dict(row))
            if identifier in selected_ids:
                if identifier in rows:
                    raise ValueError(f"duplicate dataset task: {identifier}")
                rows[identifier] = dataset, index
    missing = set(ids) - rows.keys()
    if missing:
        raise ValueError(f"catalog tasks absent from native dataset: {sorted(missing)}")
    # Dataset.select keeps the native schema and allows a mixed-domain cohort.
    parts = [rows[identifier][0].select([rows[identifier][1]]) for identifier in ids]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _score_summary(ids: list[str], rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    completed = [rows[identifier] for identifier in ids if rows[identifier]["failure"] is None]
    return {
        "task_count": len(ids),
        "completed_count": len(completed),
        "failed_count": len(ids) - len(completed),
        "mean_partial_credit": (
            sum(row["partial_credit"] for row in completed) / len(completed) if completed else None
        ),
        "strict_pass_rate": (
            sum(row["task_completed_correctly"] for row in completed) / len(completed)
            if completed else None
        ),
    }


async def evaluate_cohort(
    ids: list[str], *, catalog: dict[str, Any], output_dir: Path,
    model: str = DEFAULT_MODEL, effort: str = "xhigh", toolset: str = "api",
    max_steps: int = 50, max_concurrent: int = 10, client: Any = None,
) -> dict[str, Any]:
    if not ids:
        raise ValueError("evaluation requires at least one task")
    if effort not in EFFORTS:
        raise ValueError(f"unsupported reasoning effort: {effort}")
    if model != DEFAULT_MODEL:
        raise ValueError(f"unsupported experiment model: {model}")
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")

    entries = {item["id"]: item for item in catalog["tasks"]}
    unknown = set(ids) - entries.keys()
    if unknown:
        raise ValueError(f"unknown catalog tasks: {sorted(unknown)}")
    dataset = selected_dataset(ids, entries)
    agent = BaselineAutomationAgent(AutomationAgentConfig(toolset=toolset))
    sampling_args = agent.build_sampling_args(model, None, "openrouter", effort, None)
    runtime = AutomationBenchHarness(
        dataset=dataset, rubric=create_rubric(), max_turns=max_steps, toolset=toolset, agent=agent
    )
    if client is None:
        client = agent.build_client("openrouter", None, None, "OPENROUTER_API_KEY", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    positions = {identifier: index for index, identifier in enumerate(ids, 1)}
    rows: dict[str, dict[str, Any]] = {}
    observed_usage: list[dict[str, Any]] = []
    bar: tqdm | None = None
    started = time.monotonic()

    def save_output(output: dict[str, Any]) -> None:
        identifier = task_name(output)
        if identifier not in positions:
            raise ValueError(f"native evaluator returned unexpected task: {identifier}")
        if identifier in rows:
            raise ValueError(f"native evaluator returned task twice: {identifier}")
        token_usage = task_usage(output)
        observed_usage.append(token_usage)
        try:
            result = add_run_config(
                to_relai_result(identifier, output, entries[identifier]),
                model=model, effort=effort, toolset=toolset, max_steps=max_steps,
                benchmark_commit=entries[identifier]["metadata"]["benchmark_commit"],
            )
        except Exception as error:
            result = failure_result(identifier, error)
        if "report" in result:
            result["report"]["token_breakdown"] = token_usage
        write_result(output_dir / f"task-{positions[identifier]:04d}.json", result)
        report = result.get("report") or {}
        rows[identifier] = {
            "task_id": identifier,
            "result_file": f"task-{positions[identifier]:04d}.json",
            "partial_credit": report.get("partial_credit"),
            "task_completed_correctly": report.get("task_completed_correctly"),
            "usage": token_usage,
            "failure": result.get("failure"),
        }

    def on_start(raw_inputs, filtered_inputs) -> None:
        nonlocal bar
        bar = tqdm(total=len(ids), desc="Evaluating custom Muse agent", unit="task")

    def on_progress(all_outputs, new_outputs, metadata) -> None:
        for output in new_outputs:
            save_output(output)
        if bar is not None:
            bar.update(len(new_outputs))
            completed = [row for row in rows.values() if row["failure"] is None]
            if completed:
                bar.set_postfix({
                    "score": f"{sum(row['partial_credit'] for row in completed) / len(completed):.3f}",
                    "pass": f"{sum(row['task_completed_correctly'] for row in completed) / len(completed):.3f}",
                })

    results: dict[str, Any] | None = None
    run_error: Exception | None = None
    try:
        results = await runtime.evaluate(
            client=client, model=model, sampling_args=sampling_args,
            num_examples=len(ids), rollouts_per_example=1, max_concurrent=max_concurrent,
            state_columns=["_usage", "_debug", "_assertion_results", "_end_state", "_perf"],
            on_start=on_start, on_progress=on_progress,
        )
        # Also handle a runtime that returns outputs without invoking the callback.
        for output in results["outputs"]:
            identifier = task_name(output)
            if identifier not in rows:
                save_output(output)
    except Exception as error:
        run_error = error
    finally:
        if bar is not None:
            bar.close()

    for identifier in ids:
        if identifier not in rows:
            error = run_error or RuntimeError("native evaluator returned no rollout for this task")
            result = failure_result(identifier, error)
            write_result(output_dir / f"task-{positions[identifier]:04d}.json", result)
            rows[identifier] = {
                "task_id": identifier,
                "result_file": f"task-{positions[identifier]:04d}.json",
                "partial_credit": None,
                "task_completed_correctly": None,
                "usage": None,
                "failure": result["failure"],
            }

    domains = list(dict.fromkeys(entries[identifier]["metadata"]["domain"] for identifier in ids))
    domain_summaries = {
        domain: _score_summary(
            [identifier for identifier in ids if entries[identifier]["metadata"]["domain"] == domain],
            rows,
        )
        for domain in domains
    }
    usage = total_usage(observed_usage)
    summary = {
        "agent": "BaselineAutomationAgent",
        "model": model,
        "reasoning_effort": effort,
        "toolset": toolset,
        "max_steps": max_steps,
        "max_concurrent": max_concurrent,
        "benchmark_commit": entries[ids[0]]["metadata"]["benchmark_commit"],
        "selected_task_ids": ids,
        **_score_summary(ids, rows),
        "domains": domain_summaries,
        "duration_seconds": time.monotonic() - started,
        "usage": usage,
        "run_error": (str(run_error) if run_error else None),
        "tasks": [rows[identifier] for identifier in ids],
    }
    write_result(output_dir / "summary.json", summary)
    return summary
