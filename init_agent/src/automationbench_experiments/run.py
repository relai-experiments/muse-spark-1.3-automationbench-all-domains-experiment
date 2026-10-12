"""Run one native AutomationBench task and translate its result for RELAI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from automationbench.domains import get_domain_dataset
from automationbench.rubric import create_rubric

from automationbench_experiments.agent import BaselineAutomationAgent
from automationbench_experiments.catalog import load_catalog, task_name
from automationbench_experiments.config import AutomationAgentConfig
from automationbench_experiments.runtime import AutomationBenchHarness

DEFAULT_MODEL = "meta/muse-spark-1.3"
EFFORTS = ("xhigh", "max")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    return str(value)


def _message_value(message: Any, key: str) -> Any:
    return message.get(key) if isinstance(message, dict) else getattr(message, key, None)


def _trace_steps(output: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for message in [*(output.get("prompt") or []), *(output.get("completion") or [])]:
        role = _message_value(message, "role") or "unknown"
        step: dict[str, Any] = {
            "step_id": len(steps) + 1,
            "source": "agent" if role == "assistant" else role,
            "message": json_safe(_message_value(message, "content") or ""),
        }
        tool_calls = _message_value(message, "tool_calls")
        if tool_calls:
            step["tool_calls"] = json_safe(tool_calls)
        tool_call_id = _message_value(message, "tool_call_id")
        if tool_call_id:
            step["tool_call_id"] = tool_call_id
        steps.append(step)
    return steps


def _final_output(output: dict[str, Any]) -> Any:
    for message in reversed(output.get("completion") or []):
        if _message_value(message, "role") == "assistant":
            return json_safe(_message_value(message, "content") or "")
    return ""


def _metric(output: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = (output.get("metrics") or {}).get(key)
    return float(value if value is not None else default)


def to_relai_result(task_id: str, output: dict[str, Any], catalog_entry: dict[str, Any]) -> dict[str, Any]:
    assertions = output.get("_assertion_results")
    if not isinstance(assertions, list):
        raise ValueError(f"{task_id}: native grader returned no assertion results")
    catalog_evaluators = catalog_entry["evaluators"]
    if len(assertions) != len(catalog_evaluators):
        raise ValueError(f"{task_id}: catalog assertion count differs from native grading")
    partial_credit = _metric(output, "partial_credit", float(output.get("reward") or 0.0))
    strict_pass = _metric(output, "task_completed_correctly", float(partial_credit == 1.0))
    evaluations: list[dict[str, Any]] = []
    scored_passes: list[float] = []
    for index, assertion in enumerate(assertions):
        spec = catalog_evaluators[index]
        expected = json.loads(spec["description"])
        if assertion.get("type") != expected.get("type"):
            raise ValueError(f"{task_id}: assertion {index + 1} type changed after catalog build")
        passed = bool(assertion.get("passed"))
        excluded = bool(assertion.get("excluded"))
        if not excluded:
            scored_passes.append(float(passed))
        outcome = "PASS" if passed else "FAIL"
        status = f"EXCLUDED ({outcome})" if excluded else outcome
        evaluations.append({
            "evaluator_id": spec["id"],
            "score": float(passed),
            "min_score": 0.0,
            "max_score": 1.0,
            "feedback": f"{status} {assertion.get('type')}: "
                        f"{json.dumps(assertion.get('params') or {}, ensure_ascii=False, default=str)}",
            "result": {
                "assertion": json_safe(assertion),
                "native_partial_credit": partial_credit,
                "task_completed_correctly": strict_pass,
            },
        })
    actual_mean = sum(scored_passes) / len(scored_passes) if scored_passes else 0.0
    if abs(actual_mean - partial_credit) > 1e-9:
        raise ValueError(f"{task_id}: scored assertion mean {actual_mean} != native score {partial_credit}")
    return {
        "schema_version": "relai.custom_run.v1",
        "task_id": task_id,
        "trace": {"steps": _trace_steps(output)},
        "final_output": _final_output(output),
        "evaluations": evaluations,
        "report": {
            "partial_credit": partial_credit,
            "task_completed_correctly": strict_pass,
            "assertion_results": json_safe(assertions),
            "usage": json_safe(output.get("_usage") or {}),
            "performance": json_safe(output.get("_perf") or {}),
            "debug": json_safe(output.get("_debug") or {}),
        },
    }


def failure_result(task_id: str, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": "relai.custom_run.v1",
        "task_id": task_id,
        "trace": {"steps": []},
        "final_output": "",
        "evaluations": [],
        "evaluation_errors": [{
            "evaluator_id": "native-grader",
            "error_type": type(error).__name__,
            "message": str(error),
        }],
        "failure": {"error_type": type(error).__name__, "message": str(error)},
    }


def add_run_config(
    result: dict[str, Any], *, model: str, effort: str, toolset: str,
    max_steps: int, benchmark_commit: str,
) -> dict[str, Any]:
    config = {
        "model": model,
        "reasoning_effort": effort,
        "toolset": toolset,
        "max_steps": max_steps,
        "benchmark_commit": benchmark_commit,
    }
    result["report"].update(config)
    for evaluation in result["evaluations"]:
        evaluation["result"]["run_config"] = config
    result["evaluations"][0]["result"]["usage"] = result["report"]["usage"]
    result["evaluations"][0]["result"]["performance"] = result["report"]["performance"]
    return result


async def run_task(
    task_id: str,
    *,
    effort: str = "xhigh",
    model: str = DEFAULT_MODEL,
    toolset: str = "api",
    max_steps: int = 50,
    catalog: dict[str, Any] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    if effort not in EFFORTS:
        raise ValueError(f"unsupported reasoning effort: {effort}")
    if model != DEFAULT_MODEL:
        raise ValueError(f"unsupported experiment model: {model}")
    catalog = catalog or load_catalog()
    entry = next((item for item in catalog["tasks"] if item["id"] == task_id), None)
    if entry is None:
        raise ValueError(f"unknown catalog task: {task_id}")
    domain = entry["metadata"]["domain"]
    dataset = get_domain_dataset(domain)
    indices = [i for i, row in enumerate(dataset) if task_name(dict(row)) == task_id]
    if len(indices) != 1:
        raise ValueError(f"expected one dataset row for {task_id}, found {len(indices)}")
    selected = dataset.select(indices)
    agent = BaselineAutomationAgent(AutomationAgentConfig(toolset=toolset))
    sampling_args = agent.build_sampling_args(model, None, "openrouter", effort, None)
    runtime = AutomationBenchHarness(
        dataset=selected, rubric=create_rubric(), max_turns=max_steps, toolset=toolset, agent=agent
    )
    if client is None:
        client = agent.build_client("openrouter", None, None, "OPENROUTER_API_KEY", None)
    results = await runtime.evaluate(
        client=client,
        model=model,
        sampling_args=sampling_args,
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
        state_columns=["_usage", "_debug", "_assertion_results", "_end_state", "_perf"],
    )
    outputs = results["outputs"]
    if len(outputs) != 1:
        raise ValueError(f"expected one rollout for {task_id}, found {len(outputs)}")
    output = outputs[0]
    return add_run_config(
        to_relai_result(task_id, output, entry), model=model, effort=effort,
        toolset=toolset, max_steps=max_steps,
        benchmark_commit=entry["metadata"]["benchmark_commit"],
    )


def default_effort() -> str:
    return os.environ.get("ABEXP_REASONING_EFFORT", "xhigh")


def write_result(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
