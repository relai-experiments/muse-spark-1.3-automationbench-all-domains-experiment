"""Exercise the actual pinned task runner and rubric without a model API call."""

from __future__ import annotations

import pytest
from verifiers.types import ClientConfig, Response, ResponseMessage, ToolCall, Usage

from automationbench_experiments.catalog import load_catalog
from automationbench_experiments.client import OpenRouterChatCompletionsClient
from automationbench_experiments.evaluate import evaluate_cohort
from automationbench_experiments.run import run_task
from automationbench_experiments.runtime import AutomationBenchHarness


class StubClient(OpenRouterChatCompletionsClient):
    async def get_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        return Response(
            id="stub-response",
            created=0,
            model=model,
            usage=Usage(prompt_tokens=1, reasoning_tokens=0, completion_tokens=1, total_tokens=2),
            message=ResponseMessage(content="Done", finish_reason="stop", is_truncated=False),
        )


class ToolClient(StubClient):
    def __init__(self, config):
        super().__init__(config)
        self.calls = 0

    async def get_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return Response(
                id="stub-tool-response",
                created=0,
                model=model,
                usage=Usage(prompt_tokens=1, reasoning_tokens=0, completion_tokens=1, total_tokens=2),
                message=ResponseMessage(
                    content=None,
                    finish_reason="tool_calls",
                    is_truncated=False,
                    tool_calls=[ToolCall(id="search-1", name="api_search", arguments='{"query":"gmail"}')],
                ),
            )
        return await super().get_response(prompt, model, sampling_args, tools, **kwargs)


async def test_native_task_grades_through_custom_adapter(monkeypatch) -> None:
    monkeypatch.setenv("ABEXP_TEST_KEY", "test-only")
    catalog = load_catalog()
    task_id = next(item["id"] for item in catalog["tasks"] if item["metadata"]["domain"] == "simple")
    client = StubClient(ClientConfig(api_key_var="ABEXP_TEST_KEY", api_base_url="http://localhost"))
    result = await run_task(task_id, catalog=catalog, client=client)
    assert result["schema_version"] == "relai.custom_run.v1"
    assert result["task_id"] == task_id
    assert result["trace"]["steps"]
    assert result["evaluations"]
    assert 0.0 <= result["report"]["partial_credit"] <= 1.0
    assert result["report"]["task_completed_correctly"] in (0.0, 1.0)


async def test_old_agent_executes_native_api_tool(monkeypatch) -> None:
    monkeypatch.setenv("ABEXP_TEST_KEY", "test-only")
    catalog = load_catalog()
    task_id = next(item["id"] for item in catalog["tasks"] if item["metadata"]["domain"] == "simple")
    client = ToolClient(ClientConfig(api_key_var="ABEXP_TEST_KEY", api_base_url="http://localhost"))
    result = await run_task(task_id, catalog=catalog, client=client, max_steps=3)
    assert client.calls == 2
    assert any(step.get("tool_calls") for step in result["trace"]["steps"])
    assert any(step["source"] == "tool" for step in result["trace"]["steps"])


@pytest.mark.parametrize("domain", ["sales", "marketing", "operations", "support", "finance", "hr"])
async def test_each_scored_domain_runs_with_native_grader(monkeypatch, domain: str) -> None:
    monkeypatch.setenv("ABEXP_TEST_KEY", "test-only")
    catalog = load_catalog()
    task_id = next(item["id"] for item in catalog["tasks"] if item["metadata"]["domain"] == domain)
    client = StubClient(ClientConfig(api_key_var="ABEXP_TEST_KEY", api_base_url="http://localhost"))
    result = await run_task(task_id, catalog=catalog, client=client)
    assert result["evaluations"]
    assert result["report"]["benchmark_commit"] == catalog["tasks"][0]["metadata"]["benchmark_commit"]


async def test_cohort_uses_one_native_evaluation_and_preserves_task_mapping(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("ABEXP_TEST_KEY", "test-only")
    catalog = load_catalog()
    ids = [item["id"] for item in catalog["tasks"] if item["metadata"]["domain"] == "simple"][:2]
    client = StubClient(ClientConfig(api_key_var="ABEXP_TEST_KEY", api_base_url="http://localhost"))
    original = AutomationBenchHarness.evaluate
    calls = []

    async def recording_evaluate(self, **kwargs):
        calls.append((len(self.get_dataset()), kwargs["num_examples"], kwargs["max_concurrent"]))
        return await original(self, **kwargs)

    monkeypatch.setattr(AutomationBenchHarness, "evaluate", recording_evaluate)
    summary = await evaluate_cohort(
        ids, catalog=catalog, output_dir=tmp_path, max_concurrent=2, client=client
    )
    assert calls == [(2, 2, 2)]
    assert summary["selected_task_ids"] == ids
    assert summary["task_count"] == 2
    assert summary["failed_count"] == 0
    for index, task_id in enumerate(ids, 1):
        import json

        saved = json.loads((tmp_path / f"task-{index:04d}.json").read_text())
        assert saved["task_id"] == task_id
        assert saved["trace"]["steps"]
        assert saved["evaluations"]
    assert (tmp_path / "summary.json").exists()
