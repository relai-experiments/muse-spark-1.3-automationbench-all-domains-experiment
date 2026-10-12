from __future__ import annotations

from automationbench_experiments.agent import BaselineAutomationAgent
from automationbench_experiments.client import OpenRouterChatCompletionsClient
from automationbench_experiments.config import AutomationAgentConfig, BASELINE_SYSTEM_PROMPT


def test_muse_baseline_configuration() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    assert agent.default_model == "meta/muse-spark-1.3"
    assert agent.config.toolset == "api"
    assert agent.config.system_prompt == BASELINE_SYSTEM_PROMPT
    assert agent.resolve_api(agent.default_model, None) == "openrouter"
    for effort in ("xhigh", "max"):
        assert agent.build_sampling_args(agent.default_model, None, "openrouter", effort, None) == {
            "reasoning_effort": effort
        }


def test_openrouter_session_id_is_stable_for_a_task() -> None:
    first = OpenRouterChatCompletionsClient.session_id_for_state({"task": "finance.one", "example_id": 3})
    second = OpenRouterChatCompletionsClient.session_id_for_state({"task": "finance.one", "example_id": 3})
    other = OpenRouterChatCompletionsClient.session_id_for_state({"task": "finance.two", "example_id": 4})
    assert first == second
    assert first != other
