"""Bridge the old optimizable agent to AutomationBench 1.0.6 task semantics."""

from __future__ import annotations

from typing import Any

import verifiers as vf
from automationbench.runner import AutomationBenchEnv
from verifiers.clients import Client
from verifiers.types import RolloutInput, SamplingArgs, State

from automationbench_experiments.agent import BaselineAutomationAgent


class AutomationBenchHarness(AutomationBenchEnv):
    def __init__(self, *args: Any, agent: BaselineAutomationAgent, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.agent = agent

    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        return await self.agent.run_rollout(
            runtime=self, input=input, client=client, model=model, sampling_args=sampling_args
        )

    async def initialize_rollout_state(
        self, input: RolloutInput, client: Client, model: str, sampling_args: SamplingArgs | None
    ) -> State:
        return await self.init_state(input, client, model, sampling_args)

    async def setup_benchmark_task_state(self, state: State) -> State:
        return await super().setup_state(state)

    async def request_model_response(self, state: State, prompt_messages: vf.Messages) -> vf.Response:
        return await self.get_model_response(state, prompt_messages)

    async def append_model_response(
        self, state: State, prompt_messages: vf.Messages, response: vf.Response
    ) -> None:
        await self.add_model_response(state, prompt_messages, response)

    def inject_task_tool_args(self, tool_name: str, tool_args: dict, state: State) -> dict:
        return super().update_tool_args(tool_name, tool_args, [], state)

    async def call_task_tool(self, tool_name: str, tool_args: dict, tool_call_id: str) -> vf.ToolMessage:
        return await self.call_tool(tool_name, tool_args, tool_call_id)

    def should_stop_for_tool_error(self, error: Exception) -> bool:
        return self._should_stop_for_error(error)

    def format_runtime_error(self, error: Exception) -> str:
        return self.error_formatter(error)

    async def is_rollout_complete(self, state: State) -> bool:
        return await self.is_completed(state)

    async def finalize_rollout(self, state: State) -> None:
        await self.render_completion(state)

    async def cleanup_cancelled_rollout(self, state: State) -> None:
        await self._cleanup(state)

    def _extract_usage_and_debug(self, state: State) -> None:
        self.agent.extract_usage_and_debug(state)

    async def setup_state(self, state: State, **kwargs: Any) -> State:
        state = await self.setup_benchmark_task_state(state)
        return self.agent.prepare_task(state)

    def update_tool_args(
        self, tool_name: str, tool_args: dict, messages: vf.Messages, state: State, **kwargs: Any
    ) -> dict:
        prepared = self.agent.prepare_tool_call(tool_name, tool_args, state)
        return super().update_tool_args(tool_name, prepared, messages, state, **kwargs)

    async def env_response(self, messages: vf.Messages, state: State, **kwargs: Any) -> vf.Messages:
        return await self.agent.execute_tool_turn(self, messages, state)
