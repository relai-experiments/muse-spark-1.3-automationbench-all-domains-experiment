"""OpenRouter client policy carried over from the Muse baseline."""

from __future__ import annotations

import hashlib
from typing import Any

from automationbench.clients import RetryingOpenAIChatCompletionsClient


class OpenRouterChatCompletionsClient(RetryingOpenAIChatCompletionsClient):
    """Keep all requests in one rollout on the same OpenRouter session."""

    @staticmethod
    def _state_value(state: Any, *keys: str) -> Any:
        value = state
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    @classmethod
    def session_id_for_state(cls, state: Any) -> str | None:
        if state is None:
            return None
        candidates = [
            cls._state_value(state, "info", "env_id"),
            cls._state_value(state, "info", "task"),
            cls._state_value(state, "example_id"),
            cls._state_value(state, "task", "example_id"),
            cls._state_value(state, "task"),
        ]
        seed_parts = [str(item) for item in candidates if item not in (None, "")]
        seed = "|".join(seed_parts) if seed_parts else f"state-{id(state)}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        return f"automationbench-{digest}"

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        session_id = self.session_id_for_state(kwargs.get("state"))
        if session_id is not None:
            sampling_args = dict(sampling_args or {})
            extra_body = dict(sampling_args.get("extra_body") or {})
            extra_body.setdefault("session_id", session_id)
            sampling_args["extra_body"] = extra_body
        return await super().get_native_response(prompt, model, sampling_args, tools, **kwargs)
