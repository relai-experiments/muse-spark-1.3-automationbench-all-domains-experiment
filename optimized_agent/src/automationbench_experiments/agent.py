# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Baseline AutomationBench agent harness.

The benchmark environment owns task state and tool execution. This module owns
agent-side policy: API selection, sampling arguments, client construction,
usage/debug accounting, and conversation-history post-processing.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from verifiers.clients import Client
from verifiers.types import RolloutInput, SamplingArgs, State
from verifiers.types import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from verifiers.types import ClientConfig
from verifiers.utils.message_utils import concat_messages, maybe_normalize_messages
import verifiers as vf

from automationbench_experiments.config import (
    AutomationAgentConfig,
    BASELINE_SYSTEM_PROMPT,
    DEFAULT_BASELINE_MODEL,
)
from automationbench_experiments.client import OpenRouterChatCompletionsClient

if TYPE_CHECKING:
    from automationbench_experiments.runtime import AutomationBenchHarness


@dataclass(frozen=True)
class AgentTurnContext:
    tool_execution_started_at: float

__all__ = [
    "AutomationAgentConfig",
    "BASELINE_SYSTEM_PROMPT",
    "DEFAULT_BASELINE_MODEL",
    "BaselineAutomationAgent",
]


class BaselineAutomationAgent:
    """Default model+tool harness used by AutomationBench.

    This is intentionally separate from task/environment logic so RELAI can
    optimize harness behavior without editing benchmark task definitions.
    """

    def __init__(self, config: AutomationAgentConfig | None = None) -> None:
        self.config = config or AutomationAgentConfig()

    @property
    def default_model(self) -> str:
        """Default evaluated model for this baseline harness."""
        return self.config.default_model

    async def run_rollout(
        self,
        *,
        runtime: AutomationBenchHarness,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Own the model/tool rollout loop for this evaluated agent.

        The runtime provides verifiers integration and benchmark task services.
        Agent policy is applied here so rollout control does not bounce through
        runner hooks that call back into the agent.
        """
        state = await self.initialize_rollout(runtime, input, client, model, sampling_args)
        try:
            try:
                state = await self.prepare_rollout_task(runtime, state)
            except vf.Error as e:
                self.handle_rollout_error(state, e)

            while True:
                if await runtime.is_rollout_complete(state):
                    try:
                        if await self.reconcile_unfinished_completion(runtime, state):
                            continue
                    except vf.Error as e:
                        self.handle_rollout_error(state, e)
                    break
                try:
                    await self.run_model_turn(runtime, state)
                except vf.Error as e:
                    self.handle_rollout_error(state, e)

            await self.finalize_rollout(runtime, state)
            return state
        except asyncio.CancelledError:
            await self.cleanup_cancelled_rollout(runtime, state)
            raise

    async def initialize_rollout(
        self,
        runtime: AutomationBenchHarness,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Create the initial runtime state for one rollout."""
        return await runtime.initialize_rollout_state(input, client, model, sampling_args)

    async def prepare_rollout_task(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> State:
        """Apply benchmark setup first, then agent-owned task preparation."""
        state = await runtime.setup_benchmark_task_state(state)
        turn_limit = getattr(runtime, "max_turns", None)
        if isinstance(turn_limit, int) and turn_limit > 0:
            state["_rollout_turn_limit"] = turn_limit
        return self.prepare_task(state)

    async def run_model_turn(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> None:
        """Run one visible agent turn: tools first if pending, then model."""
        messages = self.build_conversation_from_trajectory(state)

        if self.has_pending_tool_calls(state):
            tool_messages = await self.execute_pending_tool_calls(runtime, state, messages)
            messages = self.append_tool_messages(messages, tool_messages, state)

        prompt_messages = self.normalize_model_input(messages)
        if state.get("final_env_response") is not None:
            return

        response = await runtime.request_model_response(state, prompt_messages)
        await runtime.append_model_response(state, prompt_messages, response)

    async def reconcile_unfinished_completion(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> bool:
        """Request one ordinary follow-up turn for unfinished or risky completion."""
        if not self.should_reconcile_unfinished_completion(state):
            return False

        state["_completion_reconciliation_retries"] = (
            state.get("_completion_reconciliation_retries", 0) + 1
        )
        messages = self.build_conversation_from_trajectory(state)
        messages = concat_messages([
            messages,
            [UserMessage(content=self.completion_reconciliation_prompt(state))],
        ])
        prompt_messages = self.normalize_model_input(messages)
        response = await runtime.request_model_response(state, prompt_messages)
        await runtime.append_model_response(state, prompt_messages, response)
        return True

    def should_reconcile_unfinished_completion(self, state: State) -> bool:
        """Return whether a normal apparent completion warrants one bounded audit."""
        if (
            state.get("error") is not None
            or state.get("prompt_too_long")
            or state.get("is_truncated")
            or state.get("final_env_response") is not None
            or state.get("_completion_reconciliation_retries", 0)
            >= self.config.completion_reconciliation_retry_limit
        ):
            return False

        turn_limit = state.get("_rollout_turn_limit")
        trajectory = state.get("trajectory")
        if (
            not isinstance(turn_limit, int)
            or not isinstance(trajectory, list)
            or len(trajectory) >= turn_limit
        ):
            return False

        text = self.latest_non_tool_assistant_completion(state)
        if text is None:
            return False
        if any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in self.config.blocker_completion_patterns
        ):
            return False
        if self.needs_side_effect_coverage_audit(state):
            return True
        if any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in self.config.completed_completion_patterns
        ):
            return False
        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in self.config.unfinished_completion_patterns
        )

    def needs_side_effect_coverage_audit(self, state: State) -> bool:
        """Return whether completed mutation work warrants one artifact-level audit."""
        retained_families = {
            endpoint.get("family")
            for endpoint in state.get("_discovered_mutation_endpoints", ())
            if isinstance(endpoint, dict) and endpoint.get("family")
        }
        successful_families = set(state.get("_successful_api_mutation_families", ()))
        evidence = [
            item
            for item in state.get("_successful_api_mutation_evidence", ())
            if isinstance(item, dict)
        ]
        if not successful_families:
            return False
        if len(evidence) > 1 or any(self.mutation_evidence_target_count(item) > 1 for item in evidence):
            return True
        if not (retained_families - successful_families):
            return False
        return bool(
            state.get("_preparation_execution_checkpoint_injected")
            or len(retained_families) > 1
        )

    def completion_reconciliation_prompt(self, state: State) -> str:
        """Build the bounded ordinary follow-up from retained and executed mutation evidence."""
        if not self.needs_side_effect_coverage_audit(state):
            return self.config.completion_reconciliation_message
        evidence = list(state.get("_successful_api_mutation_evidence", ()))
        return self.config.completion_side_effect_audit_message.format(
            mutation_operations=self.format_relevant_mutation_operations(state, evidence),
            successful_artifacts=self.format_successful_mutation_evidence(evidence),
        )

    def mutation_evidence_target_count(self, evidence: dict[str, Any]) -> int:
        """Count distinct explicit recipients or attendees in one evidence card."""
        targets: set[str] = set()
        for section_name in ("requested", "confirmed"):
            section = evidence.get(section_name)
            if not isinstance(section, dict):
                continue
            for field in ("recipients", "attendees", "cc", "bcc"):
                values = section.get(field)
                if isinstance(values, list):
                    targets.update(value for value in values if isinstance(value, str))
        return len(targets)

    def format_successful_mutation_evidence(self, evidence: list[Any]) -> str:
        """Format bounded request/result facts for successful artifacts."""
        accepted: list[str] = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            entry = json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            candidate = "; ".join([*accepted, entry])
            if len(candidate) > self.config.completion_evidence_message_limit:
                break
            accepted.append(entry)
        return "; ".join(accepted) or "none"

    def format_relevant_mutation_operations(
        self,
        state: State,
        evidence: list[Any],
    ) -> str:
        """Retain only compact operations relevant to successful artifact families."""
        successful_families = {
            item.get("family")
            for item in evidence
            if isinstance(item, dict) and isinstance(item.get("family"), str)
        }
        relevant_families = set(successful_families)
        if len(evidence) <= 1:
            relevant_families.update(
                endpoint.get("family")
                for endpoint in state.get("_discovered_mutation_endpoints", ())
                if isinstance(endpoint, dict) and endpoint.get("family")
            )
        entries: list[str] = []
        for endpoint in state.get("_discovered_mutation_endpoints", ()):
            if (
                not isinstance(endpoint, dict)
                or endpoint.get("family") not in relevant_families
                or not all(endpoint.get(key) for key in ("id", "method", "url"))
            ):
                continue
            entry = f"{endpoint['id']} [{endpoint['method']} {endpoint['url']}]"
            if entry not in entries:
                entries.append(entry)
            if len(entries) >= self.config.completion_mutation_operation_limit:
                break
        return "; ".join(entries) or "none retained; discover only an exact missing operation"

    def latest_non_tool_assistant_completion(self, state: State) -> str | None:
        """Read only the latest assistant completion when it has no tool calls."""
        trajectory = state.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            return None
        completion = trajectory[-1].get("completion")
        if not isinstance(completion, list) or not completion:
            return None
        message = completion[-1]
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(message, dict):
            role = message.get("role", role)
            content = message.get("content", content)
            tool_calls = message.get("tool_calls", tool_calls)
        if role != "assistant" or tool_calls or not isinstance(content, str):
            return None
        return content.strip() or None

    async def finalize_rollout(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> None:
        """Render the final completion for one rollout."""
        await runtime.finalize_rollout(state)

    async def cleanup_cancelled_rollout(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> None:
        """Run runtime cleanup for a cancelled rollout."""
        await runtime.cleanup_cancelled_rollout(state)

    async def build_turn_messages(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> vf.Messages:
        """Build the next model input, including pending tool results."""
        messages = self.build_conversation_from_trajectory(state)
        if self.has_pending_tool_calls(state):
            tool_messages = await self.execute_pending_tool_calls(runtime, state, messages)
            messages = self.append_tool_messages(messages, tool_messages, state)
        return self.normalize_model_input(messages)

    async def build_followup_turn_messages(
        self,
        runtime: AutomationBenchHarness,
        state: State,
    ) -> vf.Messages:
        """Compatibility wrapper for older callers of follow-up message building."""
        messages = self.build_conversation_from_trajectory(state)
        if self.has_pending_tool_calls(state):
            tool_messages = await self.execute_pending_tool_calls(runtime, state, messages)
            messages = self.append_tool_messages(messages, tool_messages, state)
        return messages

    def build_conversation_from_trajectory(self, state: State) -> vf.Messages:
        """Reconstruct conversation history before the next model call."""
        if len(state["trajectory"]) == 0:
            return state["prompt"]

        prev_turn_prompt = state["trajectory"][-1]["prompt"]
        prev_turn_completion = state["trajectory"][-1]["completion"]
        return concat_messages([prev_turn_prompt, prev_turn_completion])

    def has_pending_tool_calls(self, state: State) -> bool:
        """Return whether the last recorded assistant message requested tools."""
        if len(state["trajectory"]) == 0:
            return False
        completion = state["trajectory"][-1]["completion"]
        if not completion:
            return False
        last_msg = completion[-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls is None and isinstance(last_msg, dict):
            tool_calls = last_msg.get("tool_calls")
        return bool(tool_calls)

    async def execute_pending_tool_calls(
        self,
        runtime: AutomationBenchHarness,
        state: State,
        messages: vf.Messages,
    ) -> vf.Messages:
        """Execute pending tool calls before requesting the next model response."""
        tool_messages = await self.execute_tool_turn(runtime, messages, state)
        tool_messages = maybe_normalize_messages(tool_messages, field_name="env_response")
        return tool_messages

    def append_tool_messages(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: State,
    ) -> vf.Messages:
        """Append tool results to the conversation passed to the model."""
        combined = concat_messages([messages, tool_messages])
        reminders = [
            reminder
            for reminder in (
                self.authoritative_update_evidence_message(messages, tool_messages, state),
                self.candidate_reconciliation_message(messages, tool_messages, state),
                self.spreadsheet_schema_reconciliation_message(messages, tool_messages, state),
                self.discovery_progression_message(messages, tool_messages, state),
                self.preparation_reconciliation_message(messages, state),
                self.post_mutation_evidence_message(state),
            )
            if reminder is not None
        ]
        if not reminders:
            return combined
        return concat_messages([combined, reminders])

    def authoritative_update_evidence_message(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: State,
    ) -> UserMessage | None:
        """Pin bounded exact evidence returned by communication and document reads."""
        if not state.get("_freshness_sensitive_request"):
            return None

        successful_ids = set(state.get("_successful_tool_call_ids", ()))
        reads: dict[str, dict[str, str]] = {}
        for tool_call in self.get_requested_tool_calls(messages):
            if (
                tool_call.id not in successful_ids
                or tool_call.name != self.config.api_fetch_tool_name
            ):
                continue
            try:
                arguments = self.parse_tool_arguments(tool_call)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            method = arguments.get("method")
            url = arguments.get("url")
            if not isinstance(method, str) or method.upper() not in {"GET", "HEAD"}:
                continue
            service = self.authoritative_evidence_service(url)
            if service is None:
                continue
            reads[tool_call.id] = {
                "service": service,
                "source": self.authoritative_evidence_source(arguments),
            }

        cards = list(state.get("_authoritative_update_evidence", ()))
        identities = {
            (card["service"], card["snippet"])
            for card in cards
        }
        added = False
        for message in tool_messages:
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id is None and isinstance(message, dict):
                tool_call_id = message.get("tool_call_id")
            read = reads.get(tool_call_id)
            if read is None or self.tool_message_has_error(message):
                continue
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            snippet = self.authoritative_evidence_snippet(content)
            identity = (read["service"], snippet)
            if snippet is None or identity in identities:
                continue
            if len(cards) >= self.config.authoritative_evidence_card_limit:
                break
            cards.append({
                **read,
                "tool_call_id": tool_call_id,
                "snippet": snippet,
            })
            identities.add(identity)
            added = True

        if not added:
            return None
        state["_authoritative_update_evidence"] = cards
        return self.authoritative_evidence_checkpoint(state)

    def authoritative_evidence_service(self, url: Any) -> str | None:
        """Classify configured communication or document evidence URLs."""
        if not isinstance(url, str):
            return None
        for service, pattern in self.config.authoritative_evidence_sources:
            if re.search(pattern, url):
                return service
        return None

    def authoritative_evidence_source(self, arguments: dict[str, Any]) -> str:
        """Retain the targeted query when available, otherwise the read URL."""
        params = arguments.get("params")
        try:
            decoded = json.loads(params) if isinstance(params, str) else params
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, dict):
            for key in ("query", "q"):
                value = decoded.get(key)
                if isinstance(value, str) and value.strip():
                    return f"{key}={value}"[: self.config.authoritative_evidence_source_limit]
        return str(arguments.get("url") or "unknown source")[
            : self.config.authoritative_evidence_source_limit
        ]

    def authoritative_evidence_snippet(self, content: Any) -> str | None:
        """Extract one exact non-empty result item without summarizing it."""
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            return None
        item = self.find_authoritative_evidence_item(payload)
        if item is None:
            return None
        snippet = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        return snippet[: self.config.authoritative_evidence_snippet_limit]

    def find_authoritative_evidence_item(self, value: Any, depth: int = 0) -> Any | None:
        """Find the first item in a known non-empty search-result collection."""
        if depth > 5:
            return None
        if isinstance(value, dict):
            for key in self.config.authoritative_evidence_collection_keys:
                collection = value.get(key)
                if isinstance(collection, list) and collection:
                    return collection[0]
            for nested in value.values():
                item = self.find_authoritative_evidence_item(nested, depth + 1)
                if item is not None:
                    return item
        return None

    def format_authoritative_evidence(
        self,
        cards: list[dict[str, str]],
        limit: int | None = None,
    ) -> str:
        """Format bounded exact cards with source and raw-result recovery handles."""
        entries = [
            f"{card['service']} [{card['source']}; raw tool call {card['tool_call_id']}]: "
            f"{card['snippet']}"
            for card in cards
        ]
        if limit is None:
            return "; ".join(entries)
        accepted: list[str] = []
        for entry in entries:
            candidate = "; ".join([*accepted, entry])
            if len(candidate) > limit:
                break
            accepted.append(entry)
        return "; ".join(accepted)

    def authoritative_evidence_checkpoint(self, state: State) -> UserMessage | None:
        """Resurface pinned evidence within strict rollout-local message limits."""
        count = int(state.get("_authoritative_evidence_checkpoint_count", 0))
        if count >= self.config.authoritative_evidence_checkpoint_limit:
            return None
        cards = list(state.get("_authoritative_update_evidence", ()))
        if not cards:
            return None
        content = self.config.authoritative_evidence_reconciliation_message.format(
            authoritative_evidence=self.format_authoritative_evidence(cards)
        )
        if len(content) > self.config.authoritative_evidence_message_limit:
            return None
        state["_authoritative_evidence_checkpoint_count"] = count + 1
        return UserMessage(content=content)

    def post_mutation_evidence_message(self, state: State) -> UserMessage | None:
        """Resurface pinned updates after successful mutations for dependent artifacts."""
        mutation_ids = set(state.pop("_successful_api_mutation_turn_ids", ()))
        if not mutation_ids:
            return None
        return self.authoritative_evidence_checkpoint(state)

    def candidate_reconciliation_message(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: State,
    ) -> UserMessage | None:
        """Retain a bounded index of multi-record reads and request complete dispositions."""
        successful_ids = set(state.get("_successful_tool_call_ids", ()))
        read_call_ids: set[str] = set()
        for tool_call in self.get_requested_tool_calls(messages):
            if tool_call.id not in successful_ids or tool_call.name != self.config.api_fetch_tool_name:
                continue
            try:
                method = self.parse_tool_arguments(tool_call).get("method")
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if isinstance(method, str) and method.upper() in {"GET", "HEAD", "OPTIONS"}:
                read_call_ids.add(tool_call.id)

        seen_ids = set(state.get("_candidate_roster_source_ids", ()))
        roster = list(state.get("_candidate_roster", ()))
        newly_observed = 0
        for message in tool_messages:
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id is None and isinstance(message, dict):
                tool_call_id = message.get("tool_call_id")
            if tool_call_id not in read_call_ids or tool_call_id in seen_ids:
                continue
            if self.tool_message_has_error(message):
                continue

            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            entries = self.extract_collection_entries(content)
            if len(entries) < 2:
                continue

            seen_ids.add(tool_call_id)
            newly_observed += 1
            for extracted in entries:
                label = extracted["label"]
                replacement = next(
                    (
                        index
                        for index, entry in enumerate(roster)
                        if entry["label"] == label and entry["tool_call_id"] != tool_call_id
                    ),
                    None,
                )
                entry = {**extracted, "tool_call_id": tool_call_id}
                if replacement is None:
                    roster.append(entry)
                else:
                    roster[replacement] = entry

        if not newly_observed:
            return None

        state["_candidate_roster"] = self.bound_candidate_entries(roster)
        state["_candidate_roster_source_ids"] = seen_ids
        observed_count = int(state.get("_candidate_collection_count", 0)) + newly_observed
        state["_candidate_collection_count"] = observed_count
        frequency = self.config.candidate_checkpoint_frequency
        if frequency < 1 or observed_count % frequency:
            return None

        checkpoint = self.config.candidate_reconciliation_message.format(
            candidate_roster=self.format_candidate_roster(state["_candidate_roster"])
        )
        return UserMessage(content=checkpoint)

    def spreadsheet_schema_reconciliation_message(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: State,
    ) -> UserMessage | None:
        """Retain exact Google Sheets headers and checkpoint positional writes once per schema."""
        successful_ids = set(state.get("_successful_tool_call_ids", ()))
        reads: dict[str, str] = {}
        for tool_call in self.get_requested_tool_calls(messages):
            if (
                tool_call.id not in successful_ids
                or tool_call.name != self.config.api_fetch_tool_name
            ):
                continue
            try:
                arguments = self.parse_tool_arguments(tool_call)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            method = arguments.get("method")
            url = arguments.get("url")
            spreadsheet_id = self.google_sheets_spreadsheet_id(url)
            if (
                isinstance(method, str)
                and method.upper() in {"GET", "HEAD"}
                and spreadsheet_id is not None
            ):
                reads[tool_call.id] = spreadsheet_id

        cached = list(state.get("_spreadsheet_schemas", ()))
        observed: list[dict[str, Any]] = []
        for message in tool_messages:
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id is None and isinstance(message, dict):
                tool_call_id = message.get("tool_call_id")
            spreadsheet_id = reads.get(tool_call_id)
            if spreadsheet_id is None or self.tool_message_has_error(message):
                continue
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            for schema in self.extract_spreadsheet_schemas(
                content, spreadsheet_id, tool_call_id
            ):
                previous = next(
                    (
                        entry
                        for entry in reversed([*cached, *observed])
                        if entry["key"] == schema["key"]
                    ),
                    None,
                )
                if previous is None or previous["headers"] != schema["headers"]:
                    observed = [
                        entry for entry in observed if entry["key"] != schema["key"]
                    ]
                    observed.append(schema)

        if not observed:
            return None

        formatted: list[str] = []
        accepted: list[dict[str, Any]] = []
        for schema in observed:
            entry = self.format_spreadsheet_schema(schema)
            candidate = "; ".join([*formatted, entry])
            checkpoint = self.config.spreadsheet_schema_reconciliation_message.format(
                schemas=candidate
            )
            if len(checkpoint) > self.config.spreadsheet_schema_message_length_limit:
                continue
            formatted.append(entry)
            accepted.append(schema)
        if not accepted:
            return None

        for schema in accepted:
            cached = [entry for entry in cached if entry["key"] != schema["key"]]
            cached.append(schema)
        state["_spreadsheet_schemas"] = cached[-self.config.spreadsheet_schema_cache_limit :]
        checkpoint = self.config.spreadsheet_schema_reconciliation_message.format(
            schemas="; ".join(formatted)
        )
        return UserMessage(content=checkpoint)

    def google_sheets_spreadsheet_id(self, url: Any) -> str | None:
        """Extract a spreadsheet identifier only from a Sheets values URL."""
        if not isinstance(url, str):
            return None
        path = unquote(urlparse(url).path)
        match = re.search(r"/spreadsheets/([^/]+)(?:/values|$)", path)
        return match.group(1) if match else None

    def extract_spreadsheet_schemas(
        self,
        content: Any,
        spreadsheet_id: str,
        tool_call_id: str,
    ) -> list[dict[str, Any]]:
        """Extract bounded, distinguishable row-one headers from Sheets ValueRanges."""
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(payload, dict):
            return []
        value_ranges = payload.get("valueRanges")
        if isinstance(value_ranges, list):
            candidates = value_ranges
        elif "range" in payload and "values" in payload:
            candidates = [payload]
        else:
            return []

        schemas: list[dict[str, Any]] = []
        for value_range in candidates:
            if not isinstance(value_range, dict):
                continue
            range_name = value_range.get("range")
            major_dimension = value_range.get("majorDimension", "ROWS")
            values = value_range.get("values")
            if (
                not isinstance(range_name, str)
                or not self.range_starts_at_header_row(range_name)
                or major_dimension != "ROWS"
                or not isinstance(values, list)
                or not values
            ):
                continue
            headers = self.distinguishable_headers(values[0])
            if headers is None:
                continue
            worksheet = (
                range_name.split("!", 1)[0].strip("'\"")
                if "!" in range_name
                else range_name
            )
            schemas.append({
                "key": f"{spreadsheet_id}:{worksheet}",
                "spreadsheet_id": spreadsheet_id,
                "worksheet": worksheet,
                "range": range_name,
                "headers": headers,
                "tool_call_id": tool_call_id,
            })
        return schemas

    def range_starts_at_header_row(self, range_name: str) -> bool:
        """Return whether an A1 range begins at row one, where headers are distinguishable."""
        cell_range = range_name.split("!", 1)[-1].replace("$", "")
        first_cell = cell_range.split(":", 1)[0]
        return re.fullmatch(r"[A-Za-z]{1,3}1", first_cell) is not None

    def distinguishable_headers(self, value: Any) -> list[str] | None:
        """Validate a bounded, non-empty, unique string header row without inferring labels."""
        if (
            not isinstance(value, list)
            or not value
            or len(value) > self.config.spreadsheet_schema_header_limit
        ):
            return None
        headers: list[str] = []
        for header in value:
            if not isinstance(header, str) or not header.strip():
                return None
            if len(header) > self.config.spreadsheet_schema_header_length_limit:
                return None
            headers.append(header)
        if len({header.casefold() for header in headers}) != len(headers):
            return None
        return headers

    def format_spreadsheet_schema(self, schema: dict[str, Any]) -> str:
        """Format headers and their recovery handle without replaying spreadsheet rows."""
        headers = json.dumps(schema["headers"], ensure_ascii=False, separators=(",", ":"))
        return (
            f"spreadsheet {schema['spreadsheet_id']}, worksheet {schema['worksheet']}, "
            f"range {schema['range']}, headers {headers} "
            f"(raw tool call {schema['tool_call_id']})"
        )

    def extract_collection_labels(self, content: Any) -> list[str]:
        """Extract bounded record identifiers or labels from generic collection-shaped JSON."""
        return [entry["label"] for entry in self.extract_collection_entries(content)]

    def extract_collection_entries(self, content: Any) -> list[dict[str, str]]:
        """Extract bounded identities plus exact policy-sensitive row evidence."""
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            return []
        entries = self.find_collection_entries(payload)
        return self.bound_candidate_entries(entries)

    def bound_candidate_entries(
        self,
        entries: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Keep evidence-bearing identities when the total roster count is bounded."""
        limit = self.config.candidate_roster_limit
        if len(entries) <= limit:
            return entries
        prioritized_indexes = [
            index for index, entry in enumerate(entries) if entry.get("evidence")
        ]
        prioritized_indexes.extend(
            index for index, entry in enumerate(entries) if not entry.get("evidence")
        )
        selected_indexes = sorted(prioritized_indexes[:limit])
        return [entries[index] for index in selected_indexes]

    def find_collection_labels(self, value: Any, depth: int = 0) -> list[str]:
        """Find the first conservative multi-record collection in a decoded result."""
        return [entry["label"] for entry in self.find_collection_entries(value, depth)]

    def find_collection_entries(
        self,
        value: Any,
        depth: int = 0,
    ) -> list[dict[str, str]]:
        """Find the first conservative collection and retain bounded policy evidence."""
        if depth > 4:
            return []
        if isinstance(value, list):
            if len(value) >= 3 and all(isinstance(item, list) for item in value):
                headers = value[0]
                entries = [
                    entry
                    for source_row, row in enumerate(value[1:], start=2)
                    if (
                        entry := self.tabular_row_entry(
                            headers, row, source_row=source_row
                        )
                    ) is not None
                ]
                if len(entries) >= 2:
                    return entries
            if len(value) >= 2 and all(isinstance(item, dict) for item in value):
                entries = [
                    entry
                    for item in value
                    if (entry := self.record_entry(item)) is not None
                ]
                if len(entries) >= 2:
                    return entries
            for item in value:
                entries = self.find_collection_entries(item, depth + 1)
                if entries:
                    return entries
            return []
        if isinstance(value, dict):
            for nested in value.values():
                entries = self.find_collection_entries(nested, depth + 1)
                if entries:
                    return entries
        return []

    def tabular_row_label(self, headers: list[Any], row: list[Any]) -> str | None:
        """Build a compact label from useful tabular identity fields."""
        entry = self.tabular_row_entry(headers, row)
        return entry["label"] if entry is not None else None

    def tabular_row_entry(
        self,
        headers: list[Any],
        row: list[Any],
        source_row: int | None = None,
    ) -> dict[str, str] | None:
        """Retain source ordinal, compact identity fields, and policy-sensitive values."""
        available: list[tuple[str | None, str]] = []
        for index, cell in enumerate(row):
            scalar = self.compact_scalar(cell)
            if scalar is None:
                continue
            header = self.compact_scalar(headers[index]) if index < len(headers) else None
            available.append((header, scalar))
        if not available:
            return None

        identity_fields = [
            (header, scalar)
            for header, scalar in available
            if self.is_candidate_identity_field(header)
        ][:2]
        selected = identity_fields or available[:1]
        parts = [f"row {source_row}"] if source_row is not None else []
        parts.extend(
            f"{header}: {scalar}" if header else scalar
            for header, scalar in selected
        )
        label = self.bound_candidate_label(" | ".join(parts))
        evidence = self.policy_evidence_fragment(
            (
                (headers[index], cell)
                for index, cell in enumerate(row)
                if index < len(headers)
            )
        )
        return {"label": label, **({"evidence": evidence} if evidence else {})}

    def is_candidate_identity_field(self, header: str | None) -> bool:
        """Recognize generic identity columns without making domain-specific matches."""
        if header is None:
            return False
        words = {
            word.lower()
            for word in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", header)
        }
        identity_words = {"id", "uuid", "key", "code", "name", "title", "label", "email"}
        return bool(words.intersection(identity_words))

    def record_label(self, record: dict[str, Any]) -> str | None:
        """Return generic stable identifier and human label fields without domain inference."""
        entry = self.record_entry(record)
        return entry["label"] if entry is not None else None

    def record_entry(self, record: dict[str, Any]) -> dict[str, str] | None:
        """Return stable identity plus exact generic policy-sensitive field values."""
        normalized = {
            "".join(character for character in str(key).lower() if character.isalnum()): value
            for key, value in record.items()
        }
        identifiers = ("id", "uuid", "key", "code")
        labels = ("name", "title", "label", "email")
        parts: list[str] = []
        for keys in (identifiers, labels):
            for key in keys:
                scalar = self.compact_scalar(normalized.get(key))
                if scalar is not None:
                    parts.append(f"{key}: {scalar}")
                    break
        if not parts:
            return None
        evidence = self.policy_evidence_fragment(record.items())
        return {
            "label": self.bound_candidate_label(" | ".join(parts)),
            **({"evidence": evidence} if evidence else {}),
        }

    def policy_evidence_fragment(self, fields: Any) -> str:
        """Render bounded exact scalar values from generic policy-sensitive fields."""
        fragments: list[str] = []
        limit = self.config.candidate_evidence_length_limit
        for key, value in fields:
            if not self.is_policy_evidence_field(key):
                continue
            scalar = self.exact_scalar(value)
            if scalar is None:
                continue
            fragment = (
                f"{json.dumps(str(key), ensure_ascii=False)}:"
                f"{json.dumps(scalar, ensure_ascii=False)}"
            )
            candidate = ",".join([*fragments, fragment])
            if len(candidate) > limit:
                if not fragments:
                    return fragment[:limit]
                break
            fragments.append(fragment)
        return ",".join(fragments)

    def is_policy_evidence_field(self, key: Any) -> bool:
        """Match policy concepts by field words without treating e.g. 'stage' as 'tag'."""
        text = str(key)
        words = [
            word.lower()
            for word in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", text)
        ]
        normalized = "".join(words)
        for concept in self.config.candidate_policy_evidence_field_concepts:
            if concept == "optout" and concept in normalized:
                return True
            if concept in {"eligib", "duplicate", "exclusion", "exclude"}:
                if any(word.startswith(concept) for word in words):
                    return True
            elif concept in words or f"{concept}s" in words:
                return True
        return False

    def exact_scalar(self, value: Any) -> Any | None:
        """Keep scalar evidence values exact rather than normalizing their text."""
        if value is None or isinstance(value, (dict, list)):
            return None
        if isinstance(value, str) and not value:
            return None
        return value

    def compact_scalar(self, value: Any) -> str | None:
        """Normalize a scalar for the compact roster without copying nested payloads."""
        if value is None or isinstance(value, (dict, list)):
            return None
        text = " ".join(str(value).split())
        return text or None

    def bound_candidate_label(self, label: str) -> str:
        """Bound one roster label's context footprint."""
        return label[: self.config.candidate_label_length_limit]

    def format_candidate_roster(self, roster: list[dict[str, str]]) -> str:
        """Format compact identities and policy evidence with recovery handles."""
        grouped: dict[str, list[str]] = {}
        for entry in roster:
            evidence = entry.get("evidence")
            rendered = (
                f"{entry['label']} [policy evidence: {evidence}]"
                if evidence
                else entry["label"]
            )
            grouped.setdefault(entry["tool_call_id"], []).append(rendered)
        return "; ".join(
            f"{', '.join(labels)} (raw tool call {tool_call_id})"
            for tool_call_id, labels in grouped.items()
        )

    def multi_record_mutation_preflight(
        self,
        messages: vf.Messages,
        state: State,
    ) -> dict[str, vf.ToolMessage]:
        """Withhold one freshness-sensitive mutation bundle after a multi-record read."""
        roster = list(state.get("_candidate_roster", ()))
        if (
            not state.get("_freshness_sensitive_request")
            or len(roster) < 2
            or state.get("_multi_record_mutation_preflight_consumed")
        ):
            return {}

        mutation_call_ids: list[str] = []
        for tool_call in self.get_requested_tool_calls(messages):
            if tool_call.name != self.config.api_fetch_tool_name:
                continue
            try:
                method = self.parse_tool_arguments(tool_call).get("method")
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if isinstance(method, str) and method.upper() not in {"GET", "HEAD", "OPTIONS"}:
                mutation_call_ids.append(tool_call.id)

        if not mutation_call_ids:
            return {}

        state["_multi_record_mutation_preflight_consumed"] = True
        feedback = self.config.multi_record_mutation_preflight_message.format(
            proposed_count=len(mutation_call_ids),
            candidate_roster=self.format_candidate_roster(roster),
            available_services=", ".join(state.get("_allowed_service_inventory", ()))
            or "none reported",
            authoritative_evidence=self.format_authoritative_evidence(
                list(state.get("_authoritative_update_evidence", ())),
                self.config.authoritative_evidence_preflight_limit,
            )
            or "none captured",
        )
        results: dict[str, vf.ToolMessage] = {}
        for index, tool_call_id in enumerate(mutation_call_ids):
            content = {
                "ok": False,
                "withheld": True,
                "side_effect_occurred": False,
                "preflight": feedback if index == 0 else "Withheld by the same one-shot preflight.",
            }
            results[tool_call_id] = vf.ToolMessage(
                role="tool",
                content=json.dumps(content, ensure_ascii=False),
                tool_call_id=tool_call_id,
            )
        return results

    def discovery_progression_message(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: State,
    ) -> UserMessage | None:
        """Persist discovery evidence and checkpoint discovery-to-action progression."""
        tool_calls = self.get_requested_tool_calls(messages)
        search_name = self.config.api_search_tool_name
        successful_ids = set(state.get("_successful_tool_call_ids", ()))
        successful_search_ids = {
            tool_call.id
            for tool_call in tool_calls
            if tool_call.name == search_name and tool_call.id in successful_ids
        }
        had_mutations = bool(state.get("_discovered_mutation_endpoints"))
        if successful_search_ids:
            self.remember_discovered_endpoints(tool_messages, successful_search_ids, state)

        mutation_endpoints = list(state.get("_discovered_mutation_endpoints", ()))
        urgent = self.budget_execution_checkpoint(state, mutation_endpoints)
        if urgent is not None:
            return urgent

        if not tool_calls or any(tool_call.name != search_name for tool_call in tool_calls):
            state["_api_search_only_streak"] = 0
            return None

        if not successful_search_ids:
            return None

        streak = int(state.get("_api_search_only_streak", 0)) + 1
        state["_api_search_only_streak"] = streak
        mutation_transition = bool(mutation_endpoints) and not had_mutations
        if mutation_endpoints and (
            mutation_transition or (
                streak >= self.config.discovery_checkpoint_threshold
                and not state.get("_api_mutation_checkpoint_injected")
            )
        ):
            state["_api_mutation_checkpoint_injected"] = True
            progression = self.config.discovery_mutation_progression_message.format(
                mutation_endpoints=self.format_endpoint_checkpoint(mutation_endpoints),
                fetch_tool_name=self.config.api_fetch_tool_name,
            )
            return UserMessage(content=f"{progression} {self.config.internal_tool_turn_rule}")

        read_endpoints = list(state.get("_discovered_read_endpoints", ()))
        if (
            streak < self.config.discovery_checkpoint_threshold
            or state.get("_api_read_checkpoint_injected")
            or not read_endpoints
        ):
            return None

        state["_api_read_checkpoint_injected"] = True
        progression = self.config.discovery_read_progression_message.format(
            read_endpoints=self.format_endpoint_checkpoint(read_endpoints),
            fetch_tool_name=self.config.api_fetch_tool_name,
        )
        return UserMessage(content=f"{progression} {self.config.internal_tool_turn_rule}")

    def remember_discovered_endpoints(
        self,
        tool_messages: vf.Messages,
        successful_search_ids: set[str],
        state: State,
    ) -> None:
        """Retain bounded read and mutation endpoint evidence from api_search."""
        read_endpoints = list(state.get("_discovered_read_endpoints", ()))
        mutation_endpoints = list(state.get("_discovered_mutation_endpoints", ()))
        for message in tool_messages:
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id is None and isinstance(message, dict):
                tool_call_id = message.get("tool_call_id")
            if tool_call_id not in successful_search_ids:
                continue

            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            try:
                payload = json.loads(content) if isinstance(content, str) else content
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            results = payload.get("results")
            if not isinstance(results, list):
                continue
            for result in results:
                if not isinstance(result, dict):
                    continue
                endpoint_id = result.get("id")
                method = result.get("method")
                url = result.get("url")
                if not all(isinstance(value, str) and value for value in (endpoint_id, method, url)):
                    continue
                endpoint = {
                    "id": endpoint_id,
                    "method": method.upper(),
                    "url": url,
                    "schema": self.endpoint_schema_identity(result),
                }
                endpoint.update(self.endpoint_family_metadata(endpoint_id, url))
                target = (
                    read_endpoints
                    if endpoint["method"] in {"GET", "HEAD", "OPTIONS"}
                    else mutation_endpoints
                )
                target[:] = [known for known in target if known["id"] != endpoint_id]
                target.append(endpoint)

        limit = self.config.discovery_endpoint_limit
        state["_discovered_read_endpoints"] = read_endpoints[-limit:]
        state["_discovered_mutation_endpoints"] = self.select_mutation_endpoints(
            mutation_endpoints,
            self.config.discovery_mutation_endpoint_limit,
        )

    def endpoint_family_metadata(self, endpoint_id: str, url: str) -> dict[str, str]:
        """Derive generic service/resource identity from an endpoint ID, with a URL fallback."""
        parts = [part for part in endpoint_id.split(".") if part]
        if len(parts) >= 3:
            service = parts[0]
            resource = ".".join(parts[1:-1])
        else:
            parsed = urlparse(url)
            host_parts = [part for part in parsed.hostname.split(".") if part] if parsed.hostname else []
            path_parts = [part for part in parsed.path.split("/") if part]
            service = host_parts[0] if host_parts else "unknown"
            if service in {"www", "api"} and path_parts:
                service = path_parts[0]
            resource = next(
                (
                    part
                    for part in reversed(path_parts)
                    if not part.startswith("{")
                    and not re.fullmatch(r"v\d+(?:\.\d+)?", part, re.IGNORECASE)
                ),
                "resource",
            )
        return {
            "service": service,
            "resource": resource,
            "family": f"{service}.{resource}",
        }

    def mutation_endpoint_priority(self, endpoint: dict[str, str]) -> int:
        """Prefer ordinary business mutations over destructive administrative variants."""
        action = endpoint.get("id", "").rsplit(".", 1)[-1].lower()
        if action in {"create", "insert", "send", "append"}:
            return 4
        if action in {"update", "patch"}:
            return 3
        if action in {"delete", "trash", "archive", "untrash"}:
            return 0
        return 2

    def select_mutation_endpoints(
        self,
        endpoints: list[dict[str, str]],
        limit: int,
    ) -> list[dict[str, str]]:
        """Select a bounded, family-diverse mutation registry with useful actions first."""
        if limit <= 0:
            return []
        deduplicated: list[dict[str, str]] = []
        for endpoint in endpoints:
            deduplicated = [
                known for known in deduplicated if known.get("id") != endpoint.get("id")
            ]
            deduplicated.append(endpoint)

        by_family: dict[str, list[tuple[int, dict[str, str]]]] = {}
        for index, endpoint in enumerate(deduplicated):
            family = endpoint.get("family") or endpoint.get("id", "")
            by_family.setdefault(family, []).append((index, endpoint))
        for family_endpoints in by_family.values():
            family_endpoints.sort(
                key=lambda item: (self.mutation_endpoint_priority(item[1]), item[0]),
                reverse=True,
            )

        family_order = sorted(
            by_family,
            key=lambda family: min(index for index, _ in by_family[family]),
        )
        ordered: list[dict[str, str]] = []
        for priority in (4, 3, 2, 0):
            for family in family_order:
                ordered.extend(
                    endpoint
                    for _, endpoint in by_family[family]
                    if self.mutation_endpoint_priority(endpoint) == priority
                )
        return ordered[:limit]

    def endpoint_schema_identity(self, result: dict[str, Any]) -> str:
        """Return a compact request-schema identity suitable for later reuse."""
        request = result.get("request")
        if isinstance(request, str):
            identity = " ".join(request.split())
        elif request is not None:
            identity = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        else:
            parameters = result.get("parameters")
            body_fields = []
            if isinstance(parameters, dict):
                body_fields = [
                    name
                    for name, spec in parameters.items()
                    if isinstance(spec, dict) and spec.get("location") == "body"
                ]
            identity = f"body fields: {', '.join(body_fields)}" if body_fields else "no body schema"
        return identity[: self.config.discovery_schema_identity_limit]

    def format_endpoint_checkpoint(self, endpoints: list[dict[str, str]]) -> str:
        """Format retained endpoint evidence without replaying full search results."""
        return "; ".join(
            f"{endpoint['id']} [{endpoint['method']} {endpoint['url']}; "
            f"schema: {endpoint['schema']}]"
            for endpoint in endpoints
        )

    def budget_execution_checkpoint(
        self,
        state: State,
        mutation_endpoints: list[dict[str, str]],
    ) -> UserMessage | None:
        """Issue one urgent execution checkpoint before the rollout budget closes."""
        if (
            not mutation_endpoints
            or state.get("_successful_api_mutation_ids")
            or state.get("_budget_execution_checkpoint_injected")
        ):
            return None
        turn_limit = state.get("_rollout_turn_limit")
        trajectory = state.get("trajectory")
        if not isinstance(turn_limit, int) or not isinstance(trajectory, list):
            return None
        remaining_turns = turn_limit - len(trajectory)
        if not 0 < remaining_turns <= self.config.budget_checkpoint_remaining_turns:
            return None
        state["_budget_execution_checkpoint_injected"] = True
        checkpoint = self.config.budget_execution_message.format(
            remaining_turns=remaining_turns,
            mutation_endpoints=self.format_endpoint_checkpoint(mutation_endpoints),
        )
        return UserMessage(content=f"{checkpoint} {self.config.internal_tool_turn_rule}")

    def record_successful_api_mutation(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_message: vf.ToolMessage,
        tool_call_id: str,
        state: State,
    ) -> None:
        """Ground mutation progress only in successful non-read api_fetch results."""
        method = tool_args.get("method")
        if (
            tool_name != self.config.api_fetch_tool_name
            or not isinstance(method, str)
            or method.upper() in {"GET", "HEAD", "OPTIONS"}
            or self.tool_message_has_error(tool_message)
        ):
            return
        state.setdefault("_successful_api_mutation_ids", set()).add(tool_call_id)
        state.setdefault("_successful_api_mutation_turn_ids", set()).add(tool_call_id)
        url = tool_args.get("url")
        if not isinstance(url, str) or not url:
            return
        family = self.executed_mutation_family(method.upper(), url, state)
        state.setdefault("_successful_api_mutation_families", set()).add(family)
        evidence = state.setdefault("_successful_api_mutation_evidence", [])
        item: dict[str, Any] = {
            "family": family,
            "method": method.upper(),
            "destination": url[: self.config.mutation_evidence_value_length_limit],
        }
        artifact_id = self.successful_artifact_id(tool_message)
        if artifact_id is not None:
            item["artifact_id"] = artifact_id
        requested = self.api_mutation_request_facts(tool_args)
        if requested:
            item["requested"] = requested
        confirmed = self.api_mutation_result_facts(tool_message)
        if confirmed:
            item["confirmed"] = confirmed
        if item not in evidence and len(evidence) < self.config.mutation_evidence_card_limit:
            evidence.append(item)

    def api_mutation_request_facts(self, tool_args: dict[str, Any]) -> dict[str, Any]:
        """Extract bounded contract facts explicitly present in api_fetch arguments."""
        facts: dict[str, Any] = {}
        for key in ("params", "body"):
            value = tool_args.get(key)
            try:
                value = json.loads(value) if isinstance(value, str) else value
            except (json.JSONDecodeError, TypeError):
                continue
            self.collect_artifact_contract_facts(value, facts)
        return facts

    def api_mutation_result_facts(self, tool_message: vf.ToolMessage) -> dict[str, Any]:
        """Extract bounded contract facts explicitly confirmed by a successful result."""
        content = getattr(tool_message, "content", None)
        if content is None and isinstance(tool_message, dict):
            content = tool_message.get("content")
        try:
            value = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            return {}
        facts: dict[str, Any] = {}
        self.collect_artifact_contract_facts(value, facts)
        return facts

    def collect_artifact_contract_facts(
        self,
        value: Any,
        facts: dict[str, Any],
        depth: int = 0,
    ) -> None:
        """Collect only explicit, small target fields without inventing obligations."""
        if depth > 5:
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
                if normalized == "raw" and isinstance(nested, str):
                    self.collect_encoded_message_facts(nested, facts)
                    continue
                elif normalized in {"attendee", "attendees"}:
                    self.add_evidence_values(facts, "attendees", nested)
                    continue
                elif normalized in {"to", "recipient", "recipients"}:
                    self.add_evidence_values(facts, "recipients", nested)
                    continue
                elif normalized in {"cc", "bcc"}:
                    self.add_evidence_values(facts, normalized, nested)
                    continue
                elif normalized == "headers" and isinstance(nested, list):
                    self.collect_message_header_facts(nested, facts)
                    continue
                else:
                    field = dict(self.config.mutation_evidence_target_fields).get(normalized)
                    if field is not None and isinstance(nested, (str, int, float, bool)):
                        facts.setdefault(
                            field,
                            str(nested)[: self.config.mutation_evidence_value_length_limit],
                        )
                        continue
                self.collect_artifact_contract_facts(nested, facts, depth + 1)
        elif isinstance(value, list):
            for nested in value[: self.config.mutation_evidence_list_limit]:
                self.collect_artifact_contract_facts(nested, facts, depth + 1)

    def collect_message_header_facts(
        self,
        headers: list[Any],
        facts: dict[str, Any],
    ) -> None:
        """Collect explicit recipient and subject facts from structured mail headers."""
        for header in headers[: self.config.mutation_evidence_list_limit]:
            if not isinstance(header, dict):
                continue
            name = str(header.get("name") or "").lower()
            value = header.get("value")
            if name == "to":
                self.add_evidence_values(facts, "recipients", value)
            elif name in {"cc", "bcc"}:
                self.add_evidence_values(facts, name, value)
            elif name == "subject" and isinstance(value, str):
                facts.setdefault(
                    "subject", value[: self.config.mutation_evidence_value_length_limit]
                )

    def collect_encoded_message_facts(self, raw: str, facts: dict[str, Any]) -> None:
        """Decode bounded RFC-style headers from an explicit base64url message body."""
        if len(raw) > self.config.mutation_evidence_encoded_message_limit:
            return
        try:
            padding = "=" * (-len(raw) % 4)
            decoded = base64.urlsafe_b64decode(raw + padding).decode("utf-8", errors="replace")
        except (binascii.Error, ValueError, UnicodeError):
            return
        for line in decoded.splitlines():
            if not line.strip():
                break
            name, separator, value = line.partition(":")
            if not separator:
                continue
            lowered = name.strip().lower()
            if lowered == "to":
                self.add_evidence_values(facts, "recipients", value)
            elif lowered in {"cc", "bcc"}:
                self.add_evidence_values(facts, lowered, value)
            elif lowered == "subject":
                facts.setdefault(
                    "subject",
                    value.strip()[: self.config.mutation_evidence_value_length_limit],
                )

    def add_evidence_values(
        self,
        facts: dict[str, Any],
        field: str,
        value: Any,
    ) -> None:
        """Add bounded explicit identity values from strings, lists, or email objects."""
        values: list[str] = []
        if isinstance(value, str):
            emails = re.findall(r"[\w.!#$%&'*+/=?^`{|}~-]+@[\w.-]+\.[A-Za-z]{2,}", value)
            values.extend(emails or [part.strip() for part in value.split(",") if part.strip()])
        elif isinstance(value, dict):
            for key in ("email", "address", "id", "name"):
                nested = value.get(key)
                if isinstance(nested, str) and nested.strip():
                    values.append(nested.strip())
                    break
        elif isinstance(value, list):
            for nested in value[: self.config.mutation_evidence_list_limit]:
                nested_facts: dict[str, Any] = {}
                self.add_evidence_values(nested_facts, field, nested)
                values.extend(nested_facts.get(field, ()))

        existing = facts.setdefault(field, [])
        for item in values:
            bounded = item[: self.config.mutation_evidence_value_length_limit]
            if bounded not in existing and len(existing) < self.config.mutation_evidence_list_limit:
                existing.append(bounded)
        if not existing:
            facts.pop(field, None)

    def successful_artifact_id(self, tool_message: vf.ToolMessage) -> str | None:
        """Find one bounded successful artifact identifier in the result payload."""
        content = getattr(tool_message, "content", None)
        if content is None and isinstance(tool_message, dict):
            content = tool_message.get("content")
        try:
            value = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            return None
        found = self.find_artifact_id(value)
        if found is None:
            return None
        return found[: self.config.mutation_evidence_value_length_limit]

    def find_artifact_id(self, value: Any, depth: int = 0) -> str | None:
        """Find the first explicit ID in a bounded result object traversal."""
        if depth > 5:
            return None
        if isinstance(value, dict):
            identifier = value.get("id")
            if isinstance(identifier, (str, int)) and str(identifier):
                return str(identifier)
            for nested in value.values():
                found = self.find_artifact_id(nested, depth + 1)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for nested in value[: self.config.mutation_evidence_list_limit]:
                found = self.find_artifact_id(nested, depth + 1)
                if found is not None:
                    return found
        return None

    def executed_mutation_family(self, method: str, url: str, state: State) -> str:
        """Match an executed concrete URL back to retained generic endpoint metadata."""
        for endpoint in state.get("_discovered_mutation_endpoints", ()):
            if (
                isinstance(endpoint, dict)
                and endpoint.get("method") == method
                and self.endpoint_url_matches(endpoint.get("url", ""), url)
            ):
                family = endpoint.get("family")
                if isinstance(family, str) and family:
                    return family
        return self.endpoint_family_metadata("", url)["family"]

    def endpoint_url_matches(self, template: str, concrete: str) -> bool:
        """Match endpoint path templates without task-specific URL parsing."""
        expected = urlparse(template)
        actual = urlparse(concrete)
        if expected.hostname != actual.hostname:
            return False
        expected_parts = [unquote(part) for part in expected.path.split("/") if part]
        actual_parts = [unquote(part) for part in actual.path.split("/") if part]
        if len(expected_parts) != len(actual_parts):
            return False
        return all(
            (expected_part.startswith("{") and expected_part.endswith("}"))
            or expected_part == actual_part
            for expected_part, actual_part in zip(expected_parts, actual_parts)
        )

    def tool_message_has_error(self, tool_message: vf.ToolMessage) -> bool:
        """Return whether a tool result explicitly reports an API error."""
        content = getattr(tool_message, "content", None)
        if content is None and isinstance(tool_message, dict):
            content = tool_message.get("content")
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            return False
        return isinstance(payload, dict) and "error" in payload

    def preparation_reconciliation_message(
        self,
        messages: vf.Messages,
        state: State,
    ) -> UserMessage | None:
        """Turn successful preparation with known mutations into an execution checkpoint."""
        successful_ids = set(state.pop("_successful_tool_call_ids", ()))
        if not successful_ids:
            return None

        prepared_count = sum(
            tool_call.id in successful_ids
            and tool_call.name in self.config.preparation_tool_names
            for tool_call in self.get_requested_tool_calls(messages)
        )
        if prepared_count < 1:
            return None

        mutation_endpoints = list(state.get("_discovered_mutation_endpoints", ()))
        if not mutation_endpoints or state.get("_preparation_execution_checkpoint_injected"):
            return None

        state["_preparation_execution_checkpoint_injected"] = True
        state["_pending_preparation_output_count"] = prepared_count
        roster = list(state.get("_candidate_roster", ()))
        candidate_roster = (
            self.format_candidate_roster(roster)
            if roster
            else "none captured; reconcile against the authoritative raw reads"
        )
        checkpoint = self.config.preparation_reconciliation_message.format(
            count=prepared_count,
            candidate_roster=candidate_roster,
            mutation_endpoints=self.format_endpoint_checkpoint(mutation_endpoints),
            fetch_tool_name=self.config.api_fetch_tool_name,
        )
        return UserMessage(
            content=f"{checkpoint} {self.config.internal_tool_turn_rule}"
        )

    def normalize_model_input(self, messages: vf.Messages) -> vf.Messages:
        """Normalize the final model input for provider/client compatibility."""
        return maybe_normalize_messages(messages, field_name="prompt_messages")

    def handle_rollout_error(self, state: State, error: vf.Error) -> None:
        """Preserve verifiers rollout error semantics."""
        if isinstance(error, vf.OverlongPromptError):
            state["prompt_too_long"] = True
            state["is_truncated"] = True
        else:
            state["error"] = error

    def record_rollout_error(self, state: State, error: vf.Error) -> None:
        """Compatibility wrapper for older callers of rollout error handling."""
        self.handle_rollout_error(state, error)

    @staticmethod
    def resolve_api(model: str, base_url: str | None, api_override: str = "auto") -> str:
        """Return which API client to use."""
        if api_override != "auto":
            return api_override
        if model.startswith("gemini-") and base_url is None:
            return "gemini"
        if model.startswith("claude-") and (base_url is None or "anthropic.com" in base_url):
            return "anthropic"
        if model.startswith(("deepseek/", "meta/", "xiaomi/")) and base_url is None:
            return "openrouter"
        return "chat_completions"

    def build_sampling_args(
        self,
        model: str,
        base_url: str | None,
        resolved_api: str,
        reasoning_effort: str | None,
        extra_body: str | None,
    ) -> dict[str, Any] | None:
        """Build provider sampling args for one evaluation run."""
        sampling_args: dict[str, Any] | None = None
        use_anthropic_api = resolved_api == "anthropic"

        if reasoning_effort:
            if use_anthropic_api:
                _adaptive_models = (
                    "opus-4-6",
                    "opus-4-7",
                    "opus-4-8",
                    "opus-4.8",
                    "sonnet-4-6",
                    "sonnet-5",
                    "fable-5",
                )
                if any(m in model for m in _adaptive_models):
                    sampling_args = {
                        "thinking": {"type": "adaptive"},
                        "output_config": {"effort": reasoning_effort},
                        "max_tokens": 64000,
                    }
                else:
                    _budget = {
                        "low": 2000,
                        "medium": 8000,
                        "high": 16000,
                        "xhigh": 24000,
                        "max": 32000,
                    }
                    budget_tokens = _budget.get(reasoning_effort, 8000)
                    sampling_args = {
                        "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
                        "max_tokens": 64000,
                    }
            else:
                _gemini_flash_needs_native = (
                    base_url is not None
                    and "litellm" in base_url
                    and (
                        "gemini-3.1-flash" in model
                        or "gemini-3-flash" in model
                        or "gemini-3.5-flash" in model
                    )
                )
                if _gemini_flash_needs_native:
                    sampling_args = {
                        "extra_body": {"thinkingConfig": {"thinkingLevel": reasoning_effort}}
                    }
                else:
                    sampling_args = {"reasoning_effort": reasoning_effort}

        if extra_body:
            parsed = json.loads(extra_body)
            sampling_args = sampling_args or {}
            sampling_args["extra_body"] = {**sampling_args.get("extra_body", {}), **parsed}

        return sampling_args

    def build_client(
        self,
        resolved_api: str,
        base_url: str | None,
        api_key: str | None,
        api_key_var: str,
        extra_headers: dict[str, str] | None,
    ):
        """Build the verifiers client used by this evaluated agent."""
        if resolved_api != "openrouter":
            raise ValueError("Muse Spark experiments require the OpenRouter API")
        effective_key_var = "OPENROUTER_API_KEY" if api_key_var == "OPENAI_API_KEY" else api_key_var

        if api_key is not None:
            os.environ[effective_key_var] = api_key
        elif not os.environ.get(effective_key_var):
            raise ValueError(
                f"No API key found. Set {effective_key_var} environment variable, "
                "or pass --api-key argument."
            )

        config = ClientConfig(
            api_key_var=effective_key_var,
            api_base_url=base_url or "https://openrouter.ai/api/v1",
            extra_headers=extra_headers or {},
        )
        return OpenRouterChatCompletionsClient(config)

    def authoritative_time_context(self, state: State) -> str | None:
        """Format an explicitly supplied simulated clock for model context."""
        world = state.get("world")
        meta = getattr(world, "meta", None)
        if meta is None and isinstance(world, dict):
            meta = world.get("meta")
        if meta is None:
            return None

        if isinstance(meta, dict):
            current_time = meta.get("current_time")
        else:
            fields_set = getattr(meta, "model_fields_set", None)
            if fields_set is not None and "current_time" not in fields_set:
                return None
            current_time = getattr(meta, "current_time", None)

        if (
            not isinstance(current_time, datetime)
            or current_time.tzinfo is None
            or current_time.utcoffset() is None
        ):
            return None

        timestamp = current_time.isoformat(timespec="seconds")
        return (
            f"Runtime context: the simulated world's authoritative current time is {timestamp}. "
            "Use this instead of any host or model-provider time for all relative-date, due, "
            "overdue, recency, and scheduling calculations."
        )

    def allowed_service_inventory(self, state: State) -> list[str]:
        """Return the bounded runtime-provided service inventory."""
        world = state.get("world")
        meta = getattr(world, "meta", None)
        if meta is None and isinstance(world, dict):
            meta = world.get("meta")
        if meta is None:
            return []
        allowed_services = (
            meta.get("allowed_services")
            if isinstance(meta, dict)
            else getattr(meta, "allowed_services", None)
        )
        if not isinstance(allowed_services, list):
            return []
        services = sorted({
            service.strip()
            for service in allowed_services
            if isinstance(service, str) and service.strip()
        })
        return services[: self.config.allowed_service_inventory_limit]

    def allowed_service_context(self, services: list[str]) -> str | None:
        """Expose available evidence channels without task assertions or world contents."""
        if not services:
            return None
        return (
            "Runtime context: available source systems for this task are "
            f"{', '.join(services)}. For freshness-sensitive work, use targeted entity or "
            "record-identifier searches in plausible update-bearing systems rather than broad "
            "service enumeration."
        )

    def task_user_text(self, prompt: vf.Messages | None) -> str:
        """Collect only user-authored task text for request classification."""
        if not isinstance(prompt, list):
            return ""
        parts: list[str] = []
        for message in prompt:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if isinstance(message, dict):
                role = message.get("role", role)
                content = message.get("content", content)
            if role == "user" and isinstance(content, str):
                parts.append(content)
        return "\n".join(parts)

    def is_freshness_sensitive_request(self, prompt: vf.Messages | None) -> bool:
        """Classify explicit user requests for current or recently updated evidence."""
        user_text = self.task_user_text(prompt)
        return any(
            re.search(pattern, user_text, flags=re.IGNORECASE) is not None
            for pattern in self.config.freshness_sensitive_request_patterns
        )

    def prepare_initial_prompt(
        self,
        prompt: vf.Messages | None,
        runtime_context: str | None = None,
    ) -> vf.Messages | None:
        """Make shared agent instructions owned by the harness.

        Existing task data still carries a legacy system message for standalone
        compatibility. During evaluation, replace that first system message with
        the agent-owned prompt so optimizer experiments can modify one harness
        surface without rewriting every task.
        """
        if prompt is None:
            return None

        messages = list(prompt)
        system_content = self.config.system_prompt
        if runtime_context:
            system_content = f"{system_content}\n\n{runtime_context}"
        system_message = SystemMessage(content=system_content)
        if not messages:
            return [system_message]

        first = messages[0]
        role = getattr(first, "role", None)
        if role is None and isinstance(first, dict):
            role = first.get("role")

        if role == "system" and self.config.normalize_legacy_system_prompt:
            return [system_message, *messages[1:]]

        return [system_message, *messages]

    def prepare_task(self, state: State) -> State:
        """Prepare one task state before the first model call."""
        prompt = state.get("prompt")
        for key in (
            "_authoritative_update_evidence",
            "_authoritative_evidence_checkpoint_count",
            "_successful_api_mutation_ids",
            "_successful_api_mutation_turn_ids",
            "_successful_api_mutation_families",
            "_successful_api_mutation_evidence",
        ):
            state.pop(key, None)
        services = self.allowed_service_inventory(state)
        state["_allowed_service_inventory"] = services
        state["_freshness_sensitive_request"] = self.is_freshness_sensitive_request(prompt)
        runtime_context = "\n".join(
            context
            for context in (
                self.authoritative_time_context(state),
                self.allowed_service_context(services),
            )
            if context is not None
        )
        state["prompt"] = self.prepare_initial_prompt(prompt, runtime_context or None)
        return state

    def normalize_tool_args(self, tool_args: dict) -> dict:
        """Normalize model-emitted tool arguments before environment injection."""
        return {
            k: v for k, v in tool_args.items() if not (isinstance(v, dict) and len(v) == 0)
        }

    def prepare_tool_call(self, tool_name: str, tool_args: dict, state: State) -> dict:
        """Prepare one model-emitted tool call before environment context injection."""
        return self.normalize_tool_args(tool_args)

    def get_requested_tool_calls(self, messages: vf.Messages) -> list[vf.ToolCall]:
        """Return tool calls requested by the latest assistant message."""
        assert isinstance(messages, list)
        last_msg = messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        assert tool_calls is not None
        return list(tool_calls)

    def parse_tool_arguments(self, tool_call: vf.ToolCall) -> dict:
        """Parse model-emitted tool-call arguments."""
        parsed_args = json.loads(tool_call.arguments)
        if not isinstance(parsed_args, dict):
            raise ValueError(
                "Expected tool arguments to be a dict, "
                f"got {type(parsed_args).__name__}: {parsed_args}"
            )
        return parsed_args

    def parse_tool_call(self, tool_call: vf.ToolCall) -> tuple[str, dict]:
        """Parse one model-emitted tool call into a name and argument dict."""
        return tool_call.name, self.parse_tool_arguments(tool_call)

    def format_tool_parse_error(
        self,
        error: Exception,
        tool_call_id: str,
        runtime: AutomationBenchHarness,
    ) -> vf.ToolMessage:
        """Format a tool-argument parse failure for the next model turn."""
        return vf.ToolMessage(
            role="tool",
            content=runtime.format_runtime_error(error),
            tool_call_id=tool_call_id,
        )

    def format_tool_call_error(
        self,
        error: Exception,
        tool_call_id: str,
        runtime: AutomationBenchHarness,
    ) -> vf.ToolMessage:
        """Format a tool-execution failure for the next model turn."""
        return vf.ToolMessage(
            role="tool",
            content=runtime.format_runtime_error(error),
            tool_call_id=tool_call_id,
        )

    def postprocess_tool_message(
        self,
        tool_message: vf.ToolMessage,
        state: State,
    ) -> vf.ToolMessage:
        """Post-process one raw runtime tool result."""
        return tool_message

    def postprocess_tool_messages(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: State,
    ) -> vf.Messages:
        """Post-process all raw runtime tool results."""
        return tool_messages

    async def execute_tool_turn(
        self,
        runtime: AutomationBenchHarness,
        messages: vf.Messages,
        state: State,
    ) -> vf.Messages:
        """Execute requested tool calls with agent-owned tool-turn policy."""
        context = self.before_tool_execution(messages, state)
        tool_messages: vf.Messages = []
        successful_tool_call_ids: set[str] = set()
        state["_successful_tool_call_ids"] = successful_tool_call_ids
        state["_successful_api_mutation_turn_ids"] = set()
        withheld_mutations = self.multi_record_mutation_preflight(messages, state)

        for tool_call in self.get_requested_tool_calls(messages):
            tool_call_id = tool_call.id
            if tool_call_id in withheld_mutations:
                tool_messages.append(withheld_mutations[tool_call_id])
                continue
            try:
                tool_name, tool_args = self.parse_tool_call(tool_call)
            except Exception as e:
                if runtime.should_stop_for_tool_error(e):
                    raise vf.ToolParseError from e
                tool_messages.append(self.format_tool_parse_error(e, tool_call_id, runtime))
                continue

            try:
                prepared_args = self.prepare_tool_call(tool_name, tool_args, state)
                injected_args = runtime.inject_task_tool_args(tool_name, prepared_args, state)
                tool_message = await runtime.call_task_tool(
                    tool_name,
                    injected_args,
                    tool_call_id,
                )
                processed_message = self.postprocess_tool_message(tool_message, state)
                tool_messages.append(processed_message)
                successful_tool_call_ids.add(tool_call_id)
                self.record_successful_api_mutation(
                    tool_name,
                    tool_args,
                    processed_message,
                    tool_call_id,
                    state,
                )
            except Exception as e:
                if runtime.should_stop_for_tool_error(e):
                    raise vf.ToolCallError from e
                tool_messages.append(self.format_tool_call_error(e, tool_call_id, runtime))

        tool_messages = self.postprocess_tool_messages(messages, tool_messages, state)
        return self.after_tool_execution(
            messages,
            tool_messages,
            state,
            context,
            use_meta_tools=runtime.use_meta_tools,
        )

    def extract_usage_and_debug(self, state: vf.State) -> None:
        """Extract token usage and debug info from the latest model response."""
        trajectory = state.get("trajectory", [])
        if not trajectory:
            return

        step = trajectory[-1]
        response = step.get("response")
        if response is None:
            return

        usage = getattr(response, "usage", None)
        if usage is not None:
            if "_usage" not in state:
                state["_usage"] = {"input_tokens": 0, "output_tokens": 0}
            state["_usage"]["input_tokens"] += getattr(usage, "prompt_tokens", 0)
            state["_usage"]["output_tokens"] += getattr(usage, "completion_tokens", 0)

        msg = getattr(response, "message", None)
        finish_reason = getattr(msg, "finish_reason", None) if msg else None
        content = getattr(msg, "content", None) if msg else None
        tool_calls = getattr(msg, "tool_calls", None) if msg else None
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        if "_debug" not in state:
            state["_debug"] = {"finish_reasons": [], "empty_responses": [], "errors": []}

        state["_debug"]["finish_reasons"].append(finish_reason)

        if not content and not tool_calls:
            import sys

            task_name = state.get("task", "unknown")
            empty_info = {
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
            }
            state["_debug"]["empty_responses"].append(empty_info)
            print(
                f"[DEBUG] Empty response for {task_name}: "
                f"finish_reason={finish_reason}, completion_tokens={completion_tokens}",
                file=sys.stderr,
            )

    def start_tool_timing(self, messages: vf.Messages, state: vf.State) -> float:
        """Count this turn's tool calls and return the timer start."""
        perf = state.setdefault(
            "_perf",
            {
                "model_time_s": 0.0,
                "model_calls": 0,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
                "tool_time_s": 0.0,
                "tool_calls": 0,
            },
        )
        last_msg = messages[-1] if messages else None
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls is None and isinstance(last_msg, dict):
            tool_calls = last_msg.get("tool_calls")
        perf["tool_calls"] += len(tool_calls) if tool_calls else 0
        return time.monotonic()

    def finish_tool_timing(self, state: vf.State, started_at: float) -> None:
        """Accumulate elapsed tool-execution time."""
        state["_perf"]["tool_time_s"] += time.monotonic() - started_at

    def before_tool_execution(self, messages: vf.Messages, state: State) -> AgentTurnContext:
        """Apply agent policy before task-environment tools execute."""
        self.extract_usage_and_debug(state)
        return AgentTurnContext(tool_execution_started_at=self.start_tool_timing(messages, state))

    def compress_meta_messages(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: vf.State,
    ) -> vf.Messages:
        """Compress old search_tools results after execute_tool is called."""
        last_msg = messages[-1]
        if not isinstance(last_msg, AssistantMessage):
            return tool_messages
        tool_calls = last_msg.tool_calls or []
        if not tool_calls:
            return tool_messages

        current_search_ids: set[str] = set()
        has_execute = False
        for tc in tool_calls:
            if tc.name == "search_tools":
                current_search_ids.add(tc.id)
            elif tc.name == "execute_tool":
                has_execute = True

        state.setdefault("_search_call_ids", set()).update(current_search_ids)

        if not has_execute:
            return tool_messages

        compressible_ids = state["_search_call_ids"] - current_search_ids
        if not compressible_ids:
            return tool_messages

        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            if msg.tool_call_id not in compressible_ids:
                continue
            content = msg.content
            if not isinstance(content, str) or len(content) < 200:
                continue
            try:
                results = json.loads(content)
                if isinstance(results, list):
                    names = [r.get("name", "") for r in results if isinstance(r, dict)]
                    msg.content = f"[Previously found: {', '.join(names)}]"
            except (json.JSONDecodeError, TypeError):
                pass

        state["_search_call_ids"] -= compressible_ids
        return tool_messages

    def after_tool_response(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: vf.State,
        *,
        use_meta_tools: bool,
    ) -> vf.Messages:
        """Apply agent-side message policy after tools execute."""
        if use_meta_tools and self.config.enable_meta_message_compression:
            return self.compress_meta_messages(messages, tool_messages, state)
        return tool_messages

    def after_tool_execution(
        self,
        messages: vf.Messages,
        tool_messages: vf.Messages,
        state: State,
        context: AgentTurnContext,
        *,
        use_meta_tools: bool,
    ) -> vf.Messages:
        """Apply agent policy after task-environment tools execute."""
        self.finish_tool_timing(state, context.tool_execution_started_at)
        return self.after_tool_response(
            messages,
            tool_messages,
            state,
            use_meta_tools=use_meta_tools,
        )
