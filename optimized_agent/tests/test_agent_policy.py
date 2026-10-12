from __future__ import annotations

import asyncio
import base64

from automationbench.schema.world import WorldState
from verifiers.types import AssistantMessage, ToolCall, ToolMessage, UserMessage

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


def test_baseline_prompt_separates_source_provenance_from_action_values() -> None:
    assert "source values; transformed operational values" in BASELINE_SYSTEM_PROMPT
    assert "successful tool result matches every artifact-contract field" in BASELINE_SYSTEM_PROMPT
    assert "exact value adjacent to the counted entity" in BASELINE_SYSTEM_PROMPT
    assert "do not rely on a label followed by a detached value" in BASELINE_SYSTEM_PROMPT
    assert "clearly labeled source provenance" in BASELINE_SYSTEM_PROMPT
    assert "destination mutations, derived totals, and outbound summaries must use the same" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "Retain verbatim source values only where explicitly requested as provenance" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "totals were recomputed from those transformed values" in BASELINE_SYSTEM_PROMPT


def test_baseline_prompt_preserves_requested_operation_semantics() -> None:
    assert "Before searching for mutation endpoints or mutating" in BASELINE_SYSTEM_PROMPT
    assert "destination service and collection; artifact type; lifecycle state" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "recipient or target; required content; operation; completion state" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "Drafts are saved-but-unsent outbound artifacts" in BASELINE_SYSTEM_PROMPT
    assert "posted notes, comments, replies, messages, and sent items cannot satisfy" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "Append-only notes, comments, messages, and activity entries" in BASELINE_SYSTEM_PROMPT
    assert "explicit field or status changes use mutation endpoints" in BASELINE_SYSTEM_PROMPT
    assert "requested artifact noun and destination service when known" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "endpoint description and request schema must match every contract field" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "must change the requested artifact type and lifecycle state" in BASELINE_SYSTEM_PROMPT
    assert "final claims must name the action actually performed" in BASELINE_SYSTEM_PROMPT


def test_baseline_prompt_requires_canonical_identifier_provenance() -> None:
    assert "explicitly named by the task, returned by a successful read" in BASELINE_SYSTEM_PROMPT
    assert "Schema examples are illustrative, never identifier evidence" in BASELINE_SYSTEM_PROMPT
    assert "unless it is independently observed or explicitly requested" in BASELINE_SYSTEM_PROMPT
    assert "authoritative metadata maps a task-named field to a canonical identifier" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "otherwise preserve the task's literal field name" in BASELINE_SYSTEM_PROMPT
    assert "at most one specifically targeted metadata fetch" in BASELINE_SYSTEM_PROMPT
    assert "otherwise leave the obligation unresolved" in BASELINE_SYSTEM_PROMPT
    assert "do not repeat broad searches" in BASELINE_SYSTEM_PROMPT


def test_baseline_prompt_reconciles_destination_objects_before_create() -> None:
    assert "destination candidate-search evidence and exactly one disposition" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "create, append_or_update_existing, skip, or unresolved" in BASELINE_SYSTEM_PROMPT
    assert "same-type, same-lifecycle objects in the requested destination collection" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "objects already fetched, planned, or created earlier in the rollout" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "Source identity reconciliation is separate from destination-object duplicate detection" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "source records and adjacent artifact types are context" in BASELINE_SYSTEM_PROMPT
    assert "not duplicate destination artifacts" in BASELINE_SYSTEM_PROMPT
    assert "normalized identifiers or subjects, semantic issue equivalence" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "explicit duplicate, repeat, or same-issue wording" in BASELINE_SYSTEM_PROMPT
    assert "contact, organization, or relationship evidence" in BASELINE_SYSTEM_PROMPT
    assert "exact text equality is not required" in BASELINE_SYSTEM_PROMPT
    assert "A unique same-contract destination match forbids duplicate creation" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "append or update only if the requested contract permits it" in BASELINE_SYSTEM_PROMPT
    assert "withhold the create rather than guess" in BASELINE_SYSTEM_PROMPT
    assert "Do not repeat a search after a unique same-contract match" in BASELINE_SYSTEM_PROMPT
    assert "A unique match forbids creation" not in BASELINE_SYSTEM_PROMPT


def test_baseline_prompt_reconciles_explicit_source_identity_links() -> None:
    assert "duplicate registration, same person, alternate identity" in BASELINE_SYSTEM_PROMPT
    assert "reference to another source record establishes a source identity relationship" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "one canonical-versus-alternate disposition" in BASELINE_SYSTEM_PROMPT
    assert "at most one base mutation for that identity" in BASELINE_SYSTEM_PROMPT
    assert "Similar names, organizations, or vague resemblance" in BASELINE_SYSTEM_PROMPT
    assert "do not establish that relationship" in BASELINE_SYSTEM_PROMPT
    assert "notification totals from successful canonical mutations" in BASELINE_SYSTEM_PROMPT


def test_baseline_prompt_keeps_cross_artifact_obligations_separate() -> None:
    assert "Keep separate contracts for separate operations" in BASELINE_SYSTEM_PROMPT
    assert "completing or forbidding one must not suppress or satisfy another" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "operation-effect matrix" in BASELINE_SYSTEM_PROMPT
    assert "base destination mutations, recipient delivery, approval or routing" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "email, escalation, or coordinator request cannot satisfy" in BASELINE_SYSTEM_PROMPT
    assert "Exclusion from a primary destination does not cancel return, rejection" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "reconcile every contract against successful side effects" in BASELINE_SYSTEM_PROMPT
    assert "continue targeted discovery or report the obligation unresolved" in (
        BASELINE_SYSTEM_PROMPT
    )


def test_baseline_prompt_resolves_authoritative_sources_before_mutation() -> None:
    assert "inventory plausible source systems" in BASELINE_SYSTEM_PROMPT
    assert "targeted searches in each relevant system before broad enumeration" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "absence from one channel is not proof that no newer source exists" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "provenance and sender role, explicit directive, operational scope and applicability" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "with recency only among sources that apply to the same operation" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "compact internal selected-policy, artifact-contract, and per-record coverage ledger" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "chosen source identity, date, and unresolved conflicts" in BASELINE_SYSTEM_PROMPT
    assert "Before the first mutation, confirm discovery is complete enough" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "every mutation and report value matches the ledger" in BASELINE_SYSTEM_PROMPT


def test_baseline_prompt_tracks_relationship_scoped_policy_conditions() -> None:
    assert "Evaluate policy effects per operation" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "affects only that operation unless it expressly forbids the underlying base artifact" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "hierarchy or related entities" in BASELINE_SYSTEM_PROMPT
    assert "solely because it belongs to a parent or child" in BASELINE_SYSTEM_PROMPT
    assert "explicitly resolve its applicability" in BASELINE_SYSTEM_PROMPT
    assert "every discovered qualifying fact and applicable operation has an explicit disposition" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "every required action appears in the coverage ledger" in BASELINE_SYSTEM_PROMPT


def test_baseline_prompt_reconciles_per_action_dispositions_and_prerequisites() -> None:
    assert "create one reconciliation row per source record" in BASELINE_SYSTEM_PROMPT
    assert "every applicable operation marked required, forbidden, blocked, or not applicable" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "expected artifact counts and completion evidence" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "lookup, unique entity match, preference, permission, destination" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "prerequisite only for the actions it actually gates" in BASELINE_SYSTEM_PROMPT
    assert "Withhold those dependent mutations" in BASELINE_SYSTEM_PROMPT
    assert "continuing any independently required operations" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "proposed artifact count matches the expected count for each operation" in (
        BASELINE_SYSTEM_PROMPT
    )


def test_baseline_prompt_classifies_record_context_and_gates_rollbacks() -> None:
    assert "Direct task requests and explicitly applicable procedures may govern actions" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "Descriptive free text must not invent inferred classifications" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "unambiguous record-specific lifecycle statement" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "withdrawn, cancelled, opted out, on hold, or completed" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "operations it explicitly permits or forbids" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "structured status, tag, flag, and schema-defined eligibility annotations" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "keeping lifecycle facts distinct from governing instructions" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "Text format alone does not determine authority" in BASELINE_SYSTEM_PROMPT
    assert "explicit operational instructions in email, Slack, documents, or similar sources" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "provenance and sender role, explicit directive" in BASELINE_SYSTEM_PROMPT
    assert "operational scope and applicability, effective time, and supersession evidence" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "not-applicable or superseded only for a concrete conflicting rule" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "Do not undo a successful, verified requested mutation solely because record content" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "scope and authority outweigh the request" in BASELINE_SYSTEM_PROMPT


def test_baseline_prompt_reconciles_collection_report_coverage_and_totals() -> None:
    assert "every returned in-scope record ID" in BASELINE_SYSTEM_PROMPT
    assert "included, excluded, or unresolved disposition" in BASELINE_SYSTEM_PROMPT
    assert "authoritative eligibility evidence and status or category bucket" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "included-record count equals the sum of status or category buckets" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "each overall total equals the sum of its per-group breakdown" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "unresolved records or unexplained omissions block publication" in (
        BASELINE_SYSTEM_PROMPT
    )


def test_baseline_prompt_stops_resolved_discovery_without_ignoring_real_conflicts() -> None:
    assert "requested action, applicable governing procedure, and uniquely matching target" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "do not broaden into exhaustive searches" in BASELINE_SYSTEM_PROMPT
    assert "descriptive record content appears cautionary or conflicting" in BASELINE_SYSTEM_PROMPT
    assert "Keep genuine unresolved authoritative conflicts open" in BASELINE_SYSTEM_PROMPT
    assert "using those authority and applicability factors" in (
        BASELINE_SYSTEM_PROMPT
    )


def test_prepare_initial_prompt_replaces_legacy_system_message() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    user_message = {"role": "user", "content": "Complete the task."}

    prepared = agent.prepare_initial_prompt([
        {"role": "system", "content": "Legacy instructions."},
        user_message,
    ])

    assert prepared is not None
    assert len(prepared) == 2
    assert prepared[0].role == "system"
    assert prepared[0].content == BASELINE_SYSTEM_PROMPT
    assert "inventory plausible source systems" in prepared[0].content
    assert "selected-policy, artifact-contract, and per-record coverage ledger" in (
        prepared[0].content
    )
    assert "source values; transformed operational values" in prepared[0].content
    assert "required, forbidden, blocked, or not applicable" in prepared[0].content
    assert "expected artifact counts and completion evidence" in (
        prepared[0].content
    )
    assert "every discovered qualifying fact and applicable operation has an explicit disposition" in (
        prepared[0].content
    )
    assert "every required action appears in the coverage ledger" in prepared[0].content
    assert "prerequisite only for the actions it actually gates" in prepared[0].content
    assert "destination candidate-search evidence and exactly one disposition" in (
        prepared[0].content
    )
    assert "planned, or created earlier in the rollout" in prepared[0].content
    assert "A unique same-contract destination match forbids duplicate creation" in (
        prepared[0].content
    )
    assert "source records and adjacent artifact types are context" in prepared[0].content
    assert "Drafts are saved-but-unsent outbound artifacts" in prepared[0].content
    assert "withhold the create rather than guess" in prepared[0].content
    assert "Before the first mutation" in prepared[0].content
    assert "Descriptive free text must not invent inferred classifications" in (
        prepared[0].content
    )
    assert "every returned in-scope record ID" in prepared[0].content
    assert "authoritative source field" in prepared[0].content
    assert "unexplained omissions block publication" in prepared[0].content
    assert "Do not undo a successful, verified requested mutation" in prepared[0].content
    assert "do not broaden into exhaustive searches" in prepared[0].content
    assert prepared[1] == user_message


def test_prepare_task_adds_authoritative_world_time_to_system_message() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    state = {
        "world": WorldState(meta={"current_time": "2026-03-15T09:30:00Z"}),
        "prompt": [
            {"role": "system", "content": "Legacy instructions."},
            {"role": "user", "content": "Complete the task."},
        ],
    }

    prepared = agent.prepare_task(state)

    system_content = prepared["prompt"][0].content
    assert system_content.startswith(BASELINE_SYSTEM_PROMPT)
    assert "artifact type; lifecycle state" in system_content
    assert "change the requested artifact type" in system_content
    assert "policy effects per operation" in system_content
    assert "every required action appears in the coverage ledger" in system_content
    assert "proposed artifact count matches the expected count for each operation" in (
        system_content
    )
    assert "authoritative current time is 2026-03-15T09:30:00+00:00" in system_content
    assert "instead of any host or model-provider time" in system_content
    assert prepared["prompt"][1] == {"role": "user", "content": "Complete the task."}


def test_prepare_task_exposes_allowed_services_and_classifies_freshness() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    state = {
        "world": WorldState(meta={
            "allowed_services": ["google_sheets", "gmail", "google_drive"],
        }),
        "info": {"assertions": [{"secret_evaluator_value": "do-not-expose"}]},
        "prompt": [
            {"role": "system", "content": "Legacy instructions."},
            {
                "role": "user",
                "content": "Use the current policy and double-check recent updates before acting.",
            },
        ],
    }

    prepared = agent.prepare_task(state)

    system_content = prepared["prompt"][0].content
    assert (
        "available source systems for this task are gmail, google_drive, google_sheets"
        in system_content
    )
    assert "targeted entity or record-identifier searches" in system_content
    assert "do-not-expose" not in system_content
    assert prepared["_allowed_service_inventory"] == [
        "gmail",
        "google_drive",
        "google_sheets",
    ]
    assert prepared["_freshness_sensitive_request"] is True


def test_prepare_task_keeps_ordinary_request_out_of_freshness_preflight() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    state = {
        "world": WorldState(meta={"allowed_services": ["gmail"]}),
        "prompt": [{"role": "user", "content": "Send the approved welcome email."}],
        "_authoritative_update_evidence": [{"stale": "card"}],
        "_authoritative_evidence_checkpoint_count": 3,
        "_successful_api_mutation_ids": {"stale-mutation"},
        "_successful_api_mutation_turn_ids": {"stale-mutation"},
        "_successful_api_mutation_families": {"mail.messages"},
        "_successful_api_mutation_evidence": [{
            "family": "mail.messages",
            "method": "POST",
            "url": "https://mail.example.test/messages",
        }],
    }

    prepared = agent.prepare_task(state)

    assert "available source systems for this task are gmail" in prepared["prompt"][0].content
    assert prepared["_freshness_sensitive_request"] is False
    assert "_authoritative_update_evidence" not in prepared
    assert "_authoritative_evidence_checkpoint_count" not in prepared
    assert "_successful_api_mutation_ids" not in prepared
    assert "_successful_api_mutation_turn_ids" not in prepared
    assert "_successful_api_mutation_families" not in prepared
    assert "_successful_api_mutation_evidence" not in prepared


def test_prepare_task_does_not_invent_missing_world_time() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    state = {
        "world": WorldState(),
        "prompt": [{"role": "user", "content": "Complete the task."}],
    }

    prepared = agent.prepare_task(state)

    assert prepared["prompt"][0].content == BASELINE_SYSTEM_PROMPT


def test_authoritative_time_context_survives_trajectory_reconstruction() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    state = {
        "world": WorldState(meta={"current_time": "2026-03-15T09:30:00Z"}),
        "prompt": [{"role": "user", "content": "Complete the task."}],
        "trajectory": [],
    }
    agent.prepare_task(state)
    state["trajectory"] = [{
        "prompt": state["prompt"],
        "completion": [{"role": "assistant", "content": "Checking records."}],
    }]

    reconstructed = agent.build_conversation_from_trajectory(state)

    assert "authoritative current time is 2026-03-15T09:30:00+00:00" in (
        reconstructed[0].content
    )


def test_baseline_prompt_preserves_conflicting_duplicate_observations() -> None:
    assert "group duplicate identifiers without dropping any source observation" in (
        BASELINE_SYSTEM_PROMPT
    )
    assert "compare every conflicting observation" in BASELINE_SYSTEM_PROMPT
    assert "explicitly defines canonical or latest-state semantics" in BASELINE_SYSTEM_PROMPT


def test_multi_record_policy_read_injects_typed_rule_effect_checkpoint() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig(candidate_label_length_limit=60))
    call = ToolCall(
        id="policy-1",
        name="api_fetch",
        arguments='{"method":"GET","url":"https://example.test/sheet"}',
    )
    messages = [AssistantMessage(content=None, tool_calls=[call])]
    result = ToolMessage(
        content=(
            '{"valueRanges":[{"values":[["Rule","Detail"],'
            '["Submission Fields","Requests must include supporting details"],'
            '["Blocked Route","Suspended records cannot use the standard workflow"]]}]}'
        ),
        tool_call_id=call.id,
    )
    state = {"_successful_tool_call_ids": {call.id}}

    combined = agent.append_tool_messages(messages, [result], state)

    checkpoint = next(message for message in combined if isinstance(message, UserMessage))
    assert combined.index(result) < combined.index(checkpoint)
    assert "predicate; rule type" in checkpoint.content
    assert "authoritative evidence; explicit consequence" in checkpoint.content
    assert "base destination mutation, recipient delivery, approval or routing" in checkpoint.content
    assert "does not cancel the base artifact unless it expressly forbids it" in checkpoint.content
    assert "email or escalation cannot satisfy a requested calendar" in checkpoint.content
    assert "required, forbidden, blocked, or not applicable" in checkpoint.content
    assert "expected artifact counts" in checkpoint.content
    assert "row 2 | Rule: Submission Fields" in checkpoint.content
    assert "row 3 | Rule: Blocked Route" in checkpoint.content
    assert "raw tool call policy-1" in checkpoint.content
    assert all(
        term not in checkpoint.content.lower()
        for term in ("receipt", "category", "amount", "expense", "meal", "travel")
    )
    assert state["_candidate_roster"] == [
        {"label": "row 2 | Rule: Submission Fields", "tool_call_id": "policy-1"},
        {"label": "row 3 | Rule: Blocked Route", "tool_call_id": "policy-1"},
    ]


def test_update_evidence_survives_empty_searches_and_deduplicates() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    call = ToolCall(
        id="slack-1",
        name="api_fetch",
        arguments=(
            '{"method":"GET","params":"{\\"query\\":\\"client update\\"}",'
            '"url":"https://slack.com/api/search.messages"}'
        ),
    )
    messages = [AssistantMessage(content=None, tool_calls=[call])]
    result = ToolMessage(
        content=(
            '{"ok":true,"messages":{"total":1,"matches":['
            '{"ts":"1","text":"Apply the signed amendment this cycle."}]}}'
        ),
        tool_call_id=call.id,
    )
    state = {
        "_freshness_sensitive_request": True,
        "_successful_tool_call_ids": {call.id},
    }

    first = agent.append_tool_messages(messages, [result], state)
    duplicate = agent.append_tool_messages(messages, [result], state)
    empty_call = ToolCall(
        id="slack-2",
        name="api_fetch",
        arguments=(
            '{"method":"GET","params":"{\\"query\\":\\"other update\\"}",'
            '"url":"https://slack.com/api/search.messages"}'
        ),
    )
    state["_successful_tool_call_ids"] = {empty_call.id}
    empty = agent.append_tool_messages(
        [AssistantMessage(content=None, tool_calls=[empty_call])],
        [ToolMessage(
            content='{"ok":true,"messages":{"total":0,"matches":[]}}',
            tool_call_id=empty_call.id,
        )],
        state,
    )

    checkpoint = next(
        message
        for message in first
        if isinstance(message, UserMessage)
        and "Authoritative update reconciliation checkpoint" in message.content
    )
    assert '"text":"Apply the signed amendment this cycle."' in checkpoint.content
    assert "raw tool call slack-1" in checkpoint.content
    assert "Text format alone does not determine authority" in checkpoint.content
    assert "provenance and sender role, directive, scope and applicability" in checkpoint.content
    assert "not-applicable or superseded requires a concrete conflicting rule" in (
        checkpoint.content
    )
    assert len(state["_authoritative_update_evidence"]) == 1
    assert not any(
        isinstance(message, UserMessage)
        and "Authoritative update reconciliation checkpoint" in message.content
        for message in duplicate
    )
    assert not any(
        isinstance(message, UserMessage)
        and "Authoritative update reconciliation checkpoint" in message.content
        for message in empty
    )


def test_update_evidence_is_bounded_and_disabled_for_ordinary_tasks() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig(
        authoritative_evidence_card_limit=1,
        authoritative_evidence_snippet_limit=36,
    ))
    calls = [
        ToolCall(
            id=f"slack-{index}",
            name="api_fetch",
            arguments=(
                f'{{"method":"GET","params":"{{\\"query\\":\\"update {index}\\"}}",'
                '"url":"https://slack.com/api/search.messages"}'
            ),
        )
        for index in (1, 2)
    ]
    results = [
        ToolMessage(
            content=(
                '{"ok":true,"messages":{"matches":['
                f'{{"text":"Authoritative update number {index}"}}]}}'
            ),
            tool_call_id=call.id,
        )
        for index, call in zip((1, 2), calls)
    ]
    state = {
        "_freshness_sensitive_request": True,
        "_successful_tool_call_ids": {call.id for call in calls},
    }

    agent.append_tool_messages(
        [AssistantMessage(content=None, tool_calls=calls)],
        results,
        state,
    )

    assert len(state["_authoritative_update_evidence"]) == 1
    assert len(state["_authoritative_update_evidence"][0]["snippet"]) <= 36

    ordinary_state = {
        "_freshness_sensitive_request": False,
        "_successful_tool_call_ids": {calls[0].id},
    }
    agent.append_tool_messages(
        [AssistantMessage(content=None, tool_calls=[calls[0]])],
        [results[0]],
        ordinary_state,
    )
    assert "_authoritative_update_evidence" not in ordinary_state


def test_json_collection_retains_only_identity_and_policy_sensitive_evidence() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())

    labels = agent.extract_collection_labels(
        '{"records":[{"id":"r-1","name":"Alpha","eligible":true,"amount":10},'
        '{"id":"r-2","name":"Beta","eligible":false,"amount":20}]}'
    )
    entries = agent.extract_collection_entries(
        '{"records":[{"id":"r-1","name":"Alpha","eligible":true,"amount":10},'
        '{"id":"r-2","name":"Beta","eligible":false,"amount":20}]}'
    )

    assert labels == ["id: r-1 | name: Alpha", "id: r-2 | name: Beta"]
    assert entries == [
        {
            "label": "id: r-1 | name: Alpha",
            "evidence": '"eligible":true',
        },
        {
            "label": "id: r-2 | name: Beta",
            "evidence": '"eligible":false',
        },
    ]
    assert all("amount" not in str(entry) for entry in entries)


def test_tabular_duplicate_and_exclusion_notes_survive_candidate_checkpoint() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    call = ToolCall(
        id="sheet-1",
        name="api_fetch",
        arguments='{"method":"GET","url":"https://example.test/sheet"}',
    )
    messages = [AssistantMessage(content=None, tool_calls=[call])]
    result = ToolMessage(
        content=(
            '{"values":[["name","email","notes","company"],'
            '["Alpha","alpha@example.test","Duplicate of row 4","Example Co"],'
            '["Beta","beta@example.test","Exclude by current policy","Other Co"],'
            '["Gamma","gamma@example.test","Explicitly opted out","Third Co"],'
            '["Delta","delta@example.test","Direct competitor exclusion","Fourth Co"]]}'
        ),
        tool_call_id=call.id,
    )
    state = {"_successful_tool_call_ids": {call.id}}

    combined = agent.append_tool_messages(messages, [result], state)

    checkpoint = next(message for message in combined if isinstance(message, UserMessage))
    assert '"notes":"Duplicate of row 4"' in checkpoint.content
    assert '"notes":"Exclude by current policy"' in checkpoint.content
    assert '"notes":"Explicitly opted out"' in checkpoint.content
    assert '"notes":"Direct competitor exclusion"' in checkpoint.content
    assert "row 2 | name: Alpha | email: alpha@example.test" in checkpoint.content
    assert "row 3 | name: Beta | email: beta@example.test" in checkpoint.content
    assert "Example Co" not in checkpoint.content
    assert "Other Co" not in checkpoint.content
    assert "raw tool call sheet-1" in checkpoint.content
    assert "explicit source identity assertions" in (
        checkpoint.content
    )
    assert "Vague similarity cannot invent a duplicate or spam classification" in checkpoint.content
    assert "duplicate registration, same person, alternate identity" in checkpoint.content
    assert "one canonical-versus-alternate disposition" in checkpoint.content
    assert "separate from destination-object duplicate detection" in checkpoint.content
    assert "withdrawal, cancellation, opt-out, hold, or completion" in checkpoint.content
    assert "operative evidence for the operations it expressly permits or forbids" in (
        checkpoint.content
    )


def test_tabular_roster_keeps_similar_people_distinct_without_explicit_identity_evidence() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())

    entries = agent.extract_collection_entries(
        '{"values":[["name","email","notes"],'
        '["Alex Kim","alex.one@example.test","Met at event"],'
        '["Alex Kim","alex.two@example.test","Similar role at another company"]]}'
    )

    assert entries == [
        {"label": "row 2 | name: Alex Kim | email: alex.one@example.test"},
        {"label": "row 3 | name: Alex Kim | email: alex.two@example.test"},
    ]


def test_single_record_and_malformed_reads_do_not_create_roster() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())

    assert agent.extract_collection_labels('{"records":[{"id":"only"}]}') == []
    assert agent.extract_collection_labels('{"records":[') == []
    assert agent.extract_collection_labels('{"records":[null,{"value":{"nested":true}}]}') == []


def test_oversized_collection_and_labels_are_bounded() -> None:
    agent = BaselineAutomationAgent(
        AutomationAgentConfig(
            candidate_roster_limit=2,
            candidate_label_length_limit=12,
            candidate_evidence_length_limit=18,
        )
    )

    entries = agent.extract_collection_entries(
        '[{"id":"identifier-one-long","amount":"ordinary value"},'
        '{"id":"identifier-two-long","status":"second policy-sensitive value"},'
        '{"id":"identifier-three-long","flag":"third policy-sensitive value"}]'
    )

    assert len(entries) == 2
    assert [entry["label"] for entry in entries] == ["id: identifi", "id: identifi"]
    assert all(len(entry["label"]) <= 12 for entry in entries)
    assert all(len(entry["evidence"]) <= 18 for entry in entries)
    assert entries[0]["evidence"].startswith('"status"')
    assert entries[1]["evidence"].startswith('"flag"')


def test_candidate_checkpoint_is_one_shot_for_same_tool_result() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    call = ToolCall(
        id="records-1",
        name="api_fetch",
        arguments='{"method":"GET","url":"https://example.test/records"}',
    )
    messages = [AssistantMessage(content=None, tool_calls=[call])]
    result = ToolMessage(
        content='[{"id":"one"},{"id":"two"}]',
        tool_call_id=call.id,
    )
    state = {"_successful_tool_call_ids": {call.id}}

    first = agent.append_tool_messages(messages, [result], state)
    state["_successful_tool_call_ids"] = {call.id}
    second = agent.append_tool_messages(messages, [result], state)

    assert any(isinstance(message, UserMessage) for message in first)
    assert not any(isinstance(message, UserMessage) for message in second)


class RecordingRuntime:
    use_meta_tools = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str]] = []

    def should_stop_for_tool_error(self, error: Exception) -> bool:
        return False

    def format_runtime_error(self, error: Exception) -> str:
        return str(error)

    def inject_task_tool_args(self, tool_name: str, tool_args: dict, state: dict) -> dict:
        return tool_args

    async def call_task_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str
    ) -> ToolMessage:
        self.calls.append((tool_name, tool_args, tool_call_id))
        return ToolMessage(content='{"ok":true}', tool_call_id=tool_call_id)


def test_multi_record_mutation_preflight_is_one_shot_and_executes_correction() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    runtime = RecordingRuntime()
    call = ToolCall(
        id="mutate-1",
        name="api_fetch",
        arguments='{"method":"POST","url":"https://example.test/messages","body":"{}"}',
    )
    second_call = ToolCall(
        id="mutate-2",
        name="api_fetch",
        arguments='{"method":"PATCH","url":"https://example.test/records/2","body":"{}"}',
    )
    messages = [AssistantMessage(content=None, tool_calls=[call, second_call])]
    state = {
        "_freshness_sensitive_request": True,
        "_allowed_service_inventory": ["crm", "gmail"],
        "_candidate_roster": [
            {
                "label": "id: person-1 | name: Alpha",
                "evidence": '"notes":"Duplicate registration"',
                "tool_call_id": "read-1",
            },
            {"label": "id: person-2 | name: Beta", "tool_call_id": "read-1"},
        ],
        "_authoritative_update_evidence": [{
            "service": "slack",
            "source": "query=client amendment",
            "tool_call_id": "update-1",
            "snippet": '{"text":"Apply the signed amendment this cycle."}',
        }],
    }

    first = asyncio.run(agent.execute_tool_turn(runtime, messages, state))

    assert runtime.calls == []
    assert len(first) == 2
    assert [message.tool_call_id for message in first] == [call.id, second_call.id]
    assert '"withheld": true' in first[0].content
    assert '"side_effect_occurred": false' in first[0].content
    assert "withheld 2 proposed side effect(s)" in first[0].content
    assert "same one-shot preflight" in first[1].content
    assert "id: person-1 | name: Alpha" in first[0].content
    assert '"notes":"Duplicate registration"' in first[0].content
    assert "id: person-2 | name: Beta" in first[0].content
    assert "raw tool call read-1" in first[0].content
    assert "Free text cannot invent a classification from vague similarity" in (
        first[0].content
    )
    assert "duplicate registration, same person, alternate identity" in first[0].content
    assert "one canonical base mutation" in first[0].content
    assert "separate from checking the destination for an existing object" in first[0].content
    assert "withdrawal, cancellation, opt-out, hold, or completion is operative" in (
        first[0].content
    )
    assert "every roster record an explicit lifecycle disposition" in first[0].content
    assert "Available source systems: crm, gmail" in first[0].content
    assert "Pinned exact update evidence: slack [query=client amendment" in first[0].content
    assert '"text":"Apply the signed amendment this cycle."' in first[0].content
    assert "applied, not-applicable, superseded, or unresolved disposition" in first[0].content
    assert "do not exhaustively search every source system" in first[0].content
    assert "source values, authoritative transformations, final operational values" in (
        first[0].content
    )
    assert "expected-versus-proposed artifact count by operation" in first[0].content
    assert "Compute proposed counts after canonical dispositions" in first[0].content
    assert "notification totals from successful canonical mutations" in first[0].content
    assert "base destination mutation, recipient delivery, approval or routing" in first[0].content
    assert "routing artifact cannot replace the base mutation" in first[0].content
    assert "validate recipient content against record-specific conditions" in first[0].content
    assert state["_multi_record_mutation_preflight_consumed"] is True

    second = asyncio.run(agent.execute_tool_turn(runtime, messages, state))

    assert len(runtime.calls) == 2
    assert runtime.calls[0][2] == call.id
    assert runtime.calls[1][2] == second_call.id
    assert second[0].content == '{"ok":true}'
    assert state["_successful_api_mutation_ids"] == {call.id, second_call.id}


def test_update_evidence_resurfaces_after_setup_and_record_mutations() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    runtime = RecordingRuntime()
    state = {
        "_freshness_sensitive_request": True,
        "_authoritative_update_evidence": [{
            "service": "slack",
            "source": "query=client amendment",
            "tool_call_id": "slack-read",
            "snippet": '{"text":"Apply the signed amendment this cycle."}',
        }],
    }
    setup = ToolCall(
        id="setup-1",
        name="api_fetch",
        arguments='{"method":"POST","url":"https://example.test/setup","body":"{}"}',
    )
    record = ToolCall(
        id="record-1",
        name="api_fetch",
        arguments='{"method":"POST","url":"https://example.test/records","body":"{}"}',
    )

    first_tools = asyncio.run(agent.execute_tool_turn(
        runtime,
        [AssistantMessage(content=None, tool_calls=[setup])],
        state,
    ))
    first = agent.append_tool_messages(
        [AssistantMessage(content=None, tool_calls=[setup])],
        first_tools,
        state,
    )
    second_tools = asyncio.run(agent.execute_tool_turn(
        runtime,
        [AssistantMessage(content=None, tool_calls=[record])],
        state,
    ))
    second = agent.append_tool_messages(
        [AssistantMessage(content=None, tool_calls=[record])],
        second_tools,
        state,
    )

    for combined in (first, second):
        checkpoint = next(
            message
            for message in combined
            if isinstance(message, UserMessage)
            and "Authoritative update reconciliation checkpoint" in message.content
        )
        assert '"text":"Apply the signed amendment this cycle."' in checkpoint.content
        assert "raw tool call slack-read" in checkpoint.content


def test_multi_record_mutation_preflight_bypasses_reads_and_single_records() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    read = ToolCall(
        id="read-1",
        name="api_fetch",
        arguments='{"method":"GET","url":"https://example.test/records"}',
    )
    mutation = ToolCall(
        id="mutate-1",
        name="api_fetch",
        arguments='{"method":"POST","url":"https://example.test/records","body":"{}"}',
    )

    read_runtime = RecordingRuntime()
    read_state = {
        "_freshness_sensitive_request": True,
        "_candidate_roster": [
            {"label": "id: one", "tool_call_id": "source-1"},
            {"label": "id: two", "tool_call_id": "source-1"},
        ],
    }
    asyncio.run(
        agent.execute_tool_turn(
            read_runtime,
            [AssistantMessage(content=None, tool_calls=[read])],
            read_state,
        )
    )

    assert len(read_runtime.calls) == 1
    assert "_multi_record_mutation_preflight_consumed" not in read_state

    single_runtime = RecordingRuntime()
    single_state = {
        "_freshness_sensitive_request": True,
        "_candidate_roster": [{"label": "id: one", "tool_call_id": "source-1"}],
    }
    asyncio.run(
        agent.execute_tool_turn(
            single_runtime,
            [AssistantMessage(content=None, tool_calls=[mutation])],
            single_state,
        )
    )

    assert len(single_runtime.calls) == 1
    assert "_multi_record_mutation_preflight_consumed" not in single_state


def test_multi_record_mutation_preflight_bypasses_ordinary_multi_record_work() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    runtime = RecordingRuntime()
    mutation = ToolCall(
        id="mutate-1",
        name="api_fetch",
        arguments='{"method":"POST","url":"https://example.test/records","body":"{}"}',
    )
    state = {
        "_freshness_sensitive_request": False,
        "_candidate_roster": [
            {"label": "id: one", "tool_call_id": "source-1"},
            {"label": "id: two", "tool_call_id": "source-1"},
        ],
    }

    result = asyncio.run(
        agent.execute_tool_turn(
            runtime,
            [AssistantMessage(content=None, tool_calls=[mutation])],
            state,
        )
    )

    assert len(runtime.calls) == 1
    assert result[0].content == '{"ok":true}'
    assert "_multi_record_mutation_preflight_consumed" not in state


def test_multi_record_mutation_preflight_state_is_fresh_per_rollout() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    call = ToolCall(
        id="mutate-1",
        name="api_fetch",
        arguments='{"method":"PATCH","url":"https://example.test/records/1","body":"{}"}',
    )
    messages = [AssistantMessage(content=None, tool_calls=[call])]

    for _ in range(2):
        runtime = RecordingRuntime()
        state = {
            "_freshness_sensitive_request": True,
            "_candidate_roster": [
                {"label": "id: one", "tool_call_id": "source-1"},
                {"label": "id: two", "tool_call_id": "source-1"},
            ],
        }
        result = asyncio.run(agent.execute_tool_turn(runtime, messages, state))
        assert runtime.calls == []
        assert '"withheld": true' in result[0].content


def test_four_column_spreadsheet_read_creates_schema_checkpoint() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    call = ToolCall(
        id="sheet-read-1",
        name="api_fetch",
        arguments=(
            '{"method":"GET","url":"https://sheets.googleapis.com/v4/spreadsheets/'
            'ss_tracker/values:batchGet","params":"{\\"ranges\\":'
            '\\"Pending Invoices!A1:Z20\\"}"}'
        ),
    )
    messages = [AssistantMessage(content=None, tool_calls=[call])]
    result = ToolMessage(
        content=(
            '{"spreadsheetId":"ss_tracker","valueRanges":[{"range":'
            '"Pending Invoices!A1:Z20","majorDimension":"ROWS","values":'
            '[["Vendor","Invoice","Amount","Notes"]]}]}'
        ),
        tool_call_id=call.id,
    )
    state = {"_successful_tool_call_ids": {call.id}}

    combined = agent.append_tool_messages(messages, [result], state)

    checkpoint = next(
        message
        for message in combined
        if isinstance(message, UserMessage)
        and "Spreadsheet schema checkpoint" in message.content
    )
    assert 'headers ["Vendor","Invoice","Amount","Notes"]' in checkpoint.content
    assert "preserve the observed header order exactly" in checkpoint.content
    assert "never add an extra positional value" in checkpoint.content
    assert "compare each value by this same header-to-position mapping" in checkpoint.content
    assert "raw tool call sheet-read-1" in checkpoint.content
    assert state["_spreadsheet_schemas"][0]["headers"] == [
        "Vendor",
        "Invoice",
        "Amount",
        "Notes",
    ]


def test_repeated_spreadsheet_schema_does_not_duplicate_checkpoint() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    call = ToolCall(
        id="sheet-read-1",
        name="api_fetch",
        arguments=(
            '{"method":"GET","url":"https://sheets.googleapis.com/v4/spreadsheets/'
            'ss_tracker/values:batchGet"}'
        ),
    )
    messages = [AssistantMessage(content=None, tool_calls=[call])]
    result = ToolMessage(
        content=(
            '{"valueRanges":[{"range":"Sheet1!A1:D10","majorDimension":"ROWS",'
            '"values":[["Name","Date","Amount","Notes"],["A","2026-01-01","10",""]]}]}'
        ),
        tool_call_id=call.id,
    )
    state = {"_successful_tool_call_ids": {call.id}}

    first = agent.append_tool_messages(messages, [result], state)
    state["_successful_tool_call_ids"] = {call.id}
    second = agent.append_tool_messages(messages, [result], state)

    assert any(
        isinstance(message, UserMessage)
        and "Spreadsheet schema checkpoint" in message.content
        for message in first
    )
    assert not any(
        isinstance(message, UserMessage)
        and "Spreadsheet schema checkpoint" in message.content
        for message in second
    )


def test_spreadsheet_schema_checkpoint_discourages_six_fields_for_four_columns() -> None:
    message = AutomationAgentConfig().spreadsheet_schema_reconciliation_message.format(
        schemas='headers ["Vendor","Invoice","Amount","Notes"]'
    )

    assert "provide one value position per observed column" in message
    assert "Do not replace, extend, or restate the header row" in message
    assert "If a required field is absent" in message
    assert "never add an extra positional value that shifts later values" in message


def test_unrelated_api_value_result_does_not_create_schema_checkpoint() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    call = ToolCall(
        id="records-read-1",
        name="api_fetch",
        arguments='{"method":"GET","url":"https://example.test/records"}',
    )
    messages = [AssistantMessage(content=None, tool_calls=[call])]
    result = ToolMessage(
        content=(
            '{"valueRanges":[{"range":"Records!A1:B2","majorDimension":"ROWS",'
            '"values":[["Name","Status"],["Alpha","Open"]]}]}'
        ),
        tool_call_id=call.id,
    )
    state = {"_successful_tool_call_ids": {call.id}}

    combined = agent.append_tool_messages(messages, [result], state)

    assert not any(
        isinstance(message, UserMessage)
        and "Spreadsheet schema checkpoint" in message.content
        for message in combined
    )
    assert "_spreadsheet_schemas" not in state


def test_preparation_batch_gets_one_reconciliation_reminder() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    endpoints = [
        {
            "id": "mail.messages.send",
            "method": "POST",
            "url": "https://mail.example.test/messages/send",
            "schema": "Body: {raw}",
        },
        {
            "id": "crm.records.update",
            "method": "PATCH",
            "url": "https://crm.example.test/records/{recordId}",
            "schema": "Body: {properties}",
        },
    ]
    calls = [
        ToolCall(id=f"encode-{index}", name="base64_encode", arguments='{"text":"message"}')
        for index in range(4)
    ]
    messages = [AssistantMessage(content=None, tool_calls=calls)]
    tool_messages = [
        ToolMessage(content=f"encoded-{index}", tool_call_id=call.id)
        for index, call in enumerate(calls)
    ]
    state = {
        "_successful_tool_call_ids": {call.id for call in calls},
        "_discovered_mutation_endpoints": endpoints,
        "_candidate_roster": [
            {"label": "id: person-1 | name: Alpha", "tool_call_id": "read-1"},
            {"label": "id: person-2 | name: Beta", "tool_call_id": "read-1"},
        ],
    }

    combined = agent.append_tool_messages(messages, tool_messages, state)

    assert combined[1:5] == tool_messages
    reminders = [message for message in combined if isinstance(message, UserMessage)]
    assert len(reminders) == 1
    assert "4 successful output(s)" in reminders[0].content
    assert "prepared artifacts are not committed side effects" in reminders[0].content
    assert "mail.messages.send" in reminders[0].content
    assert "crm.records.update" in reminders[0].content
    assert "when every prepared artifact passes, the next response must emit api_fetch" in (
        reminders[0].content
    )
    assert "every outstanding artifact contract" in reminders[0].content
    assert "id: person-1 | name: Alpha" in reminders[0].content
    assert "id: person-2 | name: Beta" in reminders[0].content
    assert "operation-specific contracts for base destination mutations" in reminders[0].content
    assert "remove artifacts for alternate source identities" in reminders[0].content
    assert "records whose lifecycle forbids them" in reminders[0].content
    assert "one canonical base mutation" in reminders[0].content
    assert "separate from destination-object duplicate checks" in reminders[0].content
    assert "missing non-email calendar, record, or status mutations" in reminders[0].content
    assert "email or escalation cannot substitute for those base artifacts" in (
        reminders[0].content
    )
    assert "Prepare only missing permitted artifacts" in reminders[0].content
    assert "outbound values and counts derive from successful canonical destination mutations" in (
        reminders[0].content
    )
    assert "Re-prepare only deficient artifacts" in reminders[0].content
    assert "Keep internal policy conditions out of recipient content" in reminders[0].content
    assert "text-only progress" in reminders[0].content
    assert "'ready to send'" in reminders[0].content
    assert "without a progress preamble" in reminders[0].content
    assert state["_pending_preparation_output_count"] == 4

    state["_successful_tool_call_ids"] = {call.id for call in calls}
    repeated = agent.append_tool_messages(messages, tool_messages, state)
    assert not any(isinstance(message, UserMessage) for message in repeated)


def test_unrelated_tool_batch_does_not_get_preparation_reminder() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    calls = [
        ToolCall(id="search-1", name="api_search", arguments='{"query":"gmail"}'),
        ToolCall(id="fetch-1", name="api_fetch", arguments='{"url":"https://example.com"}'),
    ]
    messages = [AssistantMessage(content=None, tool_calls=calls)]
    tool_messages = [
        ToolMessage(content="result", tool_call_id=call.id)
        for call in calls
    ]
    state = {"_successful_tool_call_ids": {call.id for call in calls}}

    combined = agent.append_tool_messages(messages, tool_messages, state)

    assert combined == [*messages, *tool_messages]


def test_single_successful_preparation_call_gets_reminder() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    calls = [ToolCall(id="encode-1", name="base64_encode", arguments='{"text":"one"}')]
    messages = [AssistantMessage(content=None, tool_calls=calls)]
    tool_messages = [ToolMessage(content="encoded", tool_call_id="encode-1")]
    state = {
        "_successful_tool_call_ids": {"encode-1"},
        "_discovered_mutation_endpoints": [{
            "id": "documents.publish",
            "method": "POST",
            "url": "https://documents.example.test/publish",
            "schema": "Body: {document}",
        }],
    }

    combined = agent.append_tool_messages(messages, tool_messages, state)

    reminders = [message for message in combined if isinstance(message, UserMessage)]
    assert len(reminders) == 1
    assert "1 successful output(s)" in reminders[0].content
    assert "documents.publish" in reminders[0].content
    assert "Do not respond with text-only progress" in reminders[0].content


def test_preparation_without_discovered_mutation_does_not_get_execution_checkpoint() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    call = ToolCall(id="encode-1", name="base64_encode", arguments='{"text":"one"}')
    messages = [AssistantMessage(content=None, tool_calls=[call])]
    tool_messages = [ToolMessage(content="encoded", tool_call_id=call.id)]
    state = {"_successful_tool_call_ids": {call.id}}

    combined = agent.append_tool_messages(messages, tool_messages, state)

    assert combined == [*messages, *tool_messages]
    assert "_pending_preparation_output_count" not in state


def test_repeated_api_search_batches_get_one_endpoint_grounded_checkpoint() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    state = {}

    first_call = ToolCall(
        id="search-1", name="api_search", arguments='{"query":"inbox messages"}'
    )
    first_messages = [AssistantMessage(content="Searching.", tool_calls=[first_call])]
    first_result = ToolMessage(
        content=(
            '{"results":[{"id":"gmail.users.messages.list","method":"GET",'
            '"url":"https://example.test/messages"}]}'
        ),
        tool_call_id=first_call.id,
    )
    state["_successful_tool_call_ids"] = {first_call.id}

    first_combined = agent.append_tool_messages(first_messages, [first_result], state)

    assert first_combined == [*first_messages, first_result]
    assert first_result.content in [message.content for message in first_combined]

    second_call = ToolCall(
        id="search-2", name="api_search", arguments='{"query":"tracker rows"}'
    )
    second_messages = [AssistantMessage(content="Searching again.", tool_calls=[second_call])]
    second_result = ToolMessage(
        content=(
            '{"results":[{"id":"google_sheets.spreadsheets.values.get","method":"GET",'
            '"url":"https://example.test/rows"}]}'
        ),
        tool_call_id=second_call.id,
    )
    state["_successful_tool_call_ids"] = {second_call.id}

    second_combined = agent.append_tool_messages(second_messages, [second_result], state)

    assert second_combined[1] is second_result
    checkpoints = [message for message in second_combined if isinstance(message, UserMessage)]
    assert len(checkpoints) == 1
    assert "gmail.users.messages.list" in checkpoints[0].content
    assert "google_sheets.spreadsheets.values.get" in checkpoints[0].content
    assert "Use api_fetch" in checkpoints[0].content
    assert "specifically named missing service and operation" in checkpoints[0].content
    assert "without a progress preamble" in checkpoints[0].content
    assert "Do not shorten tool arguments or external message payloads" in (
        checkpoints[0].content
    )

    third_call = ToolCall(
        id="search-3", name="api_search", arguments='{"query":"slack users"}'
    )
    third_messages = [AssistantMessage(content=None, tool_calls=[third_call])]
    third_result = ToolMessage(
        content=(
            '{"results":[{"id":"slack.users.list","method":"GET",'
            '"url":"https://example.test/users"}]}'
        ),
        tool_call_id=third_call.id,
    )
    state["_successful_tool_call_ids"] = {third_call.id}

    third_combined = agent.append_tool_messages(third_messages, [third_result], state)

    assert not any(isinstance(message, UserMessage) for message in third_combined)


def test_api_fetch_resets_discovery_checkpoint_streak() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    state = {}

    for index in (1, 2):
        call = ToolCall(
            id=f"search-{index}", name="api_search", arguments='{"query":"source records"}'
        )
        messages = [AssistantMessage(content=None, tool_calls=[call])]
        result = ToolMessage(
            content=(
                f'{{"results":[{{"id":"service.records.list.{index}",'
                f'"method":"GET","url":"https://example.test/records/{index}"}}]}}'
            ),
            tool_call_id=call.id,
        )
        state["_successful_tool_call_ids"] = {call.id}
        combined = agent.append_tool_messages(messages, [result], state)

    assert any(isinstance(message, UserMessage) for message in combined)

    fetch_call = ToolCall(
        id="fetch-1",
        name="api_fetch",
        arguments='{"method":"GET","url":"https://example.test/records"}',
    )
    fetch_messages = [AssistantMessage(content=None, tool_calls=[fetch_call])]
    fetch_result = ToolMessage(content='{"records":[]}', tool_call_id=fetch_call.id)
    state["_successful_tool_call_ids"] = {fetch_call.id}

    fetch_combined = agent.append_tool_messages(fetch_messages, [fetch_result], state)

    assert fetch_combined == [*fetch_messages, fetch_result]
    assert state["_api_search_only_streak"] == 0

    next_search = ToolCall(
        id="search-3", name="api_search", arguments='{"query":"missing mutation"}'
    )
    next_messages = [AssistantMessage(content=None, tool_calls=[next_search])]
    next_result = ToolMessage(
        content=(
            '{"results":[{"id":"service.records.create","method":"POST",'
            '"url":"https://example.test/records","request":"Body: {record: object}"}]}'
        ),
        tool_call_id=next_search.id,
    )
    state["_successful_tool_call_ids"] = {next_search.id}

    next_combined = agent.append_tool_messages(next_messages, [next_result], state)

    checkpoint = next(message for message in next_combined if isinstance(message, UserMessage))
    assert "Known mutation endpoints" in checkpoint.content
    assert "service.records.create" in checkpoint.content
    assert "Body: {record: object}" in checkpoint.content
    assert "Endpoint discovery does not resolve opaque argument identifiers" in checkpoint.content
    assert "schema examples are not evidence" in checkpoint.content
    assert "preserve the task literal rather than substitute an example" in checkpoint.content
    assert "at most one targeted metadata fetch" in checkpoint.content
    assert "Do not repeat broad searches" in checkpoint.content
    assert state["_api_search_only_streak"] == 1


def test_mutation_checkpoint_identifier_policy_is_service_independent() -> None:
    for service, task_field, example_id in (
        ("crm", "renewal_at", "custom_17"),
        ("tickets", "approval_state", "field_200"),
        ("logistics", "dispatch_zone", "zone_example"),
    ):
        message = AutomationAgentConfig().discovery_mutation_progression_message.format(
            mutation_endpoints=(
                f"{service}.records.update [PATCH /records/{{id}}; "
                f"schema example identifier: {example_id}]"
            ),
            fetch_tool_name="api_fetch",
        )

        assert task_field not in message
        assert example_id in message
        assert "task literal, a successful read, or endpoint metadata" in message
        assert "schema examples are not evidence" in message
        assert "preserve the task literal rather than substitute an example" in message
        assert "Prefer an authoritative metadata mapping" in message


def test_mutation_endpoints_persist_when_later_read_searches_are_bounded() -> None:
    config = AutomationAgentConfig(discovery_endpoint_limit=2)
    agent = BaselineAutomationAgent(config)
    state = {}
    first_call = ToolCall(
        id="search-mutation", name="api_search", arguments='{"query":"create record"}'
    )
    first_messages = [AssistantMessage(content=None, tool_calls=[first_call])]
    first_result = ToolMessage(
        content=(
            '{"results":['
            '{"id":"service.records.list","method":"GET","url":"https://example.test/records"},'
            '{"id":"service.records.create","method":"POST","url":"https://example.test/records",'
            '"request":"Body schema: record name (string, required)"}]}'
        ),
        tool_call_id=first_call.id,
    )
    state["_successful_tool_call_ids"] = {first_call.id}

    combined = agent.append_tool_messages(first_messages, [first_result], state)

    checkpoint = next(message for message in combined if isinstance(message, UserMessage))
    assert "service.records.create" in checkpoint.content
    assert "POST https://example.test/records" in checkpoint.content
    assert "name (string, required)" in checkpoint.content
    assert [endpoint["id"] for endpoint in state["_discovered_read_endpoints"]] == [
        "service.records.list"
    ]
    assert [endpoint["id"] for endpoint in state["_discovered_mutation_endpoints"]] == [
        "service.records.create"
    ]

    call = ToolCall(
        id="search-reads", name="api_search", arguments='{"query":"read operations"}'
    )
    messages = [AssistantMessage(content=None, tool_calls=[call])]
    result = ToolMessage(
        content=(
            '{"results":['
            '{"id":"one.list","method":"GET","url":"https://example.test/one"},'
            '{"id":"two.get","method":"GET","url":"https://example.test/two"},'
            '{"id":"three.list","method":"GET","url":"https://example.test/three"}]}'
        ),
        tool_call_id=call.id,
    )
    state["_successful_tool_call_ids"] = {call.id}

    agent.append_tool_messages(messages, [result], state)

    assert [endpoint["id"] for endpoint in state["_discovered_read_endpoints"]] == [
        "two.get",
        "three.list",
    ]
    assert [endpoint["id"] for endpoint in state["_discovered_mutation_endpoints"]] == [
        "service.records.create"
    ]
    for benchmark_value in ("Senior DevOps Engineer", "U_DAVID", "VP Product"):
        assert benchmark_value not in agent.format_endpoint_checkpoint(
            state["_discovered_mutation_endpoints"]
        )


def test_later_mutation_searches_preserve_diverse_create_capability() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig(discovery_mutation_endpoint_limit=3))
    state = {}
    first_call = ToolCall(
        id="search-calendar", name="api_search", arguments='{"query":"calendar event create"}'
    )
    first_messages = [AssistantMessage(content=None, tool_calls=[first_call])]
    first_result = ToolMessage(
        content=(
            '{"results":[{"id":"calendar.events.insert","method":"POST",'
            '"url":"https://calendar.example.test/calendars/{calendarId}/events",'
            '"request":"{summary,start,end}"}]}'
        ),
        tool_call_id=first_call.id,
    )
    state["_successful_tool_call_ids"] = {first_call.id}
    agent.append_tool_messages(first_messages, [first_result], state)

    later_call = ToolCall(
        id="search-mail", name="api_search", arguments='{"query":"mail administration"}'
    )
    later_messages = [AssistantMessage(content=None, tool_calls=[later_call])]
    later_result = ToolMessage(
        content=(
            '{"results":['
            '{"id":"gmail.users.messages.trash","method":"POST",'
            '"url":"https://gmail.example.test/users/{userId}/messages/{id}/trash"},'
            '{"id":"gmail.users.messages.delete","method":"DELETE",'
            '"url":"https://gmail.example.test/users/{userId}/messages/{id}"},'
            '{"id":"gmail.users.messages.send","method":"POST",'
            '"url":"https://gmail.example.test/users/{userId}/messages/send"},'
            '{"id":"slack.conversations.archive","method":"POST",'
            '"url":"https://slack.example.test/conversations.archive"}]}'
        ),
        tool_call_id=later_call.id,
    )
    state["_successful_tool_call_ids"] = {later_call.id}
    agent.append_tool_messages(later_messages, [later_result], state)

    retained_ids = [item["id"] for item in state["_discovered_mutation_endpoints"]]
    assert "calendar.events.insert" in retained_ids
    assert "gmail.users.messages.send" in retained_ids
    assert "gmail.users.messages.trash" not in retained_ids
    calendar = next(
        item for item in state["_discovered_mutation_endpoints"]
        if item["id"] == "calendar.events.insert"
    )
    assert calendar["service"] == "calendar"
    assert calendar["resource"] == "events"
    assert calendar["family"] == "calendar.events"


def test_malformed_search_results_do_not_displace_retained_endpoints() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    retained = {
        "id": "service.records.create",
        "method": "POST",
        "url": "https://example.test/records",
        "schema": "Body: {record}",
    }
    state = {"_discovered_mutation_endpoints": [retained]}
    call = ToolCall(id="search-bad", name="api_search", arguments='{"query":"more"}')
    messages = [AssistantMessage(content=None, tool_calls=[call])]
    result = ToolMessage(
        content='{"results":[null,{"id":"missing.method"},{"method":"POST","url":"x"}]}',
        tool_call_id=call.id,
    )
    state["_successful_tool_call_ids"] = {call.id}

    agent.append_tool_messages(messages, [result], state)

    assert state["_discovered_mutation_endpoints"] == [retained]


def test_budget_checkpoint_is_one_shot_and_requires_known_mutation() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig(budget_checkpoint_remaining_turns=6))
    endpoint = {
        "id": "service.records.create",
        "method": "POST",
        "url": "https://example.test/records",
        "schema": "Body: {record}",
    }
    state = {
        "_rollout_turn_limit": 20,
        "trajectory": [{} for _ in range(14)],
    }

    first = agent.budget_execution_checkpoint(state, [endpoint])
    second = agent.budget_execution_checkpoint(state, [endpoint])

    assert first is not None
    assert "6 turns remain" in first.content
    assert "next turn must use a known mutation endpoint" in first.content
    assert "Turn pressure does not waive identifier provenance" in first.content
    assert "never a schema example" in first.content
    assert "otherwise preserve the task literal" in first.content
    assert "at most one targeted metadata fetch" in first.content
    assert "Do not repeat broad searches" in first.content
    assert "Do not explore adjacent setup" in first.content
    assert second is None
    assert agent.budget_execution_checkpoint({}, [endpoint]) is None
    assert agent.budget_execution_checkpoint(state, []) is None


def test_successful_mutation_suppresses_budget_checkpoint_but_error_does_not() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    endpoint = {
        "id": "service.records.create",
        "method": "POST",
        "url": "https://example.test/records",
        "schema": "Body: {record}",
    }
    state = {"_rollout_turn_limit": 10, "trajectory": [{}, {}, {}]}
    agent.record_successful_api_mutation(
        "api_fetch",
        {"method": "POST", "url": endpoint["url"]},
        ToolMessage(content='{"record":{"id":"1"}}', tool_call_id="fetch-ok"),
        "fetch-ok",
        state,
    )

    assert state["_successful_api_mutation_ids"] == {"fetch-ok"}
    assert state["_successful_api_mutation_families"]
    assert state["_successful_api_mutation_evidence"] == [{
        "family": "example.records",
        "method": "POST",
        "destination": endpoint["url"],
        "artifact_id": "1",
    }]
    assert agent.budget_execution_checkpoint(state, [endpoint]) is None

    error_state = {"_rollout_turn_limit": 10, "trajectory": [{}, {}, {}]}
    agent.record_successful_api_mutation(
        "api_fetch",
        {"method": "POST", "url": endpoint["url"]},
        ToolMessage(
            content='{"error":{"code":422,"message":"missing field"}}',
            tool_call_id="fetch-error",
        ),
        "fetch-error",
        error_state,
    )

    assert "_successful_api_mutation_ids" not in error_state
    assert agent.budget_execution_checkpoint(error_state, [endpoint]) is not None


def _completion_state(content: str, *, turns: int = 1, turn_limit: int = 10) -> dict:
    trajectory = [{} for _ in range(turns - 1)]
    trajectory.append({
        "prompt": [UserMessage(content="Complete the requested workflow.")],
        "completion": [AssistantMessage(content=content)],
    })
    return {"_rollout_turn_limit": turn_limit, "trajectory": trajectory}


def test_unfinished_completion_detection_is_task_agnostic() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())

    assert agent.should_reconcile_unfinished_completion(
        _completion_state("Invoice created. Creating the next invoice now.")
    )
    for content in (
        "Two notifications remain pending.",
        "Preparing client notifications.",
        "We will send the remaining update.",
    ):
        assert agent.should_reconcile_unfinished_completion(_completion_state(content))


def test_unfinished_completion_detection_excludes_completed_and_blocked_reports() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())

    for content in (
        "All requested actions completed successfully.",
        "No pending or unresolved work remains.",
        "The remaining update cannot be sent because access is denied.",
        "Blocked by missing required credentials.",
    ):
        assert not agent.should_reconcile_unfinished_completion(_completion_state(content))


def test_completed_multi_family_workflow_gets_one_evidence_audit() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    state = _completion_state("All requested actions completed successfully.")
    state["_discovered_mutation_endpoints"] = [
        {
            "id": "calendar.events.insert",
            "method": "POST",
            "url": "https://calendar.example.test/calendars/{calendarId}/events",
            "schema": "{summary,start,end}",
            "service": "calendar",
            "resource": "events",
            "family": "calendar.events",
        },
        {
            "id": "gmail.users.messages.send",
            "method": "POST",
            "url": "https://gmail.example.test/users/{userId}/messages/send",
            "schema": "{raw}",
            "service": "gmail",
            "resource": "users.messages",
            "family": "gmail.users.messages",
        },
    ]
    state["_successful_api_mutation_families"] = {"gmail.users.messages"}
    state["_successful_api_mutation_evidence"] = [{
        "family": "gmail.users.messages",
        "method": "POST",
        "destination": "https://gmail.example.test/users/me/messages/send",
        "artifact_id": "message-1",
    }]

    assert agent.should_reconcile_unfinished_completion(state)
    prompt = agent.completion_reconciliation_prompt(state)
    assert "calendar.events.insert" in prompt
    assert '"artifact_id":"message-1"' in prompt
    assert "Match each obligation one-to-one to exact successful evidence" in prompt
    assert "Do not synthesize recipients or obligations" in prompt

    state["_completion_reconciliation_retries"] = 1
    assert not agent.should_reconcile_unfinished_completion(state)


def test_completed_same_family_multi_artifact_workflow_gets_one_evidence_audit() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    state = _completion_state("All requested actions completed successfully.")
    state["_discovered_mutation_endpoints"] = [
        {
            "id": "calendar.events.insert",
            "method": "POST",
            "url": "https://calendar.example.test/calendars/{calendarId}/events",
            "schema": "{summary,start,end,attendees}",
            "service": "calendar",
            "resource": "events",
            "family": "calendar.events",
        },
        {
            "id": "sheets.spreadsheets.create",
            "method": "POST",
            "url": "https://sheets.example.test/spreadsheets",
            "schema": "{title}",
            "service": "sheets",
            "resource": "spreadsheets",
            "family": "sheets.spreadsheets",
        },
    ]
    state["_successful_api_mutation_families"] = {"calendar.events"}
    state["_successful_api_mutation_evidence"] = [
        {
            "family": "calendar.events",
            "method": "POST",
            "destination": "https://calendar.example.test/calendars/main/events",
            "artifact_id": "event-1",
            "confirmed": {"attendees": ["one@example.test"]},
        },
        {
            "family": "calendar.events",
            "method": "POST",
            "destination": "https://calendar.example.test/calendars/main/events",
            "artifact_id": "event-2",
            "confirmed": {"attendees": ["two@example.test"]},
        },
    ]

    assert agent.should_reconcile_unfinished_completion(state)
    prompt = agent.completion_reconciliation_prompt(state)
    assert '"artifact_id":"event-1"' in prompt
    assert '"artifact_id":"event-2"' in prompt
    assert prompt.count("calendar.events.insert") == 1
    assert "sheets.spreadsheets.create" not in prompt

    state["_completion_reconciliation_retries"] = 1
    assert not agent.should_reconcile_unfinished_completion(state)


def test_successful_artifact_evidence_keeps_exact_requested_and_confirmed_targets() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig(mutation_evidence_list_limit=2))
    state = {}
    agent.record_successful_api_mutation(
        "api_fetch",
        {
            "method": "POST",
            "url": "https://calendar.example.test/calendars/main/events",
            "body": (
                '{"summary":"Review","attendees":['
                '{"email":"owner@example.test"},{"email":"reviewer@example.test"},'
                '{"email":"overflow@example.test"}]}'
            ),
        },
        ToolMessage(
            content=(
                '{"success":true,"event":{"id":"event-1","summary":"Review","attendees":['
                '{"email":"owner@example.test"}]}}'
            ),
            tool_call_id="event-call",
        ),
        "event-call",
        state,
    )
    raw_message = base64.urlsafe_b64encode(
        b"To: recipient@example.test\nSubject: Review complete\n\nDone."
    ).decode()
    agent.record_successful_api_mutation(
        "api_fetch",
        {
            "method": "POST",
            "url": "https://mail.example.test/users/me/messages/send",
            "body": f'{{"raw":"{raw_message}"}}',
        },
        ToolMessage(
            content='{"id":"message-1","threadId":"thread-1"}',
            tool_call_id="mail-call",
        ),
        "mail-call",
        state,
    )

    evidence = state["_successful_api_mutation_evidence"]
    assert evidence[0]["requested"]["attendees"] == [
        "owner@example.test",
        "reviewer@example.test",
    ]
    assert evidence[0]["confirmed"]["attendees"] == ["owner@example.test"]
    assert evidence[0]["artifact_id"] == "event-1"
    assert evidence[1]["requested"]["recipients"] == ["recipient@example.test"]
    assert evidence[1]["artifact_id"] == "message-1"
    serialized = agent.format_successful_mutation_evidence(evidence)
    assert "reviewer@example.test" in serialized
    assert "recipient@example.test" in serialized
    assert "overflow@example.test" not in serialized
    assert "unobserved@example.test" not in serialized


def test_completed_single_operation_workflow_avoids_evidence_audit() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    state = _completion_state("All requested actions completed successfully.")
    state["_discovered_mutation_endpoints"] = [{
        "id": "gmail.users.messages.send",
        "method": "POST",
        "url": "https://gmail.example.test/users/{userId}/messages/send",
        "schema": "{raw}",
        "service": "gmail",
        "resource": "users.messages",
        "family": "gmail.users.messages",
    }]
    state["_successful_api_mutation_families"] = {"gmail.users.messages"}

    assert not agent.should_reconcile_unfinished_completion(state)


def test_unfinished_completion_guard_respects_retry_and_hard_stops() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    state = _completion_state("Preparing the next update.")
    state["_completion_reconciliation_retries"] = 1
    assert not agent.should_reconcile_unfinished_completion(state)

    for hard_stop in (
        {"error": RuntimeError("failed")},
        {"prompt_too_long": True},
        {"is_truncated": True},
        {"final_env_response": "cancelled"},
    ):
        state = _completion_state("Preparing the next update.")
        state.update(hard_stop)
        assert not agent.should_reconcile_unfinished_completion(state)

    tool_state = _completion_state("Preparing the next update.")
    tool_state["trajectory"][-1]["completion"] = [AssistantMessage(
        content="Preparing the next update.",
        tool_calls=[ToolCall(id="call-1", name="api_fetch", arguments="{}")],
    )]
    assert not agent.should_reconcile_unfinished_completion(tool_state)


def test_unfinished_completion_guard_requires_remaining_turn() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())

    assert agent.should_reconcile_unfinished_completion(
        _completion_state("Preparing the next update.", turns=2, turn_limit=3)
    )
    assert not agent.should_reconcile_unfinished_completion(
        _completion_state("Preparing the next update.", turns=3, turn_limit=3)
    )


class CompletionGuardRuntime:
    max_turns = 10

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts = []
        self.finalized = False

    async def initialize_rollout_state(self, input, client, model, sampling_args):
        return {
            "prompt": [UserMessage(content="Complete the requested workflow.")],
            "trajectory": [],
        }

    async def setup_benchmark_task_state(self, state):
        return state

    async def is_rollout_complete(self, state):
        return bool(state["trajectory"])

    async def request_model_response(self, state, prompt_messages):
        self.prompts.append(prompt_messages)
        return self.responses.pop(0)

    async def append_model_response(self, state, prompt_messages, response):
        state["trajectory"].append({
            "prompt": prompt_messages,
            "completion": [AssistantMessage(content=response)],
        })

    async def finalize_rollout(self, state):
        self.finalized = True


def test_run_rollout_injects_one_completion_reconciliation_turn() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    runtime = CompletionGuardRuntime([
        "Created the first record. Creating the next record now.",
        "All requested actions completed successfully.",
    ])

    state = asyncio.run(agent.run_rollout(
        runtime=runtime,
        input={},
        client=None,
        model=agent.default_model,
    ))

    assert runtime.finalized is True
    assert len(runtime.prompts) == 2
    assert state["_completion_reconciliation_retries"] == 1
    assert runtime.prompts[1][-1].content == agent.config.completion_reconciliation_message


def test_side_effect_audit_can_return_to_ordinary_tool_execution() -> None:
    agent = BaselineAutomationAgent(AutomationAgentConfig())
    state = _completion_state("All requested actions completed successfully.")
    state["_discovered_mutation_endpoints"] = [
        {
            "id": "calendar.events.insert",
            "method": "POST",
            "url": "https://calendar.example.test/calendars/{calendarId}/events",
            "schema": "{summary,start,end}",
            "service": "calendar",
            "resource": "events",
            "family": "calendar.events",
        },
        {
            "id": "gmail.users.messages.send",
            "method": "POST",
            "url": "https://gmail.example.test/users/{userId}/messages/send",
            "schema": "{raw}",
            "service": "gmail",
            "resource": "users.messages",
            "family": "gmail.users.messages",
        },
    ]
    state["_successful_api_mutation_families"] = {"gmail.users.messages"}
    state["_successful_api_mutation_evidence"] = [{
        "family": "gmail.users.messages",
        "method": "POST",
        "destination": "https://gmail.example.test/users/me/messages/send",
        "artifact_id": "message-1",
    }]

    class AuditRuntime:
        async def request_model_response(self, state, prompt_messages):
            return "tool-response"

        async def append_model_response(self, state, prompt_messages, response):
            state["trajectory"].append({
                "prompt": prompt_messages,
                "completion": [AssistantMessage(
                    content=None,
                    tool_calls=[ToolCall(
                        id="create-1",
                        name="api_fetch",
                        arguments=(
                            '{"method":"POST","url":'
                            '"https://calendar.example.test/calendars/main/events"}'
                        ),
                    )],
                )],
            })

    reconciled = asyncio.run(agent.reconcile_unfinished_completion(AuditRuntime(), state))

    assert reconciled is True
    assert agent.has_pending_tool_calls(state)


def test_openrouter_session_id_is_stable_for_a_task() -> None:
    first = OpenRouterChatCompletionsClient.session_id_for_state({"task": "finance.one", "example_id": 3})
    second = OpenRouterChatCompletionsClient.session_id_for_state({"task": "finance.one", "example_id": 3})
    other = OpenRouterChatCompletionsClient.session_id_for_state({"task": "finance.two", "example_id": 4})
    assert first == second
    assert first != other
