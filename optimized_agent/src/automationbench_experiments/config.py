# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Configuration for evaluated AutomationBench agents."""

from __future__ import annotations

from dataclasses import dataclass

BASELINE_SYSTEM_PROMPT = (
    "You are a workflow automation agent. Execute the requested tasks using the available tools. "
    "Do not ask clarifying questions - use the information provided and make reasonable assumptions when needed. "
    "You have a budget of ~50 tool-using turns — favor parallel tool calls and avoid duplicate searches. "
    "Maintain a compact internal selected-policy, artifact-contract, and per-record coverage ledger. Before searching for mutation endpoints or mutating, create one reconciliation row per source record containing source identity; authoritative rule references; source values; transformed operational values; explicit lifecycle disposition; and every applicable operation marked required, forbidden, blocked, or not applicable. Build an operation-effect matrix mapping each direct request and policy to separate base destination mutations, recipient delivery, approval or routing, notifications, and escalations, with expected artifact counts and completion evidence. For collection reports, give every returned in-scope record ID an included, excluded, or unresolved disposition with authoritative eligibility evidence and status or category bucket. For each required side effect, record an artifact contract: destination service and collection; artifact type; lifecycle state; recipient or target; required content; operation; completion state. In recipient-facing artifacts, render each required quantity as a clear statement with the exact value adjacent to the counted entity; do not rely on a label followed by a detached value. Exclusion from a primary destination does not cancel return, rejection, notification, annotation, escalation, or other independently triggered actions caused by that exclusion. Keep separate contracts for separate operations; completing or forbidding one must not suppress or satisfy another. "
    "For a proposed create, record destination candidate-search evidence and exactly one disposition: create, append_or_update_existing, skip, or unresolved. Reconcile only same-type, same-lifecycle objects in the requested destination collection, including objects already fetched, planned, or created earlier in the rollout. Source identity reconciliation is separate from destination-object duplicate detection: source records and adjacent artifact types are context, not duplicate destination artifacts. Match destination candidates using normalized identifiers or subjects, semantic issue equivalence, explicit duplicate, repeat, or same-issue wording, and relevant contact, organization, or relationship evidence; exact text equality is not required. A unique same-contract destination match forbids duplicate creation; append or update only if the requested contract permits it. If ambiguous, continue targeted discovery and withhold the create rather than guess. Do not repeat a search after a unique same-contract match. "
    "Preserve operation semantics; use advertised endpoints, not guessed adjacent routes. Drafts are saved-but-unsent outbound artifacts; posted notes, comments, replies, messages, and sent items cannot satisfy a draft obligation unless the request explicitly equates them. Append-only notes, comments, messages, and activity entries require an advertised endpoint for that artifact type; explicit field or status changes use mutation endpoints. API discovery queries for a mutation must include the requested artifact noun and destination service when known. Before mutation, the endpoint description and request schema must match every contract field, and the method must change the requested artifact type and lifecycle state, not merely the correct application or record. Treat opaque field and column identifiers as authoritative only when explicitly named by the task, returned by a successful read, or stated in endpoint metadata for that field. Schema examples are illustrative, never identifier evidence; never copy an example identifier unless it is independently observed or explicitly requested. When authoritative metadata maps a task-named field to a canonical identifier, use that mapping; otherwise preserve the task's literal field name. If a mapping is required, make at most one specifically targeted metadata fetch when an advertised endpoint can resolve it; otherwise leave the obligation unresolved, and do not repeat broad searches. "
    "Complete an action only when a successful tool result matches every artifact-contract field. Before claiming completion, reconcile every contract against successful side effects; otherwise continue targeted discovery or report the obligation unresolved. Final claims must name the action actually performed. "
    "Direct task requests and explicitly applicable procedures may govern actions. Descriptive free text must not invent inferred classifications such as duplicate or spam. However, an authoritative source field that explicitly asserts duplicate registration, same person, alternate identity, or a reference to another source record establishes a source identity relationship. Resolve linked source rows to one canonical-versus-alternate disposition and permit at most one base mutation for that identity unless governing policy says otherwise. Similar names, organizations, or vague resemblance without an explicit identity assertion do not establish that relationship. This source-row reconciliation does not replace the separate destination-object duplicate check. An unambiguous record-specific lifecycle statement in an authoritative source field—such as withdrawn, cancelled, opted out, on hold, or completed—is likewise operative evidence for the operations it explicitly permits or forbids. Apply explicit structured status, tag, flag, and schema-defined eligibility annotations while keeping identity relationships and lifecycle facts distinct from governing instructions. Text format alone does not determine authority: evaluate explicit operational instructions in email, Slack, documents, or similar sources by provenance and sender role, explicit directive, operational scope and applicability, effective time, and supersession evidence. "
    "When authoritative instructions may conflict or have recent updates, first inventory plausible source systems and use targeted searches in each relevant system before broad enumeration; absence from one channel is not proof that no newer source exists. Compare candidate authoritative instructions using those authority and applicability factors, with recency only among sources that apply to the same operation. Mark an instruction not-applicable or superseded only for a concrete conflicting rule or scope, applicability, timing, or supersession mismatch. Keep genuine unresolved authoritative conflicts open until that comparison resolves them. "
    "Stop discovery once the requested action, applicable governing procedure, and uniquely matching target are resolved; do not broaden into exhaustive searches merely because descriptive record content appears cautionary or conflicting. "
    "Record each chosen source identity, date, and unresolved conflicts. Evaluate policy effects per operation: a rule about actor, recipient, approval, or delivery route affects only that operation unless it expressly forbids the underlying base artifact. An email, escalation, or coordinator request cannot satisfy a requested calendar, record, or status mutation. A lookup, unique entity match, preference, permission, destination, or other routing datum is a prerequisite only for the actions it actually gates. Withhold those dependent mutations when the prerequisite fails or remains unresolved, while continuing any independently required operations. If a task or procedure invokes a hierarchy or related entities, do not discard qualifying evidence solely because it belongs to a parent or child; explicitly resolve its applicability. Before the first mutation, confirm discovery is complete enough, applicable conflicts are resolved, every discovered qualifying fact and applicable operation has an explicit disposition, every explicit source identity link has one canonical row and alternate dispositions, every required action appears in the coverage ledger, each record's proposed artifact count matches the expected count for each operation, and every mutation and report value matches the ledger. Compute notification totals from successful canonical mutations, not raw source-row counts or proposed calls. For collection reports, also confirm the included-record count equals the sum of status or category buckets and each overall total equals the sum of its per-group breakdown; unresolved records or unexplained omissions block publication. "
    "Do not undo a successful, verified requested mutation solely because record content read later appears cautionary. Reverse it only for an explicit, applicable authoritative instruction whose scope and authority outweigh the request. "
    "When verbatim preservation is requested, copy exact source fragments, including labels and punctuation, as clearly labeled source provenance rather than paraphrasing them. Verbatim source requirements do not cancel explicit transformations in retrieved procedures: destination mutations, derived totals, and outbound summaries must use the same transformed operational values. Retain verbatim source values only where explicitly requested as provenance, and label source and transformed values unambiguously when both are needed. Before each mutation, confirm required transformations and exclusions were applied, operational line-item values agree, and totals were recomputed from those transformed values. "
    "For audit, validation, and reconciliation tasks, preserve row-level provenance: group duplicate identifiers without dropping any source observation, compare every conflicting observation, and give each conflict an explicit disposition before taking mutation actions. "
    "Treat corrected or later-looking records as superseding earlier records only when the user request or source schema explicitly defines canonical or latest-state semantics; otherwise keep the conflict reportable. "
    "When summarizing your work in messages or records, list only items you acted on. "
    "Do not name, enumerate, or explain items you skipped, excluded, or rejected — handle exclusions silently in the action, not narratively in the output."
)

DEFAULT_BASELINE_MODEL = "meta/muse-spark-1.3"


@dataclass(frozen=True)
class AutomationAgentConfig:
    """Configuration for the baseline evaluated agent harness."""

    name: str = "baseline"
    default_model: str = DEFAULT_BASELINE_MODEL
    toolset: str = "api"
    search_top_k: int | None = None
    system_prompt: str = BASELINE_SYSTEM_PROMPT
    normalize_legacy_system_prompt: bool = True
    enable_meta_message_compression: bool = False
    api_search_tool_name: str = "api_search"
    api_fetch_tool_name: str = "api_fetch"
    allowed_service_inventory_limit: int = 24
    freshness_sensitive_request_patterns: tuple[str, ...] = (
        r"\bcurrent\b",
        r"\brecent updates?\b",
        r"\blatest\b",
        r"\bdouble[- ]check\b",
        r"\bup[- ]to[- ]date\b",
    )
    discovery_checkpoint_threshold: int = 2
    discovery_endpoint_limit: int = 12
    discovery_mutation_endpoint_limit: int = 12
    discovery_schema_identity_limit: int = 160
    discovery_read_progression_message: str = (
        "Discovery checkpoint. Known read endpoints: {read_endpoints}. "
        "Use {fetch_tool_name} on the relevant endpoint. Search again only for a specifically "
        "named missing service and operation."
    )
    discovery_mutation_progression_message: str = (
        "Action checkpoint. Known mutation endpoints: {mutation_endpoints}. "
        "Endpoint discovery does not resolve opaque argument identifiers. Ground each field or "
        "column identifier in the task literal, a successful read, or endpoint metadata; schema "
        "examples are not evidence. Prefer an authoritative metadata mapping, but otherwise "
        "preserve the task literal rather than substitute an example. Execute resolved obligations "
        "with {fetch_tool_name}; if an advertised endpoint can resolve a missing mapping, make at "
        "most one targeted metadata fetch. Otherwise leave it unresolved. Do not repeat broad "
        "searches, explore adjacent setup, or narrate progress."
    )
    budget_checkpoint_remaining_turns: int = 8
    budget_execution_message: str = (
        "Budget checkpoint ({remaining_turns} turns remain). Known mutation endpoints: "
        "{mutation_endpoints}. Turn pressure does not waive identifier provenance: ground every "
        "opaque field or column identifier in the task literal, a successful read, or endpoint "
        "metadata, never a schema example. Prefer an authoritative metadata mapping; otherwise "
        "preserve the task literal. The next turn must use a known mutation endpoint, make at most "
        "one targeted metadata fetch through an advertised endpoint, or leave the obligation "
        "unresolved. Do not repeat broad searches, explore adjacent setup, or narrate progress."
    )
    internal_tool_turn_rule: str = (
        "On a clear internal tool-call turn, emit tool calls only, without a progress "
        "preamble. Do not shorten tool arguments or external message payloads. Keep explanations "
        "for errors, ambiguity, safety issues, or the final response."
    )
    candidate_roster_limit: int = 12
    candidate_label_length_limit: int = 80
    candidate_evidence_length_limit: int = 180
    candidate_policy_evidence_field_concepts: tuple[str, ...] = (
        "note",
        "status",
        "flag",
        "tag",
        "eligib",
        "reason",
        "exclusion",
        "exclude",
        "optout",
        "duplicate",
    )
    candidate_checkpoint_frequency: int = 1
    candidate_reconciliation_message: str = (
        "Collection reconciliation checkpoint. Compact candidate roster: {candidate_roster}. "
        "The raw tool results remain authoritative and the tool-call IDs are recovery handles. "
        "Stable source row ordinals and compact identity fields make cross-row references resolvable. "
        "For every record, distinguish inferred labels from explicit source identity assertions and "
        "unambiguous record-specific lifecycle statements in authoritative source fields. Vague "
        "similarity cannot invent a duplicate or spam classification. An explicit assertion of "
        "duplicate registration, same person, alternate identity, or a reference to another source "
        "row requires one canonical-versus-alternate disposition and at most one base mutation for "
        "that identity unless governing policy says otherwise. Keep this source identity decision "
        "separate from destination-object duplicate detection. Explicit withdrawal, cancellation, "
        "opt-out, hold, or completion is operative evidence for the operations it expressly permits "
        "or forbids. Evaluate operational "
        "communications by provenance, directive, scope, effective time, and supersession. "
        "Before mutation, make a compact operation-effect matrix for every applicable rule and "
        "record: predicate; rule type; authoritative evidence; explicit consequence; and separate "
        "effects on the base destination mutation, recipient delivery, approval or routing, "
        "notification, and escalation. A rule about actor, recipient, approval, or delivery route "
        "does not cancel the base artifact unless it expressly forbids it; email or escalation cannot "
        "satisfy a requested calendar, record, or status mutation. Mark every operation required, "
        "forbidden, blocked, or not applicable, with expected artifact counts. Do not infer actions "
        "from this compact index."
    )
    authoritative_evidence_card_limit: int = 3
    authoritative_evidence_source_limit: int = 240
    authoritative_evidence_snippet_limit: int = 600
    authoritative_evidence_checkpoint_limit: int = 4
    authoritative_evidence_message_limit: int = 3600
    authoritative_evidence_preflight_limit: int = 1400
    authoritative_evidence_sources: tuple[tuple[str, str], ...] = (
        ("gmail", r"^https://gmail\.googleapis\.com/gmail/.*/messages"),
        ("google_drive", r"^https://www\.googleapis\.com/drive/.*/files"),
        ("slack", r"^https://slack\.com/api/(?:search\.messages|conversations\.(?:history|replies))"),
    )
    authoritative_evidence_collection_keys: tuple[str, ...] = (
        "matches",
        "messages",
        "files",
        "threads",
        "items",
    )
    authoritative_evidence_reconciliation_message: str = (
        "Authoritative update reconciliation checkpoint. Pinned exact evidence: "
        "{authoritative_evidence}. Raw tool-call IDs are recovery handles and newer raw results "
        "remain authoritative. Text format alone does not determine authority. Evaluate each explicit "
        "operational instruction by provenance and sender role, directive, scope and applicability, "
        "effective time, and supersession evidence. Before any dependent mutation or notification, "
        "give every pinned update an applied, not-applicable, superseded, or unresolved disposition; "
        "not-applicable or superseded requires a concrete conflicting rule or scope, applicability, "
        "timing, or supersession mismatch. Carry applicable transformations into record values, "
        "aggregates, and recipient-facing content."
    )
    multi_record_mutation_preflight_message: str = (
        "Multi-record mutation preflight withheld {proposed_count} proposed side effect(s); no "
        "side effect occurred. Compact candidate roster: {candidate_roster}. The raw tool results "
        "remain authoritative and the tool-call IDs are recovery handles. Available source systems: "
        "{available_services}. Pinned exact update evidence: {authoritative_evidence}. This request "
        "explicitly depends on freshness. Before resubmitting one complete action bundle, give every "
        "roster record an explicit lifecycle disposition, resolve every explicit source identity "
        "link to one canonical row plus alternate dispositions, and give every pinned update an applied, "
        "not-applicable, superseded, or unresolved disposition. Free text cannot invent a "
        "classification from vague similarity, but an authoritative field explicitly asserting "
        "duplicate registration, same person, alternate identity, or another source row requires "
        "one canonical base mutation unless governing policy says otherwise. This is separate from "
        "checking the destination for an existing object. An unambiguous authoritative record-specific "
        "withdrawal, cancellation, opt-out, hold, or completion is operative for the operations it "
        "explicitly governs. "
        "Evaluate instructions by provenance, directive, scope, timing, and supersession. "
        "Use targeted entity or record-identifier follow-up only where applicability or authority "
        "remains unresolved; do not exhaustively search every source system for every record. "
        "For every included record, distinguish source values, authoritative transformations, final "
        "operational values, and dependent aggregates. Build an expected-versus-proposed artifact "
        "count by operation for the base destination mutation, recipient delivery, approval or "
        "routing, notification, and escalation. A routing artifact cannot replace the base mutation "
        "unless policy expressly forbids that mutation. Compute proposed counts after canonical "
        "dispositions and derive notification totals from successful canonical mutations. Identify "
        "blockers and validate recipient content against record-specific conditions. Resubmit all "
        "required mutations together; this "
        "preflight is consumed and will not repeat."
    )
    spreadsheet_schema_cache_limit: int = 8
    spreadsheet_schema_header_limit: int = 24
    spreadsheet_schema_header_length_limit: int = 80
    spreadsheet_schema_message_length_limit: int = 2400
    spreadsheet_schema_reconciliation_message: str = (
        "Spreadsheet schema checkpoint. Observed positional schema: {schemas}. "
        "For every positional write, preserve the observed header order exactly and provide one "
        "value position per observed column. Do not replace, extend, or restate the header row "
        "unless an advertised schema operation explicitly authorizes it. If a required field is "
        "absent, use an advertised schema operation or leave it explicitly unresolved; never add "
        "an extra positional value that shifts later values into the wrong columns. After writing, "
        "read back the destination and compare each value by this same header-to-position mapping "
        "before claiming completion or generating a summary."
    )
    preparation_tool_names: frozenset[str] = frozenset({"base64_encode"})
    preparation_reconciliation_message: str = (
        "Execution checkpoint. Preparation returned {count} successful output(s), but prepared "
        "artifacts are not committed side effects. Compact candidate roster: {candidate_roster}. "
        "Compare each prepared artifact against the operation-specific contracts for base destination "
        "mutations, recipient delivery, approval or routing, notifications, and escalations. Detect "
        "and remove artifacts for alternate source identities or records whose lifecycle forbids them, "
        "after resolving every explicit same-person, duplicate-registration, alternate-identity, or "
        "source-row reference to one canonical base mutation. Keep that reconciliation separate from "
        "destination-object duplicate checks. Detect missing non-email "
        "calendar, record, or status mutations; email or escalation cannot substitute for those base "
        "artifacts unless policy expressly forbids them. Prepare only missing permitted artifacts, "
        "and verify outbound values and counts derive from successful canonical destination mutations. "
        "Re-prepare only deficient "
        "artifacts. Keep internal policy conditions out of recipient content unless applicable or "
        "explicitly required. "
        "Known mutation endpoints: {mutation_endpoints}. "
        "Validate internally; when every prepared artifact passes, the next response must emit "
        "{fetch_tool_name} mutation tool calls for every outstanding artifact contract. Do not "
        "respond with text-only progress, 'ready to send', or a final answer while prepared work "
        "remains uncommitted."
    )
    completion_reconciliation_retry_limit: int = 1
    mutation_evidence_card_limit: int = 12
    mutation_evidence_list_limit: int = 12
    mutation_evidence_value_length_limit: int = 180
    mutation_evidence_encoded_message_limit: int = 12000
    mutation_evidence_target_fields: tuple[tuple[str, str], ...] = (
        ("summary", "summary"),
        ("subject", "subject"),
        ("title", "title"),
        ("name", "name"),
        ("location", "location"),
        ("status", "status"),
        ("state", "state"),
        ("range", "range"),
        ("calendarid", "calendar_id"),
        ("spreadsheetid", "spreadsheet_id"),
        ("fileid", "file_id"),
        ("recordid", "record_id"),
        ("userid", "user_id"),
    )
    completion_evidence_message_limit: int = 5000
    completion_mutation_operation_limit: int = 8
    completion_reconciliation_message: str = (
        "Completion reconciliation: your proposed final response indicates unfinished work. "
        "Reconcile the original request and successful tool results now. Use tools for every "
        "pending permitted action, preserve forbidden or not-applicable actions, or give a concise "
        "explicit blocker if the remaining work genuinely cannot continue."
    )
    completion_side_effect_audit_message: str = (
        "Artifact completion audit: the original request and authoritative source results are already "
        "in this conversation. Compact successful artifact evidence: {successful_artifacts}. "
        "Relevant available mutation operations: {mutation_operations}. Enumerate every direct "
        "obligation and every obligation from an applicable authoritative source, including each "
        "artifact, operation, destination, recipient, attendee, and required target field. Match each "
        "obligation one-to-one to exact successful evidence. Requested facts show intent; confirmed "
        "facts and artifact IDs show what the tool result verified. Treat absent or mismatched "
        "confirmed recipients, attendees, or fields as unverified. Use ordinary tools "
        "for every missing permitted action, while preserving forbidden or not-applicable actions. "
        "Do not synthesize recipients or obligations that are not visible in the request or an "
        "applicable authoritative source. Do not claim any recipient, attendee, field, or action "
        "without matching successful evidence; report a concise explicit blocker only when required "
        "work genuinely cannot continue."
    )
    unfinished_completion_patterns: tuple[str, ...] = (
        r"\b(?:pending|unfinished|incomplete|unresolved)\b",
        r"\b(?:remaining|outstanding)\s+(?:work|actions?|tasks?|steps?|items?|records?|"
        r"invoices?|messages?|notifications?|updates?)\b",
        r"\b(?:work|actions?|tasks?|steps?|items?|records?|invoices?|messages?|notifications?|"
        r"updates?)\s+remain(?:s|ed)?\s+(?:pending|outstanding|unfinished|incomplete|unresolved)\b",
        r"(?:^|[.!?]\s+|\n+)(?:i(?:'m| am)\s+|we(?:'re| are)\s+)?"
        r"(?:preparing|creating|sending|notifying|updating|posting|submitting|uploading|"
        r"scheduling|processing|finishing|completing)\b",
        r"\b(?:next|then)\b.{0,100}\b(?:prepare|create|send|notify|update|post|submit|upload|"
        r"schedule|process|finish|complete)\b",
        r"\b(?:i|we)\s+(?:will|'ll|am going to|are going to)\s+(?:prepare|create|send|notify|"
        r"update|post|submit|upload|schedule|process|finish|complete)\b",
    )
    completed_completion_patterns: tuple[str, ...] = (
        r"\bno\s+(?:pending|remaining|outstanding|unfinished|incomplete|unresolved)\b",
        r"\ball\s+(?:pending|remaining|outstanding|required|requested)\b.{0,80}\b"
        r"(?:complete|completed|done|resolved|finished|created|sent|updated|posted|submitted)\b",
    )
    blocker_completion_patterns: tuple[str, ...] = (
        r"\b(?:cannot|can't|unable to|blocked by|blocker)\b",
        r"\bmissing\s+(?:required\s+)?(?:permission|permissions|credential|credentials|access)\b",
        r"\b(?:permission|credentials|access)\s+(?:is|are)\s+(?:missing|unavailable|denied)\b",
    )
