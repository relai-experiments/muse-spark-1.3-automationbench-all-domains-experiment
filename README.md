# Harness optimization on Muse Spark 1.3, across six domains

This repository documents a single experiment: taking one model, leaving it completely untouched, and rewriting only the **agent harness** around it — then measuring what changed across 600 tasks.

Two things define the setup, and both differ from a default evaluation:

- **The model is [Muse Spark 1.3](https://research.meta.ai/blog/introducing-muse-spark-1-3) at `xhigh` reasoning effort** — Meta's frontier reasoning model, run at its highest reasoning setting in both conditions.
- **The benchmark is [AutomationBench 1.0.6](https://github.com/zapier/AutomationBench) in full** — all six non-simple domains (sales, marketing, operations, support, finance, HR) at 100 tasks each, **600 tasks in total**. Only the `simple` domain is excluded.

The harness was optimized by [RELAI](https://relai.ai) using only 60 of those tasks; the other 540 were held out.

**Quality went up and cost went down.** Across the full 600-task suite, pass rate rose from **40.7% to 48.2%** while total inference cost fell from **$134.11 to $92.33**. The price of a successfully completed task dropped **42%**.

| Scope | Harness | Pass rate | Average score | Total cost | Cost per success |
| --- | --- | --- | --- | --- | --- |
| 60 learning tasks | default | 31.7% | 0.687 | $15.74 | $0.829 |
| 60 learning tasks | **optimized** | **53.3%** | **0.822** | **$10.00** | **$0.312** |
| All 600 | default | 40.7% | 0.784 | $134.11 | $0.550 |
| All 600 | **optimized** | **48.2%** | **0.813** | **$92.33** | **$0.319** |
| Held-out 540 | default | 41.7% | 0.795 | $118.37 | $0.526 |
| Held-out 540 | **optimized** | **47.6%** | **0.812** | **$82.33** | **$0.320** |

The gain holds on tasks the optimizer never saw: on the held-out 540, pass rate rises 5.9 points and cost falls 30%.

---

## Setup

### The model

**Muse Spark 1.3** (`meta/muse-spark-1.3`), Meta's frontier reasoning model, accessed through **OpenRouter** with unpinned provider routing and **`reasoning_effort: xhigh`** — its highest reasoning setting. The model, its weights, and the routing and reasoning configuration are identical between the two runs — the only thing that differs is the harness.

### The benchmark

**AutomationBench 1.0.6** (commit `6d21054`), run in full: all six non-simple domains — sales, marketing, operations, support, finance and HR — at **100 tasks each, 600 in total**. AutomationBench also ships a `simple` domain, which is excluded here. Tasks run over simulated business services: email, spreadsheets, chat, CRMs, project trackers and accounting systems. A task gives the agent a natural-language instruction and scores the resulting end state against a list of assertions.

Two metrics are used throughout:

- **Pass rate** — the share of tasks where *every* scored assertion passed. All-or-nothing.
- **Average score** — the mean fraction of a task's assertions that passed. Gives partial credit.

Both come straight from the eval artifacts, as `task_completed_correctly` and `partial_credit`.

### The agent

The agent runs a tool-calling loop until it finishes or reaches a ceiling of **50 turns**. It has three tools in both conditions:

| Tool | Purpose |
| --- | --- |
| `api_search` | Discover available API endpoints |
| `api_fetch` | Call an endpoint |
| `base64_encode` | Encode a value |

### What "harness" means here

The harness is everything around the model: the system prompt, the scaffolding that drives the conversation turn by turn, and the tool layer the agent calls. Optimization may change any of it.

It does **not** change the model, its weights, the task definitions, or the assertions that score them. Those are identical across both runs.

### The optimization protocol

RELAI optimized the harness from the agent's own experience on **60 of the 600 tasks** — the first ten of each domain — over **600 optimizer rollouts**. The resulting harness was then frozen and evaluated unchanged on all 600 tasks. The remaining **540 tasks were never used during optimization** and are reported separately as "held-out 540".

Both runs completed all 600 tasks with no run errors.

### How cost is estimated

Costs are computed from the token counts each run recorded and the published per-token price, not from a billed invoice. The rates are OpenRouter's listed pricing for `meta/muse-spark-1.3`, verified 2026-10-01:

| | USD per 1M tokens |
| --- | --- |
| Uncached input | $1.25 |
| Cached input | $0.15 |
| Output | $4.25 |

`analyze.py` re-derives both runs' total cost from these rates and the recorded token counts; both reproduce the recorded totals exactly.

Because routing was unpinned, the provider OpenRouter actually selected may have charged differently and reported cached tokens differently.

---

## Results by domain

Each domain contributes 100 tasks. Results vary by domain: four improve on pass rate, one is unchanged, and one declines.

| Domain | Pass rate | Average score |
| --- | --- | --- |
| sales | 39% → **48%** | 0.710 → **0.768** |
| marketing | 47% → **58%** | 0.861 → **0.868** |
| operations | 47% → 47% | 0.813 → 0.780 |
| support | 31% → 26% | 0.844 → 0.810 |
| finance | 37% → **55%** | 0.720 → **0.833** |
| hr | 43% → **55%** | 0.754 → **0.819** |

At the task level across all 600, **97 tasks moved from failing to passing and 52 moved the other way**, a net gain of 45. Scores rose on 155 tasks and fell on 128. On the held-out 540 the counts are 82 and 50.

---

## How the agent's behaviour changed

Derived from the full step traces of all 1,200 task runs:

| Metric (600 tasks each) | Default | Optimized |
| --- | --- | --- |
| `api_search` calls | 9,690 | **6,173** |
| `api_fetch` calls | 14,007 | 12,826 |
| `base64_encode` calls | 985 | 941 |
| Agent turns per task (median) | 22 | **16** |
| Tool steps per task (median) | 36 | 31 |
| Tasks reaching the 50-turn ceiling | 39 | **3** |

The optimized agent searches about a third less, finishes in fewer turns, and almost never runs out of budget.

The token split moved in two directions at once. **Input fell 41%** — 305.4M to 180.4M, with the cache hit rate rising from 85.5% to 95.5% — while **output rose 42%**, from 9.3M to 13.3M. The agent carries less context and spends more of its budget reasoning. Since input tokens dominate the bill at this cache rate, the net effect is the 31% cost reduction.

Among failed tasks, the default harness satisfied a median 73% of assertions, with 119 of its 356 failures missing by exactly one. The optimized harness fails 311 times, at a median 70%, with 90 missing by one.

---

## What the optimizer changed

The two harnesses differ by **3,931 inserted lines across 10 files**. The changes fall into three groups.

**1. The system prompt was rewritten, 85 words to 1,372.** The default prompt is five generic lines — execute the task, don't ask clarifying questions, favour parallel tool calls. The optimized prompt is an explicit operating procedure covering evidence gathering, candidate reconciliation, mutation preflight checks and completion auditing.

**2. The scaffold gained 66 new methods** in `agent.py` — completion reconciliation, mutation-evidence checkpoints, authoritative-evidence cards, candidate rosters, spreadsheet-schema reconciliation, and discovery-progression messages. These make the harness intervene during a rollout rather than only instruct at the start.

**3. Twenty-seven new numeric bounds** were added to the agent config. These are the most legible single piece of evidence in the diff, and they explain the 41% drop in input tokens directly — each one caps how much material can enter the context:

```python
allowed_service_inventory_limit: int = 24
discovery_endpoint_limit: int = 12
discovery_mutation_endpoint_limit: int = 12
discovery_schema_identity_limit: int = 160
budget_checkpoint_remaining_turns: int = 8
candidate_roster_limit: int = 12
candidate_label_length_limit: int = 80
candidate_evidence_length_limit: int = 180
authoritative_evidence_card_limit: int = 3
authoritative_evidence_snippet_limit: int = 600
authoritative_evidence_message_limit: int = 3600
```

**What did not change:** the agent's three tools are the same in both conditions — the optimizer added no new agent-facing tool. The task definitions and the scoring assertions are untouched.

---

## Seeing the changes for yourself

Both harnesses are checked in, so the whole intervention is inspectable without running anything.

### The complete change set

```bash
diff -rq init_agent optimized_agent
```

returns exactly ten files:

```
Files init_agent/src/automationbench_experiments/agent.py and optimized_agent/src/automationbench_experiments/agent.py differ
Files init_agent/src/automationbench_experiments/config.py and optimized_agent/src/automationbench_experiments/config.py differ
Files init_agent/tests/test_agent_policy.py and optimized_agent/tests/test_agent_policy.py differ
Files init_agent/vendor/automation-bench/automationbench/tools/api/fetch.py and optimized_agent/vendor/automation-bench/automationbench/tools/api/fetch.py differ
Files init_agent/vendor/automation-bench/automationbench/tools/api/schemas/asana.jsonc and optimized_agent/vendor/automation-bench/automationbench/tools/api/schemas/asana.jsonc differ
Files init_agent/vendor/automation-bench/automationbench/tools/api/schemas/confluence.jsonc and optimized_agent/vendor/automation-bench/automationbench/tools/api/schemas/confluence.jsonc differ
Files init_agent/vendor/automation-bench/tests/test_api_fetch.py and optimized_agent/vendor/automation-bench/tests/test_api_fetch.py differ
Files init_agent/vendor/automation-bench/tests/test_api_routes.py and optimized_agent/vendor/automation-bench/tests/test_api_routes.py differ
Files init_agent/vendor/automation-bench/tests/test_api_search.py and optimized_agent/vendor/automation-bench/tests/test_api_search.py differ
Files init_agent/vendor/automation-bench/tests/tools/test_google_drive.py and optimized_agent/vendor/automation-bench/tests/tools/test_google_drive.py differ
```

Every other file in the two directories is byte-identical.

### What each file contributes

| File | Change | What it is |
| --- | --- | --- |
| `src/.../agent.py` | +1,602 / −10 | The scaffold: 66 new methods |
| `src/.../config.py` | +255 | The prompt rewrite and 27 numeric bounds |
| `tests/test_agent_policy.py` | +1,943 | Tests documenting the intended behaviour |
| `vendor/.../tools/api/fetch.py` | +8 / −1 | Stricter JSON-body validation |
| `vendor/.../schemas/asana.jsonc` | +1 / −1 | Schema correction |
| `vendor/.../schemas/confluence.jsonc` | +3 / −3 | Schema correction |
| `vendor/.../tests/*` (4 files) | +119 | Tests for the above |

### The three diffs worth reading

```bash
# The prompt rewrite and the numeric bounds — start here
diff -u init_agent/src/automationbench_experiments/config.py \
        optimized_agent/src/automationbench_experiments/config.py

# The scaffold: checkpoints, evidence cards, reconciliation passes
diff -u init_agent/src/automationbench_experiments/agent.py \
        optimized_agent/src/automationbench_experiments/agent.py

# The one behavioural change to the tool layer
diff -u init_agent/vendor/automation-bench/automationbench/tools/api/fetch.py \
        optimized_agent/vendor/automation-bench/automationbench/tools/api/fetch.py
```

---

## Repository layout

```
artifacts/
  init-all-six-xhigh/        default-harness evaluation, 600 tasks
  optimized-all-six-xhigh/   optimized-harness evaluation, 600 tasks
init_agent/                  the default harness code
optimized_agent/             the optimized harness code
analyze.py                   regenerates every number above from artifacts/
```

Each artifacts directory holds a `summary.json` plus one `task-NNNN.json` per task. The summary carries per-task score, pass flag and token usage; each task file carries the full step trace, the final output, and every assertion result.

To reproduce the numbers:

```bash
python3 analyze.py
```

It reads only `artifacts/` and needs nothing beyond the Python standard library.

---

## Limitations

**One benchmark, one model, one run per condition.** Every result is AutomationBench 1.0.6 (commit `6d21054`) with Muse Spark 1.3 at `xhigh` reasoning effort. Each condition was evaluated once, so run-to-run variance is not measured and is not separated from the effect of the harness.

**The optimized harness did not improve every domain.** It raised pass rate in four of six domains, left operations unchanged while lowering its average score, and lowered both for support. The per-domain table above reports this in full. A harness optimized against 60 tasks drawn evenly from all six domains does not generalize evenly across them.

**Task-level movement runs in both directions.** The net gain of 45 tasks is the balance of 97 improvements against 52 regressions. A reader comparing individual tasks will find cases where the default harness did better.

**The harness saw 60 of the 600 tasks.** Held-out numbers are reported separately throughout and are the ones to trust: 47.6% against the 53.3% achieved on the learning tasks.

**Costs are estimates.** They are computed from recorded token counts and published per-token prices, not billed invoices. Provider routing was unpinned, so actual charges and cache behaviour may differ.

**No ablation.** The optimized harness changes a prompt, adds scaffold logic, and introduces 27 bounds at once. We have not isolated how much of the gain each contributes.

**A note on the traces.** One tool call in the default run is recorded under the name `default.api-search` rather than `api_search` — 1 call out of 23,698. It is left in the artifacts as produced.

---

## Attribution

`init_agent/` and `optimized_agent/` contain code copied from [AutomationBench](https://github.com/zapier/AutomationBench), which is © 2026 Zapier, Inc. and MIT-licensed. Per-file copyright and SPDX headers are preserved intact. The harness modifications in `optimized_agent/` were produced by RELAI's agent optimizer.

Only the harness surface and the files the optimizer touched are vendored here. The full benchmark, including all task definitions, lives upstream.
