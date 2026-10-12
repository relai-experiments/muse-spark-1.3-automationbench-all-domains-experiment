"""Regenerate every number quoted in README.md from the raw eval artifacts.

Run:  python3 analyze.py

Self-contained: reads only ./artifacts/ and needs nothing but the standard
library. Two runs are compared, both on AutomationBench commit 6d21054,
all six scored domains, 100 tasks each:

    artifacts/init-all-six-xhigh/        default harness
    artifacts/optimized-all-six-xhigh/   RELAI-optimized harness

Each run directory holds a summary.json plus one task-NNNN.json per task.
summary.json alone carries per-task score, pass flag and usage, so every
results and cost number here comes from the two summaries; the per-task
traces are read only for the behavioural table.

Three scopes are reported. The optimizer only ever saw the 60 learning
tasks; "held out" is the other 540, which is where the generalization
claim lives.
"""

import json
import statistics as st
from collections import Counter, OrderedDict
from pathlib import Path

ARTIFACTS = Path(__file__).parent / "artifacts"
DEFAULT = "init-all-six-xhigh"
OPTIMIZED = "optimized-all-six-xhigh"

# Published OpenRouter pricing for meta/muse-spark-1.3, verified 2026-10-01.
# USD per token. reproduce_cost() re-derives the recorded totals from these.
UNCACHED_INPUT_COST = 1.25e-6
CACHED_INPUT_COST = 0.15e-6
OUTPUT_COST = 4.25e-6

# The agent's turn ceiling, counted as agent-authored steps in the trace.
TURN_CEILING = 50

# The tasks the harness was optimized against: the first ten of each domain,
# in the order the benchmark lists them. Supplied with the experiment rather
# than derived from the artifacts, which do not mark the split.
LEARNING_TASKS = frozenset({
    "sales.multi_hop_lookup",
    "sales.negative_selection",
    "sales.recency_selection",
    "sales.priority_selection",
    "sales.format_ambiguity",
    "sales.implicit_rules",
    "sales.cross_reference_validation",
    "sales.soft_matching",
    "sales.weighted_priority",
    "sales.dependency_chain",
    "marketing.social_engagement_response",
    "marketing.lead_enrichment",
    "marketing.contact_data_cleanup",
    "marketing.ad_performance_review",
    "marketing.ad_platform_audit",
    "marketing.conversion_tracking",
    "marketing.campaign_launch_checklist",
    "marketing.influencer_outreach",
    "marketing.landing_page_alert",
    "marketing.referral_processing",
    "operations.asana_fire_drill",
    "operations.trello_basecamp_compliance",
    "operations.monday_email_update",
    "operations.jira_confluence_incident",
    "operations.drive_notion_archive",
    "operations.calendar_airtable_maintenance",
    "operations.pipefy_vendor_onboarding",
    "operations.asana_safety_walk_log",
    "operations.trello_vendor_hold_email",
    "operations.monday_slack_inventory",
    "support.zendesk_sf_case_sync",
    "support.helpscout_jira_bugs",
    "support.gorgias_order_lookup",
    "support.zoho_sf_enrichment",
    "support.zendesk_maintenance_notify",
    "support.intercom_demo_scheduling",
    "support.helpscout_weekly_report",
    "support.zendesk_hubspot_org_sync",
    "support.freshdesk_faq_drafts",
    "support.zoho_freshdesk_escalation",
    "finance.invoice_email_extract",
    "finance.expense_anomaly_detection",
    "finance.overdue_invoice_followup",
    "finance.weekly_expense_summary",
    "finance.timesheet_to_invoice",
    "finance.po_email_logging",
    "finance.slack_receipt_capture",
    "finance.invoice_reconciliation",
    "finance.vendor_payment_approval",
    "finance.monthend_journal_entries",
    "hr.offboarding_automation",
    "hr.training_compliance",
    "hr.performance_feedback_logging",
    "hr.job_posting_distribution",
    "hr.job_board_monitoring",
    "hr.interview_scheduling",
    "hr.salary_band_audit",
    "hr.pto_balance_alert",
    "hr.candidate_rejection_followup",
    "hr.probation_review_reminder",
})


def load_summary(run):
    with open(ARTIFACTS / run / "summary.json") as fh:
        return json.load(fh)


def by_task(summary):
    return {t["task_id"]: t for t in summary["tasks"]}


def domains_in_order(summary):
    """-> OrderedDict(domain -> [task_id, ...]) preserving benchmark order."""
    out = OrderedDict()
    for task_id in summary["selected_task_ids"]:
        out.setdefault(task_id.split(".")[0], []).append(task_id)
    return out


def scopes(summary):
    ids = summary["selected_task_ids"]
    return OrderedDict([
        ("60 learning", [i for i in ids if i in LEARNING_TASKS]),
        ("all 600", list(ids)),
        ("held-out 540", [i for i in ids if i not in LEARNING_TASKS]),
    ])


def aggregate(tasks):
    n = len(tasks)
    passed = sum(t["task_completed_correctly"] for t in tasks)
    cost = sum(t["usage"]["estimated_cost_usd"] for t in tasks)
    return {
        "n": n,
        "pass_rate": passed / n,
        "avg_score": sum(t["partial_credit"] for t in tasks) / n,
        "cost": cost,
        "cost_per_passed": cost / passed if passed else float("nan"),
        "input_tokens": sum(t["usage"]["input_tokens"] for t in tasks),
        "output_tokens": sum(t["usage"]["output_tokens"] for t in tasks),
    }


def tool_names(step):
    """-> [name, ...] for one trace step.

    Each entry of step["tool_calls"] is a JSON *string* holding
    {"id", "name", "arguments"}; decode before reading the name.
    """
    names = []
    for raw in step.get("tool_calls") or []:
        call = raw
        if isinstance(call, str):
            try:
                call = json.loads(call)
            except (ValueError, TypeError):
                continue
        if not isinstance(call, dict):
            continue
        name = call.get("name") or (call.get("function") or {}).get("name")
        if name:
            names.append(name)
    return names


def conduct(run, summary):
    """Walk every task trace: how the agent spends a task."""
    counts = Counter()
    agent_turns, tool_steps = [], []
    for entry in summary["tasks"]:
        with open(ARTIFACTS / run / entry["result_file"]) as fh:
            steps = json.load(fh)["trace"]["steps"]
        agent_turns.append(sum(1 for s in steps if s.get("source") == "agent"))
        tool_steps.append(sum(1 for s in steps if s.get("source") == "tool"))
        for step in steps:
            counts.update(tool_names(step))
    return {
        "tool_calls": counts,
        "turns_median": st.median(agent_turns),
        "tool_steps_median": st.median(tool_steps),
        "hit_ceiling": sum(1 for t in agent_turns if t >= TURN_CEILING),
    }


def failure_profile(run, summary):
    """Among failed tasks, how much of the work got done."""
    shares, by_one, failures = [], 0, 0
    for entry in summary["tasks"]:
        if entry["task_completed_correctly"] == 1:
            continue
        with open(ARTIFACTS / run / entry["result_file"]) as fh:
            report = json.load(fh)["report"]
        # Excluded assertions are not scored; leave them out of the ratio.
        results = [a for a in report["assertion_results"] if not a.get("excluded")]
        if not results:
            continue
        failures += 1
        passed = sum(1 for a in results if a["passed"])
        shares.append(passed / len(results))
        if len(results) - passed == 1:
            by_one += 1
    return {
        "failures": failures,
        "median_assertion_share": st.median(shares) if shares else float("nan"),
        "failed_by_one": by_one,
    }


def movement(before, after, ids):
    b, a = by_task(before), by_task(after)
    return {
        "fail_to_pass": sum(1 for i in ids
                            if not b[i]["task_completed_correctly"] and a[i]["task_completed_correctly"]),
        "pass_to_fail": sum(1 for i in ids
                            if b[i]["task_completed_correctly"] and not a[i]["task_completed_correctly"]),
        "score_up": sum(1 for i in ids if a[i]["partial_credit"] > b[i]["partial_credit"]),
        "score_down": sum(1 for i in ids if a[i]["partial_credit"] < b[i]["partial_credit"]),
    }


def reproduce_cost(summary):
    """Re-derive the recorded total from token counts and published rates."""
    usage = summary["usage"]
    return (usage["uncached_input_tokens"] * UNCACHED_INPUT_COST
            + usage["cached_input_tokens"] * CACHED_INPUT_COST
            + usage["output_tokens"] * OUTPUT_COST)


def main():
    runs = OrderedDict([("default", load_summary(DEFAULT)),
                        ("optimized", load_summary(OPTIMIZED))])
    default, optimized = runs["default"], runs["optimized"]
    tasks = {label: by_task(s) for label, s in runs.items()}

    print("=" * 78)
    print("RESULTS BY SCOPE")
    print("=" * 78)
    header = (f"{'scope':14} {'harness':10} {'n':>4} {'pass':>7} "
              f"{'avg score':>10} {'cost':>9} {'$/success':>10}")
    print(header)
    print("-" * len(header))
    for scope, ids in scopes(default).items():
        for label in runs:
            a = aggregate([tasks[label][i] for i in ids])
            print(f"{scope:14} {label:10} {a['n']:4} {a['pass_rate']:7.1%} "
                  f"{a['avg_score']:10.4f} {a['cost']:9.2f} {a['cost_per_passed']:10.3f}")
        print()

    full = {label: aggregate(list(tasks[label].values())) for label in runs}
    d, o = full["default"], full["optimized"]
    print(f"all 600: pass {(o['pass_rate']-d['pass_rate'])*100:+.1f} pp | "
          f"cost {(o['cost']/d['cost']-1):+.1%} | "
          f"cost per success {(o['cost_per_passed']/d['cost_per_passed']-1):+.1%}")
    print(f"tokens : input {(o['input_tokens']/d['input_tokens']-1):+.1%} | "
          f"output {(o['output_tokens']/d['output_tokens']-1):+.1%}")
    for label, s in runs.items():
        u = s["usage"]
        print(f"         {label:10} cache hit {u['cached_input_tokens']/u['input_tokens']:.1%}")

    print()
    print("=" * 78)
    print("BY DOMAIN")
    print("=" * 78)
    head = f"{'domain':12} {'pass':>16} {'Δpp':>6} {'avg score':>18} {'Δ':>8}"
    print(head)
    print("-" * len(head))
    for domain, ids in domains_in_order(default).items():
        a = aggregate([tasks["default"][i] for i in ids])
        b = aggregate([tasks["optimized"][i] for i in ids])
        print(f"{domain:12} {a['pass_rate']:7.0%} -> {b['pass_rate']:5.0%} "
              f"{(b['pass_rate']-a['pass_rate'])*100:+6.0f} "
              f"{a['avg_score']:8.3f} -> {b['avg_score']:6.3f} "
              f"{b['avg_score']-a['avg_score']:+8.3f}")

    print()
    print("=" * 78)
    print("TASK MOVEMENT")
    print("=" * 78)
    for scope, ids in scopes(default).items():
        if scope == "60 learning":
            continue
        m = movement(default, optimized, ids)
        print(f"  {scope:14} fail->pass {m['fail_to_pass']:4}  pass->fail {m['pass_to_fail']:4}"
              f"  score up {m['score_up']:4}  score down {m['score_down']:4}")
    print("  by domain (net pass-rate movement):")
    for domain, ids in domains_in_order(default).items():
        m = movement(default, optimized, ids)
        net = m["fail_to_pass"] - m["pass_to_fail"]
        print(f"    {domain:12} +{m['fail_to_pass']:<4} -{m['pass_to_fail']:<4} net {net:+d}")

    print()
    print("=" * 78)
    print("HOW THE AGENT SPENDS A TASK")
    print("=" * 78)
    cd = conduct(DEFAULT, default)
    co = conduct(OPTIMIZED, optimized)
    rows = [
        ("api_search calls", cd["tool_calls"]["api_search"], co["tool_calls"]["api_search"]),
        ("api_fetch calls", cd["tool_calls"]["api_fetch"], co["tool_calls"]["api_fetch"]),
        ("base64_encode calls", cd["tool_calls"]["base64_encode"], co["tool_calls"]["base64_encode"]),
        ("agent turns (median)", cd["turns_median"], co["turns_median"]),
        ("tool steps (median)", cd["tool_steps_median"], co["tool_steps_median"]),
        (f"tasks reaching {TURN_CEILING} turns", cd["hit_ceiling"], co["hit_ceiling"]),
    ]
    print(f"{'metric':32} {'default':>12} {'optimized':>12}")
    for name, a, b in rows:
        fmt = (lambda v: f"{v:,.0f}") if max(a, b) >= 100 else (lambda v: f"{v:g}")
        print(f"  {name:30} {fmt(a):>12} {fmt(b):>12}")
    extra = (set(cd["tool_calls"]) | set(co["tool_calls"])) - {
        "api_search", "api_fetch", "base64_encode"}
    for name in sorted(extra):
        print(f"  [other] {name:24} {cd['tool_calls'][name]:>12,} {co['tool_calls'][name]:>12,}")

    print()
    print("=" * 78)
    print("FAILURE PROFILE")
    print("=" * 78)
    for label, run in (("default", DEFAULT), ("optimized", OPTIMIZED)):
        f = failure_profile(run, runs[label])
        print(f"  {label:10} {f['failures']:4} failures | median assertions satisfied "
              f"{f['median_assertion_share']:.0%} | failed by exactly one: {f['failed_by_one']}")

    print()
    print("=" * 78)
    print("COST REPRODUCTION (published rates vs recorded)")
    print("=" * 78)
    for label, s in runs.items():
        est, rec = reproduce_cost(s), s["usage"]["estimated_cost_usd"]
        print(f"  {label:10} recomputed ${est:8.2f}   recorded ${rec:8.2f}   "
              f"delta {abs(est - rec) / rec:.3%}")


if __name__ == "__main__":
    main()
