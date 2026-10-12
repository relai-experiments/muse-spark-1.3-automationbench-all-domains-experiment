"""Local catalog, one-task runner, and held-out evaluation commands."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from automationbench.console import console
from rich.table import Table

from automationbench_experiments.catalog import catalog_path, load_catalog, write_catalog
from automationbench_experiments.evaluate import evaluate_cohort
from automationbench_experiments.run import (
    DEFAULT_MODEL,
    default_effort,
    failure_result,
    run_task,
    write_result,
)


def _selection(args: argparse.Namespace, catalog: dict) -> list[str]:
    ids: list[str] = []
    if args.tasks:
        ids.extend(part.strip() for part in args.tasks.split(",") if part.strip())
    if args.tasks_file:
        path = Path(args.tasks_file)
        if path.suffix == ".json":
            value = json.loads(path.read_text())
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError("task JSON file must contain a list of task IDs")
            ids.extend(value)
        else:
            ids.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
    for domain in args.domain or []:
        ids.extend(item["id"] for item in catalog["tasks"] if item["metadata"]["domain"] == domain)
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError("select tasks with --tasks, --tasks-file, or --domain")
    known = {item["id"] for item in catalog["tasks"]}
    missing = [item for item in ids if item not in known]
    if missing:
        raise ValueError(f"unknown task IDs: {missing}")
    return ids


def _run_one(args: argparse.Namespace) -> int:
    try:
        result = asyncio.run(run_task(
            args.task_id,
            effort=args.effort,
            model=args.model,
            toolset=args.toolset,
            max_steps=args.max_steps,
        ))
    except Exception as error:
        result = failure_result(args.task_id, error)
        print(f"{args.task_id}: {type(error).__name__}: {error}", file=sys.stderr)
    write_result(Path(args.output), result)
    print(f"{args.task_id}: {result.get('report', {}).get('partial_credit', 'ERROR')}", file=sys.stderr)
    return 0  # RELAI reads structured failures from the result file.


def _evaluate(args: argparse.Namespace) -> int:
    if args.num_examples != -1 and args.num_examples < 1:
        raise ValueError("--num-examples must be positive or -1 for all")
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be positive")
    catalog = load_catalog()
    ids = _selection(args, catalog)
    if args.num_examples != -1:
        ids = ids[:args.num_examples]
    output_dir = Path(args.output_dir)
    summary = asyncio.run(evaluate_cohort(
        ids, catalog=catalog, output_dir=output_dir, model=args.model,
        effort=args.effort, toolset=args.toolset, max_steps=args.max_steps,
        max_concurrent=args.max_concurrent,
    ))
    table = Table(title="Custom Muse agent · AutomationBench")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Tasks", str(summary["task_count"]))
    table.add_row("Completed / failed", f"{summary['completed_count']} / {summary['failed_count']}")
    for label, key in (("Mean partial credit", "mean_partial_credit"),
                       ("Strict pass rate", "strict_pass_rate")):
        value = summary[key]
        table.add_row(label, f"{value:.1%}" if value is not None else "N/A")
    table.add_row("Elapsed", f"{summary['duration_seconds']:.1f} s")
    usage = summary["usage"]
    if usage:
        table.add_row("Input tokens", f"{usage['input_tokens']:,}")
        table.add_row("  Uncached / cached", (
            f"{usage['uncached_input_tokens']:,} / {usage['cached_input_tokens']:,}"
        ))
        table.add_row("Output tokens", f"{usage['output_tokens']:,}")
        cost = usage["estimated_cost_usd"]
        table.add_row("Estimated cost", (
            f"${cost:.4f}" if cost is not None
            else f"N/A ({usage['priced_task_count']} of {summary['task_count']} tasks have usage)"
        ))
        table.add_row("Pricing", (
            f"{usage['pricing']['source']} ({usage['pricing']['verified_date']})"
        ))
    console.print(table)
    domain_table = Table(title="Scores by domain")
    domain_table.add_column("Domain")
    domain_table.add_column("Tasks", justify="right")
    domain_table.add_column("Completed / failed", justify="right")
    domain_table.add_column("Mean partial credit", justify="right")
    domain_table.add_column("Strict pass rate", justify="right")
    for domain, scores in summary["domains"].items():
        partial = scores["mean_partial_credit"]
        strict = scores["strict_pass_rate"]
        domain_table.add_row(
            domain,
            str(scores["task_count"]),
            f"{scores['completed_count']} / {scores['failed_count']}",
            f"{partial:.1%}" if partial is not None else "N/A",
            f"{strict:.1%}" if strict is not None else "N/A",
        )
    console.print(domain_table)
    if summary["run_error"]:
        console.print(f"[red]Evaluation error:[/red] {summary['run_error']}")
    console.print(f"Per-task traces and summary: {output_dir}")
    return 1 if summary["failed_count"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="AutomationBench Muse experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    catalog_parser = sub.add_parser("catalog", help="generate or check the full public task catalog")
    catalog_parser.add_argument("--check", action="store_true")
    run_parser = sub.add_parser("run-task", help="run one RELAI custom adapter task")
    run_parser.add_argument("--task-id", required=True)
    run_parser.add_argument("--output", required=True)
    eval_parser = sub.add_parser("evaluate", help="evaluate explicitly selected task IDs")
    eval_parser.add_argument("--tasks", help="comma-separated task IDs")
    eval_parser.add_argument("--tasks-file", help="newline-delimited or JSON list of task IDs")
    eval_parser.add_argument("--domain", action="append", choices=["sales", "marketing", "operations", "support", "finance", "hr", "simple"], help="domain to evaluate (repeatable)")
    eval_parser.add_argument("--num-examples", type=int, default=-1, help="number of selected tasks (-1 for all)")
    eval_parser.add_argument("--max-concurrent", type=int, default=10, help="maximum parallel tasks")
    eval_parser.add_argument("--output-dir", required=True)
    for target in (run_parser, eval_parser):
        target.add_argument("--model", default=DEFAULT_MODEL)
        target.add_argument("--effort", choices=["xhigh", "max"], default=default_effort())
        target.add_argument("--toolset", choices=["api", "zapier", "limited_zapier"], default="api")
        target.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()
    if args.command == "catalog":
        generated = write_catalog() if not args.check else None
        if args.check:
            from automationbench_experiments.catalog import build_catalog
            generated = build_catalog()
            if generated != load_catalog():
                raise SystemExit(f"catalog is stale: regenerate with abexp catalog ({catalog_path()})")
        print(f"{len(generated['tasks'])} tasks: {catalog_path()}")
        return
    if args.command == "run-task":
        raise SystemExit(_run_one(args))
    raise SystemExit(_evaluate(args))


if __name__ == "__main__":
    main()
