"""Task selection supports repeatable domains and keeps single-domain behavior."""

from __future__ import annotations

from argparse import Namespace

from automationbench_experiments.catalog import load_catalog
from automationbench_experiments.cli import _selection


def test_repeated_domains_select_all_tasks_once_in_flag_order() -> None:
    catalog = load_catalog()
    args = Namespace(tasks=None, tasks_file=None, domain=["finance", "sales", "finance"])
    ids = _selection(args, catalog)
    assert len(ids) == 200
    assert ids[:100] == [
        task["id"] for task in catalog["tasks"] if task["metadata"]["domain"] == "finance"
    ]
    assert ids[100:] == [
        task["id"] for task in catalog["tasks"] if task["metadata"]["domain"] == "sales"
    ]


def test_six_scored_domains_select_600_tasks_without_simple() -> None:
    catalog = load_catalog()
    args = Namespace(
        tasks=None,
        tasks_file=None,
        domain=["sales", "marketing", "operations", "support", "finance", "hr"],
    )
    ids = _selection(args, catalog)
    assert len(ids) == 600
    assert all(not task_id.startswith("simple.") for task_id in ids)
