"""The catalog exposes both domain and per-task RELAI memory scopes."""

from __future__ import annotations

from automationbench_experiments.catalog import load_catalog


def test_each_catalog_task_has_a_unique_task_tag() -> None:
    tasks = load_catalog()["tasks"]
    task_tags = []
    for task in tasks:
        domain = task["metadata"]["domain"]
        unique_tag = f"task_{task['id'].replace('.', '_')}"
        assert task["tags"] == [domain, unique_tag]
        task_tags.append(unique_tag)
    assert len(tasks) == 800
    assert len(set(task_tags)) == len(tasks)
    assert sum("finance" in task["tags"] for task in tasks) == 100
    assert [task["id"] for task in tasks if "task_finance_invoice_email_extract" in task["tags"]] == [
        "finance.invoice_email_extract"
    ]
