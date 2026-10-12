"""Generate RELAI task context from the pinned AutomationBench public dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from automationbench.domains import get_available_domains, get_domain_dataset
from automationbench.runner import strip_none_values

BENCHMARK_COMMIT = "6d210543b7a046f0f451c828cd2dadef774276eb"
ADAPTER_NAME = "automationbench"
CATALOG_RELATIVE_PATH = Path(".relai/custom-adaptor/automationbench/catalog.json")


def catalog_path(project_root: Path | None = None) -> Path:
    return (project_root or Path.cwd()) / CATALOG_RELATIVE_PATH


def task_info(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("info") or {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"invalid task info for {row.get('task')}")
    return strip_none_values(value)


def task_name(row: dict[str, Any]) -> str:
    value = task_info(row).get("task_name") or row.get("task")
    if isinstance(value, dict):
        value = task_info(row).get("env_id") or value.get("env_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid AutomationBench task ID: {value!r}")
    return value


def _prompt_text(row: dict[str, Any]) -> str:
    messages = row.get("prompt") or []
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        content = message.get("content")
        parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
    return "\n\n".join(parts)


def build_catalog() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for domain in get_available_domains():
        for row in get_domain_dataset(domain):
            row = dict(row)
            identifier = task_name(row)
            if identifier in seen:
                raise ValueError(f"duplicate task ID: {identifier}")
            if not identifier.startswith(domain + "."):
                raise ValueError(f"task {identifier} is in unexpected domain {domain}")
            seen.add(identifier)
            assertions = task_info(row).get("assertions") or []
            if not assertions:
                raise ValueError(f"task {identifier} has no assertions")
            evaluators = [
                {
                    "id": f"assertion-{index:03d}",
                    "description": json.dumps(assertion, ensure_ascii=False, sort_keys=True),
                }
                for index, assertion in enumerate(assertions, 1)
            ]
            entries.append({
                "id": identifier,
                "tags": [domain, f"task_{identifier.replace('.', '_')}"],
                "description": f"AutomationBench {domain} task; native assertion grading",
                "input_summary": _prompt_text(row),
                "evaluators": evaluators,
                "metadata": {
                    "domain": domain,
                    "example_id": row.get("example_id"),
                    "scored_domain": domain != "simple",
                    "benchmark_commit": BENCHMARK_COMMIT,
                },
            })
    return {
        "schema_version": "relai.custom_adapter.v1",
        "runner": ["uv", "run", "--no-sync", "abexp", "run-task"],
        "prepare": ["uv", "sync"],
        "tasks": entries,
    }


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or catalog_path()).read_text())


def write_catalog(path: Path | None = None) -> dict[str, Any]:
    output = path or catalog_path()
    result = build_catalog()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
