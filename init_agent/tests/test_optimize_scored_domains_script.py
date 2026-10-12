"""Verify six-domain task selection and RELAI argument forwarding."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_first_ten_per_scored_domain_and_passthrough(tmp_path: Path) -> None:
    fake_relai = tmp_path / "relai"
    fake_relai.write_text("#!/bin/sh\nprintf 'ARG:%s\\n' \"$@\"\nexit 23\n")
    fake_relai.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}"}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/optimize_first10_scored_domains.py"),
         "--total-rollouts", "60", "--skip-final-eval", "--no-pr"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 23
    catalog = json.loads((ROOT / ".relai/custom-adaptor/automationbench/catalog.json").read_text())
    domains: dict[str, list[str]] = {}
    for task in catalog["tasks"]:
        if task["metadata"]["scored_domain"]:
            domains.setdefault(task["metadata"]["domain"], []).append(task["id"])
    assert len(domains) == 6
    assert "simple" not in domains
    selected = [(domain, task_id) for domain, ids in domains.items() for task_id in ids[:10]]
    lines = proc.stdout.splitlines()
    assert lines[0] == "Optimizing against 60 tasks (10 per scored domain):"
    assert lines[1:61] == [f"  {domain}: {task_id}" for domain, task_id in selected]
    expected_args = ["optimize"]
    for _, task_id in selected:
        expected_args.extend(("--custom-tests", f"automationbench::{task_id}"))
    expected_args.extend(("--total-rollouts", "60", "--skip-final-eval", "--no-pr"))
    assert lines[61:] == [f"ARG:{arg}" for arg in expected_args]
