"""Verify the wrapper's selection and argument forwarding without running RELAI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_first_twenty_finance_tasks_and_passthrough(tmp_path: Path) -> None:
    fake_relai = tmp_path / "relai"
    fake_relai.write_text("#!/bin/sh\nprintf 'ARG:%s\\n' \"$@\"\nexit 23\n")
    fake_relai.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}"}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/optimize_finance_first20.py"),
         "--total-rollouts", "50", "--skip-final-eval", "--no-pr"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 23
    catalog = json.loads((ROOT / ".relai/custom-adaptor/automationbench/catalog.json").read_text())
    expected_ids = [
        task["id"] for task in catalog["tasks"]
        if task["metadata"]["domain"] == "finance"
    ][:20]
    assert len(expected_ids) == 20
    lines = proc.stdout.splitlines()
    assert lines[0] == "Optimizing against these 20 finance tasks:"
    assert lines[1:21] == [f"  {task_id}" for task_id in expected_ids]
    expected_args = ["optimize"]
    for task_id in expected_ids:
        expected_args.extend(("--custom-tests", f"automationbench::{task_id}"))
    expected_args.extend(("--total-rollouts", "50", "--skip-final-eval", "--no-pr"))
    assert lines[21:] == [f"ARG:{arg}" for arg in expected_args]
