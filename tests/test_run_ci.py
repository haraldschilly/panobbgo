# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""run_ci.py (``./test.sh``) runs workflow steps the way the GitHub runner does."""

from __future__ import annotations

from pathlib import Path

import yaml

import run_ci

WORKFLOW = """
env:
  PYTHON: "3.14"
jobs:
  second:
    needs: first
    steps:
      - name: Run it
        run: |
          echo second >> "$LOG"
  first:
    steps:
      - uses: actions/checkout@v6
      - name: Install dependencies
        run: exit 1
      - name: Multi-line block
        run: |
          mkdir -p sub
          cd sub
          pwd > where.txt
          echo summary >> $GITHUB_STEP_SUMMARY
          echo "$PYTHON" > py.txt
      - name: Advisory
        run: |
          false
        continue-on-error: true
      - name: After advisory
        run: echo first >> "$LOG"
  deploy:
    if: github.ref == 'refs/heads/master'
    steps:
      - run: echo never
"""


def _jobs():
    return run_ci.collect_jobs({"wf": yaml.safe_load(WORKFLOW)})


def test_extraction_order_skips_and_flags():
    jobs = _jobs()
    # `needs:` puts `first` before `second`; the `if:` deploy job is dropped.
    assert [j.name for j in jobs] == ["wf:first", "wf:second"]
    first = jobs[0]
    # setup steps are skipped; a multi-line block stays one script.
    assert [s.name for s in first.steps] == ["Multi-line block", "Advisory", "After advisory"]
    assert first.steps[0].script.count("\n") == 5
    assert first.steps[1].continue_on_error
    assert first.steps[0].env["PYTHON"] == "3.14"


def test_run_block_is_one_shell_and_advisory_failure_passes(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "log.txt"
    monkeypatch.setenv("LOG", str(log))
    assert run_ci.run_jobs(_jobs())
    # `cd sub` applied to the next line of the same block
    assert (tmp_path / "sub" / "where.txt").read_text().strip() == str(tmp_path / "sub")
    assert (tmp_path / "sub" / "py.txt").read_text().strip() == "3.14"
    assert log.read_text().split() == ["first", "second"]


def test_failing_step_fails_job_and_skips_dependents(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "log.txt"
    monkeypatch.setenv("LOG", str(log))
    jobs = _jobs()
    jobs[0].steps[1].continue_on_error = False
    assert not run_ci.run_jobs(jobs)
    assert not log.exists()  # neither the rest of `first` nor `second` ran


def test_repo_workflows_include_every_gate():
    names = {j.name for j in run_ci.collect_jobs(run_ci.load_workflow_configs())}
    for gate in ("test", "lint", "typecheck", "docs", "benchmark", "format"):
        assert f"tests:{gate}" in names
    assert "docs:deploy" not in names
