#!/usr/bin/env python3
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""
run_ci.py - replay the GitHub Actions workflows locally (``./test.sh``).

Parses every workflow in ``.github/workflows`` and runs the ``run`` steps of
its jobs the way the GitHub runner does:

* each step's whole ``run`` block is one ``bash -e`` script, so a ``cd`` in
  line 1 still applies to line 2 and a failing line fails the step;
* ``$GITHUB_STEP_SUMMARY`` / ``$GITHUB_OUTPUT`` / ``$GITHUB_ENV`` /
  ``$GITHUB_PATH`` point at ``/dev/null`` (a redirect to an unset variable is
  an "ambiguous redirect" error);
* ``continue-on-error: true`` on a step or job makes its failure advisory;
* jobs run in workflow file order, with ``needs:`` respected.

Steps that only make sense on a fresh runner (checkout, set up Python,
install uv, the cached ``uv sync``, ``uv add``, artifact upload) are skipped;
``./test.sh`` does the ``uv sync --extra dev`` once up front.  Jobs with a
job-level ``if:`` (the gh-pages deploy) are skipped, since the condition
refers to GitHub context that does not exist locally.

``python run_ci.py --dry-run`` prints the plan without running anything.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Step names (lower-cased substrings) that only prepare a fresh runner.
SKIP_STEP_NAMES = (
    "set up python",
    "install uv",
    "checkout",
    "upload coverage",
    "install dependencies",
    "setup uv path",
    "cache uv",
    # `uv add --dev sphinx` would rewrite pyproject.toml and uv.lock in the
    # working tree; sphinx is in the dev extra that test.sh syncs.
    "install sphinx",
)

# The runner's per-step files; locally nothing reads them.
GITHUB_FILE_VARS = ("GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT", "GITHUB_ENV", "GITHUB_PATH")


@dataclass
class Step:
    """One ``run`` step of a job."""

    name: str
    script: str
    continue_on_error: bool = False
    working_directory: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Job:
    """One workflow job, reduced to its locally runnable steps."""

    name: str  # "<workflow>:<job id>"
    steps: list[Step]
    needs: list[str] = field(default_factory=list)
    continue_on_error: bool = False


def _plain_env(mapping) -> dict[str, str]:
    """Env entries whose value is a literal (``${{ ... }}`` cannot be evaluated locally)."""
    out = {}
    for key, value in (mapping or {}).items():
        text = str(value)
        if "${{" not in text:
            out[str(key)] = text
    return out


def _truthy(value) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def load_workflow_configs(workflows_dir: Path = WORKFLOWS_DIR) -> dict[str, dict]:
    """Load all workflow files, keyed by file stem, in sorted file order."""
    if not workflows_dir.exists():
        print(f"Error: Workflows directory not found at {workflows_dir}")
        sys.exit(1)
    files = sorted(list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml")))
    if not files:
        print(f"Error: No YAML workflow files found in {workflows_dir}")
        sys.exit(1)
    configs = {}
    for path in files:
        with open(path) as f:
            configs[path.stem] = yaml.safe_load(f)
    return configs


def extract_jobs(workflow_name: str, config: dict) -> list[Job]:
    """The locally runnable jobs of one workflow, in file order."""
    wf_env = _plain_env(config.get("env"))
    jobs = []
    for job_id, job_cfg in (config.get("jobs") or {}).items():
        if "if" in job_cfg:
            print(f"   (skipping {workflow_name}:{job_id}: job-level `if:` needs GitHub context)")
            continue
        job_env = {**wf_env, **_plain_env(job_cfg.get("env"))}
        default_wd = ((job_cfg.get("defaults") or {}).get("run") or {}).get("working-directory")
        steps = []
        for i, step in enumerate(job_cfg.get("steps") or []):
            if "run" not in step:
                continue
            name = step.get("name") or f"step {i + 1}"
            if any(skip in name.lower() for skip in SKIP_STEP_NAMES):
                continue
            steps.append(
                Step(
                    name=name,
                    script=step["run"],
                    continue_on_error=_truthy(step.get("continue-on-error")),
                    working_directory=step.get("working-directory", default_wd),
                    env={**job_env, **_plain_env(step.get("env"))},
                )
            )
        if not steps:
            continue
        needs = job_cfg.get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        jobs.append(
            Job(
                name=f"{workflow_name}:{job_id}",
                steps=steps,
                needs=[f"{workflow_name}:{n}" for n in needs],
                continue_on_error=_truthy(job_cfg.get("continue-on-error")),
            )
        )
    return jobs


def order_jobs(jobs: list[Job]) -> list[Job]:
    """File order, except that a job comes after everything it ``needs``."""
    by_name = {j.name: j for j in jobs}
    done: set[str] = set()
    ordered: list[Job] = []

    def visit(job: Job, stack: tuple[str, ...] = ()) -> None:
        if job.name in done:
            return
        if job.name in stack:
            raise ValueError(f"cyclic `needs:` through {job.name}")
        for dep in job.needs:
            if dep in by_name:
                visit(by_name[dep], (*stack, job.name))
        done.add(job.name)
        ordered.append(job)

    for job in jobs:
        visit(job)
    return ordered


def step_command(step: Step) -> list[str]:
    """The argv that runs a step: its whole block in one ``bash -e``, as on the runner."""
    return ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step.script]


def step_env(step: Step) -> dict[str, str]:
    """The environment of a step: ours, the workflow's literal ``env:``, and /dev/null runner files."""
    env = {**os.environ, **step.env}
    for var in GITHUB_FILE_VARS:
        env[var] = os.devnull
    env.setdefault("CI", "true")
    return env


def run_step(step: Step, job_name: str) -> bool:
    """Run one step; True on success."""
    print(f"\n🔄 {job_name} - {step.name}:")
    for line in step.script.rstrip().splitlines():
        print(f"   $ {line}")
    print("-" * 50)
    result = subprocess.run(step_command(step), env=step_env(step), cwd=step.working_directory)
    if result.returncode != 0:
        print(f"❌ Step failed with exit code {result.returncode}")
        return False
    return True


def run_jobs(jobs: list[Job]) -> bool:
    """Run jobs in order; returns False iff a non-advisory job failed."""
    failed: set[str] = set()
    for job in jobs:
        blocked = [d for d in job.needs if d in failed]
        if blocked:
            print(f"\n⏭️  Skipping {job.name}: needs failed job(s) {', '.join(blocked)}")
            failed.add(job.name)
            continue
        print(f"\n🚀 Starting job: {job.name}")
        job_ok = True
        for step in job.steps:
            if run_step(step, job.name):
                continue
            if step.continue_on_error:
                print(f"⚠️  '{step.name}' failed but is continue-on-error (advisory)")
                continue
            job_ok = False
            break
        if job_ok:
            print(f"✅ Job '{job.name}' completed successfully")
        elif job.continue_on_error:
            print(f"⚠️  Job '{job.name}' failed but is continue-on-error (advisory)")
        else:
            failed.add(job.name)
            print(f"❌ Job '{job.name}' failed")
    return not failed


def collect_jobs(configs: dict[str, dict]) -> list[Job]:
    """All runnable jobs of all workflows, in run order."""
    jobs = []
    for workflow_name, config in configs.items():
        jobs.extend(extract_jobs(workflow_name, config))
    return order_jobs(jobs)


def print_plan(jobs: list[Job]) -> None:
    """Print what a real run would execute."""
    for job in jobs:
        flags = " (continue-on-error)" if job.continue_on_error else ""
        print(f"\n== {job.name}{flags}")
        for step in job.steps:
            flags = " (continue-on-error)" if step.continue_on_error else ""
            wd = f" [cwd {step.working_directory}]" if step.working_directory else ""
            print(f"-- {step.name}{flags}{wd}: {' '.join(step_command(step)[:-1])} <<'EOF'")
            print(step.script.rstrip())
            print("EOF")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the jobs and scripts, run nothing")
    parser.add_argument("--job", action="append", default=[], help="run only this job (e.g. tests:lint); repeatable")
    args = parser.parse_args(argv)

    print("🔍 Parsing CI workflow configurations...")
    jobs = collect_jobs(load_workflow_configs())
    if args.job:
        wanted = set(args.job)
        jobs = [j for j in jobs if j.name in wanted or j.name.split(":")[-1] in wanted]
    if not jobs:
        print("❌ No executable commands found in any workflow")
        sys.exit(1)

    print(f"📋 {len(jobs)} jobs: {', '.join(j.name for j in jobs)}")
    if args.dry_run:
        print_plan(jobs)
        return

    if run_jobs(jobs):
        print("\n🎉 All CI jobs completed successfully!")
        print("✅ Local CI equivalent passed!")
    else:
        print("\n💥 Some CI jobs failed!")
        print("❌ Local CI equivalent failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
