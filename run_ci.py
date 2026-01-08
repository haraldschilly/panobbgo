#!/usr/bin/env python3
"""
run_ci.py - Dynamic CI runner that mirrors GitHub Actions workflow locally

This script parses the .github/workflows/tests.yml file and executes
the same commands that CI would run, adapting dynamically to workflow changes.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path


def load_workflow_config():
    """Load the GitHub Actions workflow configuration."""
    workflow_path = Path(".github/workflows/tests.yml")
    if not workflow_path.exists():
        print(f"Error: Workflow file not found at {workflow_path}")
        sys.exit(1)

    with open(workflow_path, 'r') as f:
        return yaml.safe_load(f)


def extract_run_commands(workflow_config):
    """Extract run commands from workflow jobs, excluding setup steps."""
    commands = {}

    for job_name, job_config in workflow_config.get('jobs', {}).items():
        job_commands = []

        # Get the steps for this job
        steps = job_config.get('steps', [])

        for step in steps:
            # Skip setup steps that aren't relevant locally
            step_name = step.get('name', '').lower()
            if any(skip in step_name for skip in [
                'set up python', 'install uv', 'checkout',
                'upload coverage', 'install dependencies'
            ]):
                continue

            # Extract run commands
            if 'run' in step:
                run_content = step['run']
                # Split multi-line run commands
                if '\n' in run_content:
                    # Handle multi-line commands (YAML block scalar)
                    job_commands.extend([cmd.strip() for cmd in run_content.split('\n') if cmd.strip()])
                else:
                    job_commands.append(run_content.strip())

        if job_commands:
            commands[job_name] = job_commands

    return commands


def run_command(command, job_name, step_name=None):
    """Run a single command and handle errors."""
    print(f"\n🔄 Running {job_name}" + (f" - {step_name}" if step_name else "") + ":")
    print(f"   $ {command}")
    print("-" * 50)

    try:
        result = subprocess.run(command, shell=True, check=True, text=True,
                              capture_output=False)  # Let output stream to console
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False


def run_ci_jobs(commands):
    """Run all CI jobs in appropriate order."""
    # Define job execution order (dependencies matter)
    job_order = ['lint', 'typecheck', 'format', 'test']

    failed_jobs = []

    for job_name in job_order:
        if job_name not in commands:
            print(f"⚠️  Job '{job_name}' not found in workflow, skipping")
            continue

        print(f"\n🚀 Starting job: {job_name}")
        job_commands = commands[job_name]

        job_failed = False
        for i, cmd in enumerate(job_commands):
            step_name = f"step {i+1}" if len(job_commands) > 1 else None
            if not run_command(cmd, job_name, step_name):
                job_failed = True
                break

        if job_failed:
            failed_jobs.append(job_name)
            print(f"❌ Job '{job_name}' failed")
        else:
            print(f"✅ Job '{job_name}' completed successfully")

    return len(failed_jobs) == 0


def main():
    """Main entry point."""
    print("🔍 Parsing CI workflow configuration...")

    try:
        workflow_config = load_workflow_config()
    except Exception as e:
        print(f"Error loading workflow config: {e}")
        sys.exit(1)

    commands = extract_run_commands(workflow_config)

    if not commands:
        print("❌ No executable commands found in workflow")
        sys.exit(1)

    print(f"📋 Found {len(commands)} jobs with commands:")
    for job_name, job_commands in commands.items():
        print(f"   • {job_name}: {len(job_commands)} commands")
    print()

    # Run the CI jobs
    success = run_ci_jobs(commands)

    if success:
        print("\n🎉 All CI jobs completed successfully!")
        print("✅ Local CI equivalent passed!")
    else:
        print("\n💥 Some CI jobs failed!")
        print("❌ Local CI equivalent failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()