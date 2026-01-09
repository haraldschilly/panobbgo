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


def load_workflow_configs():
    """Load all GitHub Actions workflow configurations."""
    workflows_dir = Path(".github/workflows")
    if not workflows_dir.exists():
        print(f"Error: Workflows directory not found at {workflows_dir}")
        sys.exit(1)

    configs = {}
    yaml_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))

    if not yaml_files:
        print(f"Error: No YAML workflow files found in {workflows_dir}")
        sys.exit(1)

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
                configs[yaml_file.stem] = config
                print(f"Loaded workflow: {yaml_file.name}")
        except Exception as e:
            print(f"Error loading {yaml_file}: {e}")
            continue

    return configs


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
                'upload coverage', 'install dependencies', 'setup uv path',
                'cache uv'
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
    # Define job execution order (dependencies matter) - extract base names
    job_order = ['lint', 'typecheck', 'format', 'test']

    failed_jobs = []

    # Create mapping from base job names to full job names
    job_mapping = {}
    for full_job_name in commands.keys():
        base_job_name = full_job_name.split(':')[-1]  # Get the part after the last colon
        job_mapping[base_job_name] = full_job_name

    for base_job_name in job_order:
        full_job_name = job_mapping.get(base_job_name)
        if full_job_name not in commands:
            print(f"⚠️  Job '{base_job_name}' not found in workflows, skipping")
            continue

        print(f"\n🚀 Starting job: {full_job_name}")
        job_commands = commands[full_job_name]

        job_failed = False
        for i, cmd in enumerate(job_commands):
            step_name = f"step {i+1}" if len(job_commands) > 1 else None
            if not run_command(cmd, full_job_name, step_name):
                job_failed = True
                break

        if job_failed:
            failed_jobs.append(full_job_name)
            print(f"❌ Job '{full_job_name}' failed")
        else:
            print(f"✅ Job '{full_job_name}' completed successfully")

    return len(failed_jobs) == 0


def main():
    """Main entry point."""
    print("🔍 Parsing CI workflow configurations...")

    try:
        workflow_configs = load_workflow_configs()
    except Exception as e:
        print(f"Error loading workflow configs: {e}")
        sys.exit(1)

    if not workflow_configs:
        print("❌ No workflow configurations loaded")
        sys.exit(1)

    all_commands = {}
    for workflow_name, config in workflow_configs.items():
        commands = extract_run_commands(config)
        if commands:
            print(f"📋 Workflow '{workflow_name}': {len(commands)} jobs")
            for job_name, job_commands in commands.items():
                full_job_name = f"{workflow_name}:{job_name}"
                all_commands[full_job_name] = job_commands

    if not all_commands:
        print("❌ No executable commands found in any workflow")
        sys.exit(1)

    print(f"📋 Total: {len(all_commands)} jobs with commands across {len(workflow_configs)} workflows:")
    for job_name, job_commands in all_commands.items():
        print(f"   • {job_name}: {len(job_commands)} commands")
    print()

    # Run the CI jobs
    success = run_ci_jobs(all_commands)

    if success:
        print("\n🎉 All CI jobs completed successfully!")
        print("✅ Local CI equivalent passed!")
    else:
        print("\n💥 Some CI jobs failed!")
        print("❌ Local CI equivalent failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()