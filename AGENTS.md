# AGENTS.md

Entry point for agents (and humans) working on Panobbgo.  Keep it short;
details live in the linked files, history and designs under `planning/`.

## What this is

Panobbgo is Harald Schilly's 2012 thesis project, a Snobfit-inspired
optimizer for **expensive, noisy black-box problems**, revived in 2026 with
coding agents.  It is a **library**: a strategy, heuristics and analyzers
are composed in a Python script (`README.md`, `doc/source/guide_usage.rst`).
Evaluation is threaded and local by default; Dask is an optional extra.
`sketchpad/` holds scratch scripts, not curated demos.

The idea to keep in front of every design: **every evaluated point is
shared by all solvers** through one archive.  That shared archive is
Panobbgo's structural advantage over a plain portfolio of solvers.

Domain rules:

*   `max_eval` is a hard cap, not a target; in-flight evaluations count
    against it (phased strategies too).
*   Generating candidates is cheap, evaluating them is expensive: propose
    freely, submit carefully.
*   Restartability matters: the storage backend checkpoints and resumes,
    because one evaluation can take minutes.

## Read next

| File | What for |
|------|----------|
| [`planning/GOAL.md`](planning/GOAL.md) | Goal contract, state snapshot, plan of record — read first for any optimization-quality task |
| [`TODO.md`](TODO.md) | Open work and open decisions |
| [`doc/dev/process.md`](doc/dev/process.md) | Roles, PRs, reviews, merging, git rules, record keeping |
| [`doc/dev/environment.md`](doc/dev/environment.md) | Commands, CI, dependency policy, scripts and local runs |
| [`doc/dev/benchmarking.md`](doc/dev/benchmarking.md) | Metrics, evidence a PR needs, harness and AOCC tracks, re-baselining |
| `planning/DISCOVERY_2026-09-09.md` | Research log (§n): every measured result with its numbers |
| `planning/DESIGN_roadmap_2026-09-26.md` | Current roadmap |
| `doc/source/guide_benchmarking.rst` | User-facing benchmarking guide |
| `tools/ioh_worker/README.md` | IOH / MA-BBOB worker setup |

The nightly self-improvement loop was removed on 2026-09-25; its design and
ledgers are in `planning/done/`, its history in
`planning/SELF_IMPROVEMENT_LOG.md`.

## Rules that always apply

*   `uv run ...` for every tool; never `.venv/bin/...` or bare `python`.
*   Every change goes through a PR; merge only on green CI and after review
    (`doc/dev/process.md`).  Never use `git stash`.
*   PEP 8; `ruff` (line length 120) formats and lints.  Public functions and
    classes get Google-style docstrings.
*   New code is tested (`tests/`).  Stochastic tests that may fail
    intermittently get `@pytest.mark.flaky(retries=3)`.
*   Update the copyright banner on files you edit: `2012-<current year>`.
*   The composite-score formula and the default randomized battery are
    frozen contracts (`doc/dev/benchmarking.md`).
*   Update `TODO.md` when you fix a bug, find an issue, finish a task or
    make a decision; log measured results in the DISCOVERY log.
