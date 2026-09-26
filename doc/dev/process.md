# Development process

How work gets planned, reviewed and merged.  Several coding agents work on
this repository at once, so most rules here exist to keep them from
stepping on each other.

## Roles

*   **One coordinating session** plans, reads results, checks them and
    decides what to try next.  Implementation, benchmark runs and
    mechanical work go to subagents.
*   **Model choice** (Claude): Fable coordinates; Opus implements
    design- or judgement-heavy work and measurement passes that need
    interpretation; Sonnet takes simple, fully specified tasks.  A spec
    for Sonnet names the files, functions, tests and acceptance criteria,
    so no design decision is left to the agent.
*   Subagent prompts are self-contained: a fresh agent does not see the
    coordinator's context.
*   The coordinator checks every reported number against the raw result
    files before believing it.
*   Harald decides at phase boundaries and on the items under "Waiting for
    Harald" in `TODO.md`.

## Changes and pull requests

*   **Deduplicate first.**  `gh pr list --state open` (drafts included);
    if the idea is already in an open PR, finish that one.
*   **Every change goes through a PR**, one change per PR, on its own
    branch in its own git worktree.  Three individually positive changes
    are not evidence for their combination.
*   **Never `git stash`.**  The stash stack is shared by all worktrees of
    the repository, which has already caused a collision between agents.
    Park work in a WIP commit on your branch instead.
*   **The PR description** says what changed, what was measured and how,
    and what was not measured (evidence rules:
    [benchmarking.md](benchmarking.md#evidence-for-a-pr)).
*   **Before pushing**, run `ruff`, `pyright` and the tests of the modules
    you touched.  The full suite runs in CI.

## Review and merge

1.  The implementing agent opens the PR, waits for CI
    (`gh pr checks <N> --watch`) and stops.  It does not merge.  It may run
    its own reviewers before handing back.
2.  The coordinator runs one or more fresh reviewer agents on the diff.
    The findings get fixed on the same branch (commits `Review #N: ...`).
    Perfection is not the bar; the reviewers' real findings are.
3.  **Merge only on green CI**, on a branch rebased onto current master.
    Changes by different agents that pass alone can fail together (on
    2026-09-11 an invariant test and a Splitter change did, and master
    stayed red for hours).
4.  If a red master cannot be avoided, mark the failing cases
    `xfail(reason=...)` and land that first.

## Record keeping

*   `TODO.md` — open work only.  Remove an item when it is done.
*   `planning/DISCOVERY_2026-09-09.md` — every measured result, negative
    ones included, as a numbered section (§n).  Negatives that are not
    logged get retried.
*   `planning/GOAL.md` — update the state snapshot (§2) when it changes
    materially.
*   Designs go to `planning/DESIGN_<topic>_<date>.md`; finished plans and
    archives go to `planning/done/`.
