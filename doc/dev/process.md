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

## Autonomy (Harald, 2026-09-26)

Agents working on this project may, without asking:

*   create branches, git worktrees and pull requests;
*   in a larger pipeline of work, **merge their own PRs** — under the loop
    below.

The loop for every PR:

1.  **CI green.**
2.  **Review by a subagent.**  Spawn a fresh reviewer agent and tell it to
    check the PR (split by angle when the change is large: substance —
    fidelity, fairness, realism — and integration).
3.  **Fix** the raised issues on the same branch and amend the PR.
4.  **Check again**: a verifier agent confirms the fixes and looks for
    regressions (mutation-test: break a fixed behaviour, a test must fail).
5.  **Merge** only when CI is green and the remaining issues are minor;
    then continue with the next item.

**Stop and ask Harald** — and stop working on that line — when something
is impossible to resolve, raises a real concern, contradicts a standing
decision or another source, or you are otherwise stuck.  Do not guess
around it.

Decisions that are conventional or reversible, the agent makes itself and
states in the PR.  Direction changes, irreversible actions beyond merging
(deleting releases or tags, rewriting history) and anything that affects
Harald's machine or accounts are his.

## Review and merge

The mechanics of the loop above when a coordinator delegates:

1.  The implementing agent opens the PR, waits for CI
    (`gh pr checks <N> --watch`) and stops.  It does not merge unless the
    coordinator tells it to after the review loop.  It may run its own
    reviewers before handing back.
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
