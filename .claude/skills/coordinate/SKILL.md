---
name: coordinate
description: Run panobbgo work as a coordinator — delegate implementation to worktree agents, review every PR with fresh reviewer agents, fix, verify, merge on green CI; also triage and cleanup rounds. Use for "fix these issues", "implement X", "triage", "clean up TODO/docs", or any multi-PR effort.
argument-hint: "<task, issue list, or 'triage' / 'cleanup'>"
---

# Coordinate

You are the coordinator. You plan, decide, review and merge; agents
implement. Rules of record are in `doc/dev/process.md` (roles, PRs, review
and merge, record keeping) and `doc/dev/benchmarking.md` (evidence). This
skill is the practice that goes with them. Don't restate those rules to
agents; point to them.

## 0. Machine and budget

- Harald's machine is shared with long computations. Hard rule: every
  local run is `nice -n 10 ionice -c3 <cmd>` and uses at most half the
  cores in total across all agents (8 of 16; `--jobs`, `pytest -n`, BLAS
  threads). Locally only targeted tests and one-off snippets; full suites
  go to CI. Put this rule verbatim into every subagent prompt.
- Measurements count objective evaluations, not wall time. Heavy runs go
  to GitHub runners: `gh workflow run rebaseline.yml ...`, or a new
  `workflow_dispatch` workflow. Never block the laptop.
- If tokens run low, run agents one at a time.

## 1. Frame the work

- **`triage`**:
  1. Spawn read-only search agents, one per angle: bugs, simplification
     and dedup, performance, memory, tests and docs.
  2. Merge their findings into ordered blocks T1…Tn: measurement integrity
     first, then correctness, runtime, performance, cleanup.
  3. Record the blocks in `TODO.md`.
  4. Work the blocks in order. Move to the next block only when the
     reviewers say the current one is done.
- **Feature or fix**: write down the design first (a `planning/DESIGN_*`
  section, or the PR body) when it involves a decision Harald should see.
- **Decisions**: make the conventional ones yourself and say so in one
  line. Ask Harald only about real forks (direction, what we optimize for,
  anything irreversible), and batch those questions.

## 2. Implement: one agent, one worktree, one PR

Spawn with `isolation: "worktree"`. Every prompt is self-contained and
includes:

- Scope and acceptance criteria: files, behaviour, tests.
- The constraints:
  - no full suite or benchmarks locally, only niced targeted tests;
  - never `git stash`;
  - rebase on origin/master before the PR;
  - which other agents touch nearby files.
- The commit and PR attribution lines from the session's system reminder.
- The flow: open the PR, `gh pr checks N --watch`, fix failures, **stop
  without merging**, report the PR number, the design as built, and
  anything that needs a decision.

Run independent PRs in parallel. When master moves, `SendMessage` the
running agents what landed and where moved text now lives.

## 3. Review: fresh agents, split by angle

For each PR, spawn at least two `general-purpose` reviewers in parallel:

- **Substance**: fidelity to the paper or definition, the realism of a
  simulation, fairness of a baseline, correctness of a metric. Ask
  pointed questions ("does X model a real async run with q workers?").
- **Integration**: every harness and evaluation path, persistence and
  pickling, CLI and config validation, CI, docs, dead code, test quality.

The reviewer prompt says:

- Read the diff with `gh pr diff N`.
- Run things in a scratch worktree under the scratchpad, and remove it
  afterwards. No stash, no push, niced snippets only.
- Reviewers and verifiers name scratch worktrees and branches uniquely (PR, role, a random suffix): two verifiers collided on a shared name.
- Report numbered findings, ranked by severity. Each gives file:line, a
  concrete failure scenario, a fix, and confirmed or plausible.

## 4. Fix, then verify

- Send **all** findings from both reviewers to the **original**
  implementer in one `SendMessage`, so it keeps its context. Attach your
  decisions on the open questions, and say "do NOT merge".
- When the fix commit is in, spawn a **verifier** on that commit. Ask it
  to:
  - confirm each item as fixed, not fixed, or regressed;
  - hunt for regressions (e.g. "bit-identical on problems without
    failures?");
  - **mutation-test**: break each fixed behaviour on purpose and check
    that a test fails. This reliably finds tests that cannot fail;
  - end with a verdict.
- Repeat until the verdict is "ready". Small non-blocking polish can go
  in together with the merge instruction.

## 5. Merge

- Merge only on green CI, on a branch rebased onto current master:
  `gh pr merge N --rebase --delete-branch`. Merge dependent PRs in a
  deliberate order and tell the others to rebase.
- Merging after a watch in the background: re-check the checks for the
  **current head** before merging. A push made after the watch started is
  not covered by it.
- Afterwards:
  - `git pull --ff-only`;
  - `git worktree prune` and remove finished agents' worktrees (locked
    ones free up when their agent ends);
  - delete stale branches;
  - watch master's CI, and fix flakes at their root; a failing test
    helper counts.

## 6. Cleanup round (`cleanup`)

Rules for TODO, docs and memory, applied after a batch of PRs lands:

- Deduplicate, resolve inconsistencies (the newest dated statement wins),
  remove done and outdated items, simplify rules, shorten text.
- `TODO.md` holds open work only. Measured results go to the DISCOVERY
  log. Process rules go to `doc/dev/`.
- Memory holds only machine- or person-specific facts. Project and process
  rules live in the repo.
- A conflict you can't resolve from dates or git history is a question for
  Harald, not a guess.

## 7. Report to Harald

- Keep it short.
- Give a status table (PR, content, state) whenever the state changes.
- Headline what reviewers found in plain words, especially where a bug
  would have distorted a measurement.
- Never report an agent's result before its notification arrives.
