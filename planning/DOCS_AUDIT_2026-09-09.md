# Documentation audit — 2026-09-09

Input for the Phase-2 consolidation (see `TODO.md` session section and
`planning/DISCOVERY_2026-09-09.md`). Goal: consistent, compact,
deduplicated docs; agent instructions separated from history.

## Inventory (approx. lines)

| File | Lines | Purpose | Audience |
|---|---|---|---|
| `README.md` | 132 | install / test / usage entry | user |
| `AGENTS.md` | 1463 | agent rules + harness + loop changelog | agent (mostly history) |
| `CLAUDE.md` | 6 | pointer to AGENTS.md | agent |
| `DEVELOPMENT_PROMPT.md` | 669 | onboarding prompt, stale counts | obsolete |
| `test_plan.md` | 3 | finished one-off note | delete |
| `TODO.md` | 434 | newest-first status log | maintainer |
| `doc/benchmark-functions-todo.md` | 62 | candidate benchmark functions | maintainer |
| `doc/source/guide_usage.rst` / `guide_setup.rst` | 1262 / 371 | install + config + examples (overlapping) | user |
| `doc/source/guide_benchmarking.rst` | 2761 | composite score (~580) + loop internals (~2180) | user, then agent |
| `doc/source/guide_{introduction,architecture,mathematical_foundation,extending,research}.rst` | 195–762 | user guide chapters | user |
| `doc/source/{heuristics,analyzers,strategies,...}.rst` | 5–38 | autodoc stubs + component lists | user |
| `planning/GOAL.md` | 238 | goal contract | agent |
| `planning/SELF_IMPROVEMENT_LOOP.md` | 732 | loop design §1–§12 | agent/maintainer |
| `planning/SELF_IMPROVEMENT_LOG.md` | 1211 | dated change log | history |
| `planning/NEXT.md` | 263 | three "missing" blocks — all shipped | obsolete |
| `planning/LOOP_DIAGNOSIS_2026-08-11.md` | 217 | 34-night audit | history |
| `planning/TEST_PERFORMANCE.md` | 128 | test timing notes | maintainer |
| `tools/ioh_worker/README.md` | 76 | worker protocol | maintainer |
| `.github/workflows/README.md` | 183 | CI caching; only covers tests.yml | maintainer |
| `benchmarks/`, `sketchpad/` | — | no README | — |

## Duplication clusters (canonical source in bold)

1. Self-improvement loop flags/internals: `AGENTS.md:284-1335`,
   **`guide_benchmarking.rst:583-2761`**, `SELF_IMPROVEMENT_LOOP.md` §7–§12,
   and a ~1500-word run-on paragraph at `guide.rst:44`.
2. Harness workflow (baseline → change → compare): `AGENTS.md:64-176`,
   `CLAUDE.md`, **`guide_benchmarking.rst:95-330`**, `GOAL.md:69-104`.
3. Installation: `README.md:15-57`, **`guide_usage.rst:6-60`**,
   `guide_setup.rst:18-40`, `AGENTS.md:25-36`, `DEVELOPMENT_PROMPT.md`.
4. IOH worker setup: `AGENTS.md:1336-1385` ≈ **`tools/ioh_worker/README.md`**.
5. Component inventories: **`doc/source/{heuristics,analyzers,strategies}.rst`**
   vs `DEVELOPMENT_PROMPT.md:56-92`, `planning/NEXT.md`.
6. Architecture framing: **`guide_architecture.rst`** vs `guide_introduction.rst`,
   `index.rst:87-114`, `AGENTS.md:44-63`, `DEVELOPMENT_PROMPT.md:15-43`.
7. Test/CI instructions: `README.md:59-84`, `AGENTS.md:1456-1463`,
   `.github/workflows/README.md`, `DEVELOPMENT_PROMPT.md:334-350`.

## Stale or contradictory statements (verified)

* `README.md:72` "All 27 tests should pass" — 2014 tests today.
* `AGENTS.md:38-42` Known Issues: `DataFrame.append` no longer used anywhere;
  "Result needs `__hash__`" and NumPy-2 matplotlib issue are stale.
* `AGENTS.md:9-10` "migrate tests to pytest / move to tests/" — done.
* Coverage: `AGENTS.md:13` 45 % vs `DEVELOPMENT_PROMPT.md:417` 58 % vs `:127` ">80 %".
* `AGENTS.md:53` "runs on Dask distributed" vs `README.md:90` threaded default.
* `AGENTS.md:284` "(in progress)" heading over all-shipped bullets.
* `AGENTS.md:1127` flake8 as linter; CI gate is ruff (`tests.yml:286`), flake8 is `--exit-zero`.
* `DEVELOPMENT_PROMPT.md`: "11 heuristics" (27 exist), "4 analyzers" (7), "2 strategies" (6),
  "143 tests", bandit strategies and convergence detection listed as future work.
* `planning/NEXT.md`: Sensitivity and Restart analyzers proposed as missing — both exist with tests.
* `doc/source/index.rst:164` stray `.lib` line; `:172-177` empty "User Interface" toctree.
* `doc/source/heuristics.rst` omits `RegionUCB`.
* `.github/workflows/README.md` ignores `docs.yml` and `self_improve_nightly.yml`; quoted CI timings predate the suite.
* `README.md:8` / `AGENTS.md:49` point to `sketchpad/` as demos; it is 6 scratch scripts without a README.

## AGENTS.md triage

* Keep (~180 lines): `:1-63` (after correcting Known Issues), `:64-176`
  harness workflow/modes/score, `:177-239` PR-evidence policy (the most
  valuable agent-only section), `:240-283` statistical rigor, `:1456-1463` CI.
* Move out (~1150 lines): `:284-1335` — dated "shipped YYYY-MM-DD" flag
  narrative inside bash comments → flag semantics to a loop reference,
  dated entries to `SELF_IMPROVEMENT_LOG.md`.
* Replace with link (~50 lines): `:1336-1385` IOH worker setup.

## Target structure

* **Keep**: `README.md` (fix counts, link to Sphinx), `AGENTS.md` (~200 lines),
  `CLAUDE.md`, user-guide chapters, autodoc stubs, `planning/GOAL.md`,
  `SELF_IMPROVEMENT_LOOP.md`, `SELF_IMPROVEMENT_LOG.md`, `TEST_PERFORMANCE.md`,
  `tools/ioh_worker/README.md`.
* **Merge**: `guide_setup.rst` → `guide_usage.rst`; `guide_benchmarking.rst`
  split into a ~600-line user chapter + `planning/LOOP_REFERENCE.md` for
  loop internals; `AGENTS.md:284-1335` → log + loop reference;
  `.github/workflows/README.md` → cover all three workflows;
  `doc/benchmark-functions-todo.md` → `GOAL.md` §5 backlog.
* **Move to `planning/done/`**: `DEVELOPMENT_PROMPT.md`, `planning/NEXT.md`,
  `planning/LOOP_DIAGNOSIS_2026-08-11.md`.
* **Delete**: `test_plan.md`, stray `index.rst` lines.
* **Add**: `benchmarks/README.md`; `sketchpad/README.md` or drop the "demos" claim.
