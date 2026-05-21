# panobbgo IOH worker

Stateful child process hosting one `ioh.problem.RealSingleObjective`
instance.  Spawned on demand by `panobbgo.lib.ioh_wrapper.IOHProblem`,
proxies `eval(x) -> fx` over JSON-Lines stdin/stdout.

## Why this is a separate uv project

The `ioh` PyPI package ships pybind11/C++ binary wheels for **cp311 and
cp312 only** (as of `ioh==0.3.22`, 2025-09-26).  Pinning the entire
panobbgo core to Python ≤3.12 just to install `ioh` blocks the main
package from moving to 3.13+.

This subdirectory is an isolated uv project with its own
`.python-version = 3.12` and its own lockfile.  The parent panobbgo venv
has no `ioh` dependency at all, so the core is free to use any modern
Python.

## First-time setup

```bash
cd tools/ioh_worker
uv sync
```

The `ioh` wheel downloads a manylinux/macOS/Windows binary directly — no
C++ compile, no memory pressure on the host.  (If your platform/python
combo isn't covered, uv will fall back to building the sdist with
pybind11, which can use ~1 GiB resident per `cc1plus`.  Cap parallelism
with `CMAKE_BUILD_PARALLEL_LEVEL=1` in that case.)

After setup, the panobbgo IOH tests can find the worker automatically.

## Protocol

One JSON request per line on stdin, one JSON response per line on stdout.

### Commands

| Command | Request | Response |
| --- | --- | --- |
| `create` | `{"cmd": "create", "kind": "MA-BBOB"\|"BBOB", "instance": N, "dim": D, "fid": M}` (`fid` for BBOB only) | `{"ok": true, "lb": [...], "ub": [...], "optimum_y": F, "problem_id": I, "instance": N, "name": "..."}` |
| `eval`   | `{"cmd": "eval", "x": [float, ...]}` | `{"ok": true, "fx": F}` |
| `reset`  | `{"cmd": "reset"}` | `{"ok": true}` |
| `shutdown` | `{"cmd": "shutdown"}` | `{"ok": true}` then worker exits 0 |

### Errors

Any failure (unknown command, missing `create`, ioh exception) returns
`{"ok": false, "error": "<type>: <message>"}` on the same line.  The
worker stays alive and continues to read further requests so the parent
can recover or send `shutdown`.

### Concurrency

The worker is single-threaded by design — `IOHProblem` on the parent
side serialises calls behind a `threading.Lock`.  IOH problem objects
are not thread-safe in C++, so this is the correct behaviour anyway.

## Discovery

`panobbgo.lib.ioh_wrapper.IOHProblem` finds the worker directory by
walking up from the importing module until it sees
`tools/ioh_worker/pyproject.toml`.  Override with the
`PANOBBGO_IOH_WORKER` environment variable (absolute path).

## Invocation

The parent spawns the worker as

```bash
uv run --project tools/ioh_worker python -m ioh_worker
```

so the worker always runs in its own pinned venv, independent of the
calling process's interpreter.
