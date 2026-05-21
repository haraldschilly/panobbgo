"""JSON-Lines protocol worker.

One request per line on stdin, one response per line on stdout.

Requests
--------
* ``{"cmd": "create", "kind": "MA-BBOB"|"BBOB", "instance": N, "dim": D, "fid": M (BBOB only)}``
* ``{"cmd": "eval", "x": [float, ...]}``
* ``{"cmd": "reset"}``
* ``{"cmd": "shutdown"}``

Responses
---------
* On success: ``{"ok": true, ...}`` (fields depend on the command).
* On error: ``{"ok": false, "error": "<type>: <message>"}``.

The worker holds at most one problem instance at a time; sending ``create``
twice replaces the previous one.
"""

from __future__ import annotations

import json
import sys

import ioh
import numpy as np


def _build_problem(kind: str, instance: int, dim: int, fid: int | None = None):
    """Instantiate an ``ioh.problem.RealSingleObjective`` by kind."""
    if kind == "MA-BBOB":
        return ioh.problem.ManyAffine(instance=instance, n_variables=dim)
    if kind == "BBOB":
        if fid is None:
            raise ValueError("BBOB kind requires a 'fid' (1..24)")
        return ioh.get_problem(int(fid), instance, dim, "BBOB")
    raise ValueError(f"Unknown problem kind {kind!r}")


def _meta_response(problem) -> dict:
    meta = problem.meta_data
    return {
        "ok": True,
        "lb": [float(v) for v in problem.bounds.lb],
        "ub": [float(v) for v in problem.bounds.ub],
        "optimum_y": float(problem.optimum.y),
        "problem_id": int(meta.problem_id),
        "instance": int(meta.instance),
        "name": str(meta.name),
    }


def _write(resp: dict) -> None:
    sys.stdout.write(json.dumps(resp))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> int:
    problem = None
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")

            if cmd == "create":
                problem = _build_problem(
                    kind=str(req["kind"]),
                    instance=int(req["instance"]),
                    dim=int(req["dim"]),
                    fid=req.get("fid"),
                )
                _write(_meta_response(problem))
            elif cmd == "eval":
                if problem is None:
                    raise RuntimeError("eval called before create")
                x = np.asarray(req["x"], dtype=np.float64)
                fx = float(problem(x))
                _write({"ok": True, "fx": fx})
            elif cmd == "reset":
                if problem is None:
                    raise RuntimeError("reset called before create")
                problem.reset()
                _write({"ok": True})
            elif cmd == "shutdown":
                _write({"ok": True})
                return 0
            else:
                raise ValueError(f"Unknown cmd {cmd!r}")
        except Exception as e:  # noqa: BLE001 — protocol error: send back, keep going
            _write({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
