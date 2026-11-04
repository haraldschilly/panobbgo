"""JSON-Lines protocol worker.

One request per line on stdin, one response per line on stdout.

Requests
--------
* ``{"cmd": "create", "kind": "MA-BBOB"|"BBOB", "instance": N, "dim": D, "fid": M (BBOB only)}``
* ``{"cmd": "create", "kind": "MA-BBOB-official", "instance": ROW, "dim": D, "xopt": [...], "weights": [...], "iids": [...]}``
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

#: The problem id the competition's notebook sets on its instances.
OFFICIAL_PROBLEM_ID = 100


def _build_official(row: int, dim: int, xopt: list, weights: list, iids: list):
    """One row of the MA-BBOB competition's instance files, built the way its notebook builds it.

    The parent reads and verifies the files (``panobbgo/mabbob_official.py``)
    and sends the row; ``instance`` is the row index, not an ``ioh`` instance id.
    """
    if len(xopt) != dim:
        raise ValueError(f"xopt has {len(xopt)} coordinates, expected {dim}")
    problem = ioh.problem.ManyAffine(
        xopt=np.asarray(xopt, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        instances=np.asarray(iids, dtype=int),
        n_variables=dim,
    )
    problem.set_id(OFFICIAL_PROBLEM_ID)
    problem.set_instance(row)
    return problem


def _build_problem(
    kind: str,
    instance: int,
    dim: int,
    fid: int | None = None,
    xopt: list | None = None,
    weights: list | None = None,
    iids: list | None = None,
):
    """Instantiate an ``ioh.problem.RealSingleObjective`` by kind."""
    if kind == "MA-BBOB":
        return ioh.problem.ManyAffine(instance=instance, n_variables=dim)
    if kind == "MA-BBOB-official":
        if xopt is None or weights is None or iids is None:
            raise ValueError("MA-BBOB-official kind requires 'xopt', 'weights' and 'iids'")
        return _build_official(instance, dim, xopt, weights, iids)
    if kind == "BBOB":
        if fid is None:
            raise ValueError("BBOB kind requires a 'fid' (1..24)")
        # ``problem_class`` must be the enum member, not the string: the
        # current ioh binding rejects ``"BBOB"`` with
        # ``AttributeError: problem_class should be of type <enum 'ProblemClass'>``.
        return ioh.get_problem(int(fid), instance, dim, ioh.ProblemClass.BBOB)
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
                    xopt=req.get("xopt"),
                    weights=req.get("weights"),
                    iids=req.get("iids"),
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
