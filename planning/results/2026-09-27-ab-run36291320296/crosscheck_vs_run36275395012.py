"""Are the specs whose code path did not change bit-identical between run 36275395012 (§58) and the §60 run?"""

import json

S = "/tmp/claude-1000/-home-hsy-p-panobbgo/fbff2f72-8a36-4fa7-81a4-fda34ca2f079/scratchpad/cmaab"
SPECS = (
    "RoundRobin_CMAES",
    "RoundRobin_CMAES_active",
    "RoundRobin_CMAES_resample",
    "RoundRobin_CMAES_resample_randstart_active",
    "Baseline_Optuna_CmaEs",
    "Baseline_pycma_BIPOP",
)


def index(d):
    out = {}
    for seed, res in zip(d["base_seeds"], d["results"]):
        for r in res["runs"]:
            out[(seed, r["strategy_name"], r.get("fid"), r["dim"], r["instance"])] = (r["aocc"], r["best_fx"], r["trace_fx"])
    return out


for f in ("ref_ioh_standard_cma_ab.json", "ref_ioh_bbob_cma_ab_b200.json", "ref_ioh_bbob_cma_ab_b500.json"):
    a = index(json.load(open(f"{S}/ref_full/{f}")))
    b = index(json.load(open(f"{S}/ref60/{f}")))
    for spec in SPECS:
        keys = [k for k in a if k[1] == spec]
        same = sum(1 for k in keys if k in b and a[k] == b[k])
        print(f"{f:34s} {spec:44s} {same}/{len(keys)} bit-identical to §58's run")
