"""Empirical landscape characterisation of the wide families (for DISCOVERY §68)."""
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from panobbgo.harness_families import make_wide_battery, make_families_battery

B = 5.0
dim = int(sys.argv[1]) if len(sys.argv) > 1 else 5
which = sys.argv[2] if len(sys.argv) > 2 else "wide"
inst = (make_wide_battery if which == "wide" else make_families_battery)(dims=(dim,))


def quad_resid(f, X, y):
    d = X.shape[1]
    cols = [np.ones(len(X))] + [X[:, i] for i in range(d)]
    cols += [X[:, i] * X[:, j] for i in range(d) for j in range(i, d)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    return float(np.sum(r * r) / np.sum((y - y.mean()) ** 2))


rows = {}
for name, p in inst:
    rng = np.random.default_rng(1)
    d = p.dim
    X = rng.uniform(-B, B, size=(max(1000, 40 * d * d), d))
    y = np.array([p.eval(x) for x in X])
    f = p.eval
    # quadratic model residual (f-scale), near the optimum (radius 1) and global
    Xl = np.clip(p.x_opt + rng.uniform(-1, 1, size=(400, d)), -B, B)
    yl = np.array([f(x) for x in Xl])
    qg = quad_resid(f, X, y)
    ql = quad_resid(f, Xl, yl)
    # fitness-distance correlation (Spearman)
    dist = np.linalg.norm(X - p.x_opt, axis=1)
    fdc = spearmanr(y, dist)[0]
    # interaction index: second differences over coordinate pairs, step 1
    inter = []
    for _ in range(200):
        x = rng.uniform(-B + 1, B - 1, size=d)
        i, j = rng.choice(d, 2, replace=False)
        hi = np.zeros(d); hi[i] = 1.0
        hj = np.zeros(d); hj[j] = 1.0
        a, b_, c, e = f(x + hi + hj), f(x + hi), f(x + hj), f(x)
        den = abs(b_ - e) + abs(c - e)
        if den > 0:
            inter.append(abs(a - b_ - c + e) / den)
    inter = float(np.median(inter)) if inter else float("nan")
    # neutrality: equal values under a small random step
    steps = X[:300] + rng.normal(scale=0.05, size=(300, d))
    neut = float(np.mean([f(np.clip(s, -B, B)) == v for s, v in zip(steps, y[:300])]))
    # centre: quantile of f(0) among uniform samples
    cq = float(np.mean(y < f(np.zeros(d))))
    # local search success: L-BFGS-B from random starts in the box
    succ = 0
    n_ls = 20
    for s in range(n_ls):
        x0 = rng.uniform(-B, B, size=d)
        r = minimize(f, x0, method="L-BFGS-B", bounds=[(-B, B)] * d, options={"maxiter": 2000})
        succ += (r.fun - p.f_opt) < 1e-4 * max(1.0, np.ptp(y) * 1e-4)
    rows.setdefault(p.family, []).append(
        (qg, ql, fdc, inter, neut, cq, succ / n_ls, np.linalg.norm(p.x_opt) / (B * np.sqrt(d)), np.ptp(y))
    )

print(f"d = {dim} ({which}), means over 3 instances")
print("| family | 1-R² quad (box) | 1-R² quad (r=1 at opt) | FDC | interaction | neutral | f(centre) quantile | local-search hits | ‖x_opt‖/(B√d) | f range |")
print("|---|---|---|---|---|---|---|---|---|---|")
for fam, rs in rows.items():
    m = np.mean(np.array(rs), axis=0)
    print(f"| {fam} | {m[0]:.2f} | {m[1]:.1e} | {m[2]:+.2f} | {m[3]:.2f} | {m[4]:.2f} | {m[5]:.2f} | {m[6]:.2f} | {m[7]:.2f} | {m[8]:.1e} |")
