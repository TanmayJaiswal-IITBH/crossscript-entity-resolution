"""Per-entity decision rules, swept on cached scores.

A single global threshold is the naive choice. The metric is macro-averaged and
precision-weighted, so what matters is the shape of the decision *per entity*:

    tau        global probability floor for the top candidate
    delta      accept a non-top candidate only within delta of the entity's best
    tau2       a separate, higher floor for the 2nd and later matches
    tau_single explicit singleton gate -- if the best candidate is below this,
               emit nothing at all
    kmax       hard cap on matches per entity

The exact F_0.5 is computed in closed form per entity from (tp, n_pred, n_true),
vectorised over the whole split, so a full grid is seconds rather than minutes.
"""
import os
import sys
import argparse
import itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK


def load(path):
    z = np.load(path, allow_pickle=True)
    gid = z["gid"]
    score = z["score"]
    label = z["label"]
    n_true = z["n_true"]
    n_q = int(z["n_q"])
    # sort by (entity, -score) once; every rule below is then a prefix decision
    order = np.lexsort((-score, gid))
    return gid[order], score[order], label[order], n_true, n_q, z["entity"]


def macro_f05(gid, score, label, n_true, n_q, tau, delta, tau2, tau_single, kmax):
    n = len(gid)
    if n == 0:
        return 0.0, 0.0, 0.0
    starts = np.searchsorted(gid, np.arange(n_q), side="left")
    ends = np.searchsorted(gid, np.arange(n_q), side="right")
    has = ends > starts
    best = np.zeros(n_q, dtype=np.float32)
    best[has] = score[starts[has]]

    rank = np.arange(n, dtype=np.int64) - starts[gid]
    bestg = best[gid]
    keep = score >= tau
    keep &= score >= (bestg - delta)
    keep &= (rank == 0) | (score >= tau2)
    keep &= rank < kmax
    keep &= bestg >= tau_single          # singleton gate kills the whole entity

    tp = np.bincount(gid[keep], weights=label[keep].astype(np.float64),
                     minlength=n_q)
    npred = np.bincount(gid[keep], minlength=n_q).astype(np.float64)
    nt = n_true.astype(np.float64)

    f = np.zeros(n_q, dtype=np.float64)
    single = nt == 0
    f[single & (npred == 0)] = 1.0
    ok = (~single) & (npred > 0) & (tp > 0)
    p = np.zeros(n_q)
    r = np.zeros(n_q)
    p[ok] = tp[ok] / npred[ok]
    r[ok] = tp[ok] / nt[ok]
    f[ok] = (1.25 * p[ok] * r[ok]) / (0.25 * p[ok] + r[ok])
    return f.mean(), npred.mean(), float((npred == 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default=os.path.join(WORK, "val_scored.npz"))
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    gid, score, label, n_true, n_q, _ent = load(args.scored)
    print("entities %d   candidates %d   true %d" % (n_q, len(gid), int(n_true.sum())))

    lo, hi = float(score.min()), float(score.max())
    print("score range %.4f .. %.4f" % (lo, hi))
    if args.coarse:
        taus = np.round(np.arange(0.10, 0.91, 0.10), 3)
        deltas = [1.0, 0.5, 0.3, 0.15]
        tau2s = [0.0, 0.3, 0.5, 0.7]
        singles = [0.0, 0.3, 0.5, 0.7]
        kmaxs = [12]
    else:
        taus = np.round(np.arange(0.20, 0.86, 0.04), 3)
        deltas = [1.0, 0.6, 0.45, 0.35, 0.25, 0.18, 0.12]
        tau2s = [0.0, 0.35, 0.45, 0.55, 0.65, 0.75]
        singles = [0.0, 0.35, 0.45, 0.55, 0.65]
        kmaxs = [6, 8, 12]

    results = []
    for tau, delta, tau2, tsing, kmax in itertools.product(
            taus, deltas, tau2s, singles, kmaxs):
        if tau2 and tau2 < tau:
            continue
        if tsing and tsing < tau:
            continue
        f, pq, emp = macro_f05(gid, score, label, n_true, n_q,
                               tau, delta, tau2, tsing, kmax)
        results.append((f, tau, delta, tau2, tsing, kmax, pq, emp))
    results.sort(reverse=True)
    print("\n%-9s %-6s %-7s %-6s %-8s %-5s %-8s %-8s"
          % ("F0.5", "tau", "delta", "tau2", "tsingle", "kmax", "pred/q", "empty"))
    for r in results[:args.top]:
        print("%-9.5f %-6.2f %-7.2f %-6.2f %-8.2f %-5d %-8.2f %-8.3f"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
    best = results[0]
    print("\nBEST F0.5 = %.5f  tau=%.2f delta=%.2f tau2=%.2f tau_single=%.2f kmax=%d"
          % (best[0], best[1], best[2], best[3], best[4], best[5]))
    np.save(os.path.join(WORK, "best_rule.npy"),
            np.array([best[1], best[2], best[3], best[4], best[5]], dtype=np.float64))

    # what each rule component is worth, relative to a plain global threshold
    f_plain = max(macro_f05(gid, score, label, n_true, n_q, t, 1.0, 0.0, 0.0, 99)[0]
                  for t in taus)
    print("\nablation:")
    print("  global threshold only          %.5f" % f_plain)
    print("  + full rule family             %.5f  (+%.5f)" % (best[0], best[0] - f_plain))


if __name__ == "__main__":
    main()
