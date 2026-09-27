"""Step 4: compare models x post-processing variants on the out-of-fold scores.

Models:     every oof_<kind>.npy present, plus score-averaged blends.
Variants:   none       the plain per-entity rule
            drop_all   a record claimed by >1 entity is dropped from all of them
            keep_best  a contested record goes to its highest-scoring claimant
            margin05   keep_best, but only if the winner leads by >= 0.05;
                       otherwise dropped from everyone

Conflict resolution runs GLOBALLY over the whole evaluation set, exactly as it
would on test -- which is why the set was chosen to be competition-closed.

The decision rule (tau, delta) is tuned NESTED: for each fold it is chosen on the
other four folds and scored on the held-out one. Without nesting, a variant with
more knobs looks better merely from tuning on the data it is scored on.
Differences are reported as paired per-fold deltas against lgb/none.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv

CV = os.path.join(WORK, "cv")
TAUS = np.round(np.arange(0.70, 0.971, 0.02), 3)
DELTAS = (1.0, 0.12, 0.08, 0.05)
KMAX = 12
VARIANTS = ("none", "drop_all", "keep_best", "margin05")


def entity_f(g, keep, lab, n_true, n_q):
    tp = np.bincount(g[keep], weights=lab[keep].astype(np.float64), minlength=n_q)
    npred = np.bincount(g[keep], minlength=n_q).astype(np.float64)
    nt = n_true.astype(np.float64)
    f = np.zeros(n_q)
    s = nt == 0
    f[s & (npred == 0)] = 1.0
    ok = (~s) & (npred > 0) & (tp > 0)
    p = tp[ok] / npred[ok]
    r = tp[ok] / nt[ok]
    f[ok] = (1.25 * p * r) / (0.25 * p + r)
    return f


def resolve(idx, cd, sc, variant):
    """idx: accepted row positions. Returns the subset surviving resolution."""
    if variant == "none" or len(idx) == 0:
        return idx
    c = cd[idx]
    s = sc[idx]
    o = np.lexsort((-s, c))                      # by record, best score first
    c, s, idx = c[o], s[o], idx[o]
    first = np.ones(len(c), dtype=bool)
    first[1:] = c[1:] != c[:-1]
    grp = np.cumsum(first) - 1
    size = np.bincount(grp)
    contested = size[grp] > 1
    if variant == "drop_all":
        return idx[~contested]
    if variant == "keep_best":
        return idx[first]
    if variant == "margin05":
        # lead of the top claimant over the runner-up, per record
        lead = np.full(len(size), np.inf)
        second = np.flatnonzero(~first & np.r_[False, first[:-1]])   # 2nd row of group
        lead[grp[second]] = s[second - 1] - s[second]
        ok_grp = lead >= 0.05
        return idx[first & ok_grp[grp]]
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(WORK, "cvcache"))
    args = ap.parse_args()

    z = np.load(args.data + "_meta.npz", allow_pickle=True)
    gid, label, cand = z["gid"], z["label"], z["cand"]
    n_true, n_q = z["n_true"], int(z["n_q"])
    ef = np.load(os.path.join(CV, "entity_fold.npy"))
    K = int(ef.max()) + 1
    fold_n = np.bincount(ef, minlength=K).astype(np.float64)
    meta = read_tsv(os.path.join(WORK, "cv_meta.tsv"))
    cmap = dict(zip(meta["source1_entity_id"].to_pylist(), meta["country"].to_pylist()))
    country = np.array([cmap.get(e, "?") for e in z["entity"]])

    # ---- reference points
    retrieved = np.bincount(gid[label == 1], minlength=n_q).astype(np.float64)
    print("evaluation set: %d entities, %d candidate rows, %d true pairs"
          % (n_q, len(gid), int(n_true.sum())))
    print("blocking recall %.4f" % (retrieved.sum() / n_true.sum()))
    orc = entity_f(gid, label == 1, label, n_true, n_q)
    print("oracle macro F0.5 (perfect matcher on these candidates) %.5f\n" % orc.mean())

    scores = {}
    for kind in ("lgb", "xgb", "cat"):
        p = os.path.join(CV, "oof_%s.npy" % kind)
        if os.path.exists(p):
            scores[kind] = np.load(p)
    if not scores:
        raise SystemExit("no oof_*.npy found -- run cv_train.py first")
    if len(scores) >= 2:
        scores["blend_mean"] = np.mean([scores[k] for k in list(scores)], axis=0)
    if "lgb" in scores and "xgb" in scores:
        scores["blend_lgb_xgb"] = (scores["lgb"] + scores["xgb"]) / 2

    results = {}
    for mname, s_all in scores.items():
        sub = np.flatnonzero(s_all >= min(TAUS) - max(d for d in DELTAS if d < 1) - 0.2)
        o = np.lexsort((-s_all[sub], gid[sub]))
        rows = sub[o]
        g, sc, lab, cd = gid[rows], s_all[rows], label[rows], cand[rows]
        starts = np.searchsorted(g, np.arange(n_q), "left")
        ends = np.searchsorted(g, np.arange(n_q), "right")
        has = ends > starts
        best = np.full(n_q, -1.0, dtype=np.float32)
        best[has] = sc[starts[has]]
        rank = np.arange(len(g)) - starts[g]
        for variant in VARIANTS:
            grid = {}
            for t in TAUS:
                for d in DELTAS:
                    acc = np.flatnonzero((sc >= t) & (sc >= best[g] - d) & (rank < KMAX))
                    keep = np.zeros(len(g), dtype=bool)
                    keep[resolve(acc, cd, sc, variant)] = True
                    f = entity_f(g, keep, lab, n_true, n_q)
                    grid[(t, d)] = (np.bincount(ef, weights=f, minlength=K) / fold_n, f)
            # nested selection
            per_fold, chosen, f_nested = [], [], np.zeros(n_q)
            for k in range(K):
                other = [j for j in range(K) if j != k]
                w = fold_n[other]
                pick = max(grid, key=lambda p: (grid[p][0][other] * w).sum() / w.sum())
                chosen.append(pick)
                per_fold.append(grid[pick][0][k])
                f_nested[ef == k] = grid[pick][1][ef == k]
            results[(mname, variant)] = (np.array(per_fold), chosen, f_nested)
            print("  %-14s %-10s nested F0.5 %.5f  (folds %s)"
                  % (mname, variant, f_nested.mean(),
                     " ".join("%.4f" % v for v in per_fold)), flush=True)

    base = results.get(("lgb", "none"))
    print("\n" + "=" * 92)
    print("%-14s %-10s %-9s %-8s %-19s %-7s %-7s %s"
          % ("model", "variant", "F0.5", "±sd", "delta vs lgb/none", "wins", "US",
             "India"))
    print("-" * 92)
    rank_rows = sorted(results.items(), key=lambda kv: -kv[1][2].mean())
    for (mname, variant), (pf, chosen, fn) in rank_rows:
        if base is not None:
            d = pf - base[0]
            dtxt = "%+.5f ± %.5f" % (d.mean(), d.std(ddof=1))
            wins = "%d/%d" % (int((d > 0).sum()), len(d))
        else:
            dtxt, wins = "-", "-"
        us = fn[country == "US"].mean()
        ind = fn[country == "India"].mean()
        print("%-14s %-10s %-9.5f %-8.5f %-19s %-7s %-7.4f %.4f"
              % (mname, variant, fn.mean(), pf.std(ddof=1), dtxt, wins, us, ind))
    print("=" * 92)
    (bm, bv), (bpf, bch, bfn) = rank_rows[0]
    print("\nBEST: %s + %s   nested macro F0.5 = %.5f" % (bm, bv, bfn.mean()))
    print("      thresholds chosen per fold (tau, delta): %s"
          % ", ".join("(%.2f, %.2f)" % c for c in bch))
    np.save(os.path.join(CV, "summary.npy"),
            np.array([(m, v, r[2].mean()) for (m, v), r in results.items()], dtype=object),
            allow_pickle=True)


if __name__ == "__main__":
    main()
