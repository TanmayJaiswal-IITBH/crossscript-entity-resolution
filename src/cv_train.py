"""Step 3: 5-fold cross-validation over the evaluation set, one model at a time.

Folds are random by S1 entity. Each fold's model is trained on the other four
folds and scores every candidate of the held-out fold, so every entity receives
an out-of-fold score from a model that never saw it. Nothing from the production
model is reused.

Memory: the 7.6 GB feature file is never memory-mapped. Random row selection on
a memmap pulls most of the file into the process working set (~6 GB extra on this
machine). Instead:
  pass 1  one sequential read collects a fixed training subsample (all positives,
          a fraction of negatives) -- ~1.8 GB
  train   the five fold models are fit from that subsample
  pass 2  one sequential streaming read scores every row with its fold's model
Peak private memory is ~4 GB.

LightGBM trains on CPU; XGBoost and CatBoost on the GPU.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK
from features2 import FEATURE_NAMES

CV = os.path.join(WORK, "cv")
os.makedirs(CV, exist_ok=True)
K = 5
FOLD_SEED = 20260927
BLOCK = 500_000


def folds_for(n_q):
    path = os.path.join(CV, "entity_fold.npy")
    if os.path.exists(path):
        return np.load(path)
    rng = np.random.default_rng(FOLD_SEED)
    f = (rng.permutation(n_q) % K).astype(np.int8)
    np.save(path, f)
    return f


def read_selected(path, n_rows, n_feat, rows):
    """Sequentially read the (sorted) selected rows of the raw float32 file."""
    out = np.empty((len(rows), n_feat), dtype=np.float32)
    pos = 0
    with open(path, "rb") as f:
        for start in range(0, n_rows, BLOCK):
            end = min(start + BLOCK, n_rows)
            lo, hi = np.searchsorted(rows, [start, end])
            if hi == lo:
                continue
            f.seek(start * n_feat * 4)
            buf = np.fromfile(f, dtype=np.float32, count=(end - start) * n_feat)
            buf = buf.reshape(end - start, n_feat)
            out[pos:pos + (hi - lo)] = buf[rows[lo:hi] - start]
            pos += hi - lo
    return out


def fit(kind, Xtr, ytr, Xes, yes, device):
    """-> (predict_fn, best_iteration)"""
    if kind == "lgb":
        import lightgbm as lgb
        p = dict(objective="binary", metric="average_precision", learning_rate=0.06,
                 num_leaves=127, min_data_in_leaf=150, feature_fraction=0.85,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                 num_threads=8, verbosity=-1, seed=0)
        dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES)
        des = lgb.Dataset(Xes, label=yes, feature_name=FEATURE_NAMES, reference=dtr)
        b = lgb.train(p, dtr, num_boost_round=1500, valid_sets=[des],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        it = b.best_iteration
        return (lambda X: b.predict(X, num_iteration=it)), it
    if kind == "xgb":
        import xgboost as xgb
        p = dict(objective="binary:logistic", eval_metric="aucpr", eta=0.08,
                 max_depth=8, min_child_weight=20, subsample=0.8,
                 colsample_bytree=0.85, reg_lambda=2.0, seed=0, tree_method="hist",
                 device="cuda" if device == "gpu" else "cpu")
        if device == "gpu":
            dtr = xgb.QuantileDMatrix(Xtr, label=ytr, feature_names=FEATURE_NAMES)
            des = xgb.QuantileDMatrix(Xes, label=yes, feature_names=FEATURE_NAMES, ref=dtr)
        else:
            dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATURE_NAMES)
            des = xgb.DMatrix(Xes, label=yes, feature_names=FEATURE_NAMES)
        b = xgb.train(p, dtr, num_boost_round=1500, evals=[(des, "es")],
                      early_stopping_rounds=60, verbose_eval=False)
        it = b.best_iteration
        del dtr, des
        return (lambda X: b.predict(xgb.DMatrix(X, feature_names=FEATURE_NAMES),
                                    iteration_range=(0, it + 1))), it
    if kind == "cat":
        from catboost import CatBoostClassifier, Pool as CPool
        kw = dict(iterations=1500, learning_rate=0.08, depth=8, l2_leaf_reg=4.0,
                  random_seed=0, od_type="Iter", od_wait=60, verbose=False)
        if device == "gpu":
            kw.update(task_type="GPU", devices="0", eval_metric="AUC")
        else:
            kw.update(thread_count=8, eval_metric="PRAUC")
        m = CatBoostClassifier(**kw)
        m.fit(CPool(Xtr, ytr), eval_set=CPool(Xes, yes), use_best_model=True)
        return (lambda X: m.predict_proba(X)[:, 1]), m.get_best_iteration()
    raise SystemExit("unknown kind %r" % kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, help="lgb | xgb | cat")
    ap.add_argument("--data", default=os.path.join(WORK, "cvcache"))
    ap.add_argument("--neg-rate", type=float, default=0.22)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    z = np.load(args.data + "_meta.npz", allow_pickle=True)
    gid, label = z["gid"], z["label"]
    n_q, n_rows, n_feat = int(z["n_q"]), int(z["n_rows"]), int(z["n_feat"])
    xpath = args.data + "_X.f32"
    ef = folds_for(n_q)
    row_fold = ef[gid]
    dev = args.device if args.kind in ("xgb", "cat") else "cpu"
    t0 = time.time()
    print("%s: %d rows, %d entities, %d folds, device=%s"
          % (args.kind, n_rows, n_q, K, dev), flush=True)

    # pass 1: fixed training subsample -- every positive, a fraction of negatives
    rng = np.random.default_rng(1)
    s_rows = np.flatnonzero((label == 1) | (rng.random(n_rows) < args.neg_rate))
    Xs = read_selected(xpath, n_rows, n_feat, s_rows)
    ys, gs, fs = label[s_rows], gid[s_rows], row_fold[s_rows]
    print("  subsample: %d rows (%.2f GB) read in %.0fs"
          % (len(s_rows), Xs.nbytes / 1e9, time.time() - t0), flush=True)

    predictors = []
    for k in range(K):
        tr_ent = np.flatnonzero(ef != k)
        es_mask = np.zeros(n_q, dtype=bool)          # early-stop set: 10% of ENTITIES
        es_mask[rng.choice(tr_ent, int(len(tr_ent) * 0.10), replace=False)] = True
        in_tr = fs != k
        is_es = es_mask[gs]
        tr = np.flatnonzero(in_tr & ~is_es)
        es = np.flatnonzero(in_tr & is_es)
        pf, it = fit(args.kind, Xs[tr], ys[tr], Xs[es], ys[es], dev)
        predictors.append(pf)
        print("  fold %d trained: %d rows, early-stop %d rows, iter=%s  %.0fs"
              % (k, len(tr), len(es), it, time.time() - t0), flush=True)
    del Xs

    # pass 2: stream every row once, score it with its own fold's model
    oof = np.full(n_rows, np.nan, dtype=np.float32)
    with open(xpath, "rb") as f:
        for start in range(0, n_rows, BLOCK):
            end = min(start + BLOCK, n_rows)
            buf = np.fromfile(f, dtype=np.float32, count=(end - start) * n_feat)
            buf = buf.reshape(end - start, n_feat)
            rf = row_fold[start:end]
            for k in range(K):
                m = np.flatnonzero(rf == k)
                if len(m):
                    oof[start + m] = predictors[k](buf[m])
    assert not np.isnan(oof).any(), "some rows never received an out-of-fold score"
    np.save(os.path.join(CV, "oof_%s.npy" % args.kind), oof)
    print("saved oof_%s.npy  (%.0fs total)" % (args.kind, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
