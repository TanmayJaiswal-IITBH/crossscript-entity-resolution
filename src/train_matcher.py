"""Train the pairwise matcher: LightGBM + isotonic calibration.

Cross-validation is grouped by S1 entity -- pairs from one entity are highly
correlated (they share the query side verbatim), so letting them straddle folds
would leak and inflate every estimate. The same grouping produces the
out-of-fold predictions that the isotonic calibrator is fitted on, which is what
makes the Day-2 threshold sweep meaningful: tau is then an actual probability.
"""
import os
import sys
import time
import pickle
import argparse
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK
from features2 import FEATURE_NAMES

MODEL_PATH = os.path.join(WORK, "matcher_lgb.txt")
CALIB_PATH = os.path.join(WORK, "matcher_calib.pkl")


def params(args):
    return dict(objective="binary", metric="average_precision",
                learning_rate=args.lr, num_leaves=args.leaves,
                min_data_in_leaf=args.min_leaf, feature_fraction=0.85,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                num_threads=args.threads, verbosity=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(WORK, "fit"))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=1200)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--min-leaf", type=int, default=150)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-calib", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    X = np.load(args.data + "_X.npy")
    y = np.load(args.data + "_y.npy")
    g = np.load(args.data + "_g.npy")
    assert X.shape[1] == len(FEATURE_NAMES), \
        "feature count %d != %d" % (X.shape[1], len(FEATURE_NAMES))
    print("pairs %d   positives %d (%.2f%%)   entities %d   features %d"
          % (len(y), int(y.sum()), 100 * y.mean(), len(np.unique(g)), X.shape[1]))

    oof = np.zeros(len(y), dtype=np.float64)
    iters = []
    gkf = GroupKFold(n_splits=args.folds)
    for k, (tr, va) in enumerate(gkf.split(X, y, groups=g)):
        dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=FEATURE_NAMES)
        dva = lgb.Dataset(X[va], label=y[va], feature_name=FEATURE_NAMES, reference=dtr)
        b = lgb.train(params(args), dtr, num_boost_round=args.rounds,
                      valid_sets=[dva], valid_names=["va"],
                      callbacks=[lgb.early_stopping(60, verbose=False),
                                 lgb.log_evaluation(200)])
        oof[va] = b.predict(X[va], num_iteration=b.best_iteration)
        iters.append(b.best_iteration)
        print("  fold %d: best_iter=%d  AP=%.5f  AUC=%.5f"
              % (k, b.best_iteration, average_precision_score(y[va], oof[va]),
                 roc_auc_score(y[va], oof[va])), flush=True)
    print("\nOOF  AP=%.5f  AUC=%.5f" % (average_precision_score(y, oof),
                                        roc_auc_score(y, oof)))

    n_final = int(np.mean(iters) * (1 + 1.0 / args.folds))
    print("training final model on all data for %d rounds" % n_final, flush=True)
    dall = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    booster = lgb.train(params(args), dall, num_boost_round=n_final)
    booster.save_model(MODEL_PATH)
    print("saved %s" % MODEL_PATH)

    if not args.no_calib:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(oof, y)
        with open(CALIB_PATH, "wb") as f:
            pickle.dump(iso, f)
        print("saved %s (isotonic on out-of-fold predictions)" % CALIB_PATH)
        # calibration sanity: predicted vs observed rate by decile of OOF score
        q = np.quantile(oof, np.linspace(0, 1, 11))
        print("\ncalibration check (OOF):")
        for i in range(10):
            m = (oof >= q[i]) & (oof <= q[i + 1])
            if m.sum() == 0:
                continue
            print("  bin %2d  raw=%.4f  calibrated=%.4f  actual=%.4f  n=%d"
                  % (i, oof[m].mean(), iso.predict(oof[m]).mean(), y[m].mean(),
                     int(m.sum())))

    imp = booster.feature_importance(importance_type="gain")
    order = np.argsort(-imp)
    print("\ntop features by gain:")
    for i in order[:20]:
        print("  %-14s %12.0f" % (FEATURE_NAMES[i], imp[i]))
    print("\ntotal %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
