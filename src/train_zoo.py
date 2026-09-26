"""Train the model zoo on the cached training matrix, folds grouped by S1 entity.

Members: LightGBM, CatBoost, XGBoost, and logistic regression as a diagnostic
baseline. All share the same 60 features and the same fold assignment, so their
out-of-fold predictions are directly comparable and can be blended.

Writes each model plus its OOF prediction vector; ensembles are formed later from
the OOF vectors, which is what lets blend weights be chosen without leakage.
"""
import os
import sys
import time
import pickle
import argparse
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK
from features2 import FEATURE_NAMES

ZOO = os.path.join(WORK, "zoo")
os.makedirs(ZOO, exist_ok=True)


def folds(g, n_splits, seed=0):
    """Deterministic entity-grouped folds, shared by every model."""
    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(g), 1), dtype=np.float32)
    return list(gkf.split(X_dummy, groups=g))


def fit_lgb(Xtr, ytr, Xva, yva, seed):
    import lightgbm as lgb
    p = dict(objective="binary", metric="average_precision", learning_rate=0.06,
             num_leaves=127, min_data_in_leaf=150, feature_fraction=0.85,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
             num_threads=8, verbosity=-1, seed=seed)
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES)
    dva = lgb.Dataset(Xva, label=yva, feature_name=FEATURE_NAMES, reference=dtr)
    b = lgb.train(p, dtr, num_boost_round=1400, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    return b, b.predict(Xva, num_iteration=b.best_iteration), b.best_iteration


def fit_cat(Xtr, ytr, Xva, yva, seed):
    from catboost import CatBoostClassifier, Pool as CPool
    m = CatBoostClassifier(iterations=1500, learning_rate=0.08, depth=8,
                           l2_leaf_reg=4.0, eval_metric="PRAUC",
                           random_seed=seed, thread_count=8, verbose=False,
                           od_type="Iter", od_wait=60)
    m.fit(CPool(Xtr, ytr), eval_set=CPool(Xva, yva), use_best_model=True)
    return m, m.predict_proba(Xva)[:, 1], m.get_best_iteration()


def fit_xgb(Xtr, ytr, Xva, yva, seed):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="aucpr", eta=0.08,
             max_depth=8, min_child_weight=20, subsample=0.8,
             colsample_bytree=0.85, reg_lambda=2.0, nthread=8, seed=seed)
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATURE_NAMES)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=FEATURE_NAMES)
    b = xgb.train(p, dtr, num_boost_round=1200, evals=[(dva, "va")],
                  early_stopping_rounds=60, verbose_eval=False)
    return b, b.predict(dva, iteration_range=(0, b.best_iteration + 1)), b.best_iteration


def fit_lr(Xtr, ytr, Xva, yva, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=300, C=1.0, n_jobs=8))
    m.fit(Xtr, ytr)
    return m, m.predict_proba(Xva)[:, 1], 0


FITTERS = {"lgb": fit_lgb, "cat": fit_cat, "xgb": fit_xgb, "lr": fit_lr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(WORK, "fitw"))
    ap.add_argument("--models", default="lgb,cat,xgb,lr")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsample", type=float, default=1.0,
                    help="fraction of entities to train on (for the slower members)")
    args = ap.parse_args()

    X = np.load(args.data + "_X.npy")
    y = np.load(args.data + "_y.npy")
    g = np.load(args.data + "_g.npy")
    if args.subsample < 1.0:
        uq = np.unique(g)
        rng = np.random.default_rng(11)
        keep_e = set(rng.choice(uq, int(len(uq) * args.subsample),
                                replace=False).tolist())
        m = np.fromiter((e in keep_e for e in g), dtype=bool, count=len(g))
        X, y, g = X[m], y[m], g[m]
        print("subsampled to %d pairs / %d entities" % (len(y), len(np.unique(g))))
    print("pairs %d   positives %d (%.2f%%)   features %d"
          % (len(y), int(y.sum()), 100 * y.mean(), X.shape[1]), flush=True)

    fl = folds(g, args.folds)
    np.save(os.path.join(ZOO, "fold_assign.npy"),
            np.concatenate([np.full(len(va), k) for k, (_, va) in enumerate(fl)]))

    for name in args.models.split(","):
        name = name.strip()
        if name not in FITTERS:
            print("skip unknown model %r" % name)
            continue
        t0 = time.time()
        oof = np.zeros(len(y), dtype=np.float32)
        iters = []
        for k, (tr, va) in enumerate(fl):
            mdl, pv, bi = FITTERS[name](X[tr], y[tr], X[va], y[va], args.seed + k)
            oof[va] = pv
            iters.append(bi)
            print("  %s fold %d: iter=%s AP=%.5f  (%.0fs)"
                  % (name, k, bi, average_precision_score(y[va], pv),
                     time.time() - t0), flush=True)
            if k == 0:
                with open(os.path.join(ZOO, "%s_fold0.pkl" % name), "wb") as f:
                    pickle.dump(mdl, f)
        ap_ = average_precision_score(y, oof)
        auc = roc_auc_score(y, oof)
        np.save(os.path.join(ZOO, "%s_oof.npy" % name), oof)
        print("%-4s OOF AP=%.5f AUC=%.5f  mean_iter=%s  %.0fs"
              % (name, ap_, auc, int(np.mean(iters)), time.time() - t0), flush=True)

        # refit on everything for inference
        n_final = max(50, int(np.mean(iters) * (1 + 1.0 / args.folds))) if iters[0] else 0
        t1 = time.time()
        if name == "lgb":
            import lightgbm as lgb
            b = lgb.train(dict(objective="binary", learning_rate=0.06, num_leaves=127,
                               min_data_in_leaf=150, feature_fraction=0.85,
                               bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                               num_threads=8, verbosity=-1, seed=args.seed),
                          lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES),
                          num_boost_round=n_final)
            b.save_model(os.path.join(ZOO, "lgb_full.txt"))
        elif name == "cat":
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(iterations=n_final, learning_rate=0.08, depth=8,
                                   l2_leaf_reg=4.0, random_seed=args.seed,
                                   thread_count=8, verbose=False)
            m.fit(X, y)
            m.save_model(os.path.join(ZOO, "cat_full.cbm"))
        elif name == "xgb":
            import xgboost as xgb
            b = xgb.train(dict(objective="binary:logistic", eta=0.08, max_depth=8,
                               min_child_weight=20, subsample=0.8,
                               colsample_bytree=0.85, reg_lambda=2.0, nthread=8,
                               seed=args.seed),
                          xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES),
                          num_boost_round=n_final)
            b.save_model(os.path.join(ZOO, "xgb_full.json"))
        else:
            mdl, _, _ = FITTERS[name](X, y, X[:1000], y[:1000], args.seed)
            with open(os.path.join(ZOO, "lr_full.pkl"), "wb") as f:
                pickle.dump(mdl, f)
        print("  refit on all data (%s rounds) in %.0fs" % (n_final, time.time() - t1),
              flush=True)

    np.save(os.path.join(ZOO, "y.npy"), y)
    np.save(os.path.join(ZOO, "g.npy"), g)
    print("\nzoo written to %s" % ZOO)


if __name__ == "__main__":
    main()
