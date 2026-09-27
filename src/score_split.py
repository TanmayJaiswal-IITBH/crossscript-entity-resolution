"""Score every candidate of a labelled split and cache (group, score, label).

That cache is what the decision-rule sweep runs over, so rule tuning costs
seconds instead of re-running retrieval.
"""
import os
import sys
import time
import argparse
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from features2 import base_score, N_ABS
from pairs import Pools, load_queries
from engine import featurize_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--truth", default=os.path.join(WORK, "val_truth.tsv"))
    ap.add_argument("--model", default=os.path.join(WORK, "matcher_lgb.txt"))
    ap.add_argument("--out", default=os.path.join(WORK, "val_scored.npz"))
    ap.add_argument("--kn", type=int, default=30)
    ap.add_argument("--ka", type=int, default=30)
    ap.add_argument("--bn", type=int, default=4000)
    ap.add_argument("--ba", type=int, default=4000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--no-calib", action="store_true")
    ap.add_argument("--dense", default=None,
                    help="dense candidate file from embed_retrieve.py (unioned in)")
    ap.add_argument("--dense-k", type=int, default=25)
    args = ap.parse_args()

    booster = iso = None
    if args.model and os.path.exists(args.model):
        import lightgbm as lgb
        booster = lgb.Booster(model_file=args.model)
        print("model: %s (%d trees)" % (args.model, booster.num_trees()))
        cp = os.path.join(WORK, "matcher_calib.pkl")
        if os.path.exists(cp) and not args.no_calib:
            import pickle
            with open(cp, "rb") as f:
                iso = pickle.load(f)
            print("calibration: %s" % cp)
    else:
        print("no model -- scoring with the unsupervised blend")

    t = read_tsv(args.truth)
    tids = t["source1_entity_id"].to_pylist()
    cells = t["matched_entity_ids"].fill_null("").to_pylist()
    truth = {i: (set(c.split(",")) if c else set()) for i, c in zip(tids, cells)}
    q = load_queries(args.split, set(tids))
    n_q = len(q["entity_id"])
    n_true = np.fromiter((len(truth[e]) for e in q["entity_id"]), dtype=np.int32,
                         count=n_q)
    print("entities: %d   true pairs: %d" % (n_q, int(n_true.sum())), flush=True)

    pools = Pools(args.split, dense=args.dense, dense_k=args.dense_k)
    G, S, Y = [], [], []
    t0 = time.time()
    with Pool(args.procs) as pool:
        for s in range(0, n_q, args.chunk):
            e = min(s + args.chunk, n_q)
            X, ids, starts, ends = featurize_chunk(pools, q, s, e, args, pool)
            if len(ids) == 0:
                continue
            gid = np.zeros(len(ids), dtype=np.int32)
            for i, (a, b) in enumerate(zip(starts, ends)):
                gid[a:b] = s + i
            if booster is not None:
                sc = booster.predict(X)
                if iso is not None:
                    sc = iso.predict(sc)
            else:
                sc = base_score(X[:, :N_ABS])
            y = np.fromiter((1 if ids[k] in truth[q["entity_id"][gid[k]]] else 0
                             for k in range(len(ids))), dtype=np.uint8, count=len(ids))
            G.append(gid)
            S.append(sc.astype(np.float32))
            Y.append(y)
            if (s // args.chunk) % 5 == 0:
                el = time.time() - t0
                print("  %d/%d  %.0fs (eta %.0fs)"
                      % (e, n_q, el, el * (n_q - e) / max(e, 1)), flush=True)
    G = np.concatenate(G)
    S = np.concatenate(S)
    Y = np.concatenate(Y)
    np.savez(args.out, gid=G, score=S, label=Y, n_true=n_true, n_q=np.int32(n_q),
             entity=np.array(q["entity_id"], dtype=object), allow_pickle=True)
    recall = Y.sum() / max(int(n_true.sum()), 1)
    print("saved %s : %d pairs, blocking recall %.4f, %.0fs"
          % (args.out, len(Y), recall, time.time() - t0))


if __name__ == "__main__":
    main()
