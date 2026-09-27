"""Cache a split's full feature matrix so any model can be scored on it instantly.

Candidate generation and featurisation are ~95% of inference cost while scoring is
~1%, so caching the matrix once turns a model bake-off from hours into seconds.
Writes a memmappable .npy plus the group ids, labels and candidate ids.
"""
import os
import sys
import time
import argparse
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from features2 import N_FEATURES
from pairs import Pools, load_queries
from engine import featurize_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--truth", default=os.path.join(WORK, "val_truth.tsv"))
    ap.add_argument("--out", default=os.path.join(WORK, "valcache"))
    ap.add_argument("--kn", type=int, default=60)
    ap.add_argument("--ka", type=int, default=60)
    ap.add_argument("--bn", type=int, default=8000)
    ap.add_argument("--ba", type=int, default=8000)
    ap.add_argument("--chunk", type=int, default=1500)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--dense", default=None,
                    help="dense candidate file from embed_retrieve.py (unioned in)")
    ap.add_argument("--dense-k", type=int, default=25)
    args = ap.parse_args()

    t = read_tsv(args.truth)
    tids = t["source1_entity_id"].to_pylist()
    cells = t["matched_entity_ids"].fill_null("").to_pylist()
    truth = {i: (set(c.split(",")) if c else set()) for i, c in zip(tids, cells)}
    q = load_queries(args.split, set(tids))
    n_q = len(q["entity_id"])
    n_true = np.fromiter((len(truth[e]) for e in q["entity_id"]), dtype=np.int32,
                         count=n_q)
    print("entities %d   true pairs %d   features %d"
          % (n_q, int(n_true.sum()), N_FEATURES), flush=True)

    pools = Pools(args.split, dense=args.dense, dense_k=args.dense_k)
    Xs, G, Y, ids_all = [], [], [], []
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
            y = np.fromiter((1 if ids[k] in truth[q["entity_id"][gid[k]]] else 0
                             for k in range(len(ids))), dtype=np.uint8, count=len(ids))
            Xs.append(X)
            G.append(gid)
            Y.append(y)
            ids_all.extend(ids)
            if (s // args.chunk) % 5 == 0:
                el = time.time() - t0
                print("  %d/%d  %.0fs (eta %.0fs)"
                      % (e, n_q, el, el * (n_q - e) / max(e, 1)), flush=True)
    X = np.concatenate(Xs)
    del Xs
    np.save(args.out + "_X.npy", X)
    np.savez(args.out + "_meta.npz", gid=np.concatenate(G), label=np.concatenate(Y),
             n_true=n_true, n_q=np.int32(n_q),
             entity=np.array(q["entity_id"], dtype=object),
             cand=np.array(ids_all, dtype=object))
    print("saved %s_X.npy (%d x %d, %.1f GB) and %s_meta.npz in %.0fs"
          % (args.out, X.shape[0], X.shape[1], X.nbytes / 1e9, args.out,
             time.time() - t0))


if __name__ == "__main__":
    main()
