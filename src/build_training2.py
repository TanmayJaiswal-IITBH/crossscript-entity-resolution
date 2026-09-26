"""Build the labelled pair dataset for the Day-2 matcher.

Negatives come from the blocking output, which is exactly the distribution the
model meets at inference -- not random pairs. Positives are never subsampled;
negatives are, to keep the matrix in memory.
"""
import os
import sys
import time
import argparse
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from features2 import N_FEATURES, FEATURE_NAMES
from pairs import Pools, load_queries
from engine import featurize_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--truth", default=os.path.join(WORK, "fit_truth.tsv"))
    ap.add_argument("--out", default=os.path.join(WORK, "fit"))
    ap.add_argument("--kn", type=int, default=30)
    ap.add_argument("--ka", type=int, default=30)
    ap.add_argument("--bn", type=int, default=4000)
    ap.add_argument("--ba", type=int, default=4000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--neg-rate", type=float, default=0.30)
    ap.add_argument("--max-entities", type=int, default=0)
    args = ap.parse_args()

    t = read_tsv(args.truth)
    tids = t["source1_entity_id"].to_pylist()
    cells = t["matched_entity_ids"].fill_null("").to_pylist()
    truth = {i: (set(c.split(",")) if c else set()) for i, c in zip(tids, cells)}

    q = load_queries(args.split, set(tids))
    n_q = len(q["entity_id"])
    if args.max_entities:
        n_q = min(n_q, args.max_entities)
    print("entities: %d   features: %d" % (n_q, N_FEATURES), flush=True)

    pools = Pools(args.split)
    print("pools loaded", flush=True)
    rng = np.random.default_rng(99)
    Xs, ys, gs = [], [], []
    n_pos = n_tot = 0
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
            if args.neg_rate < 1.0:
                keep = (y == 1) | (rng.random(len(y)) < args.neg_rate)
                X, y, gid = X[keep], y[keep], gid[keep]
            Xs.append(X)
            ys.append(y)
            gs.append(gid)
            n_pos += int(y.sum())
            n_tot += len(y)
            if (s // args.chunk) % 10 == 0:
                el = time.time() - t0
                print("  %d/%d entities  pairs=%d pos=%d  %.0fs (eta %.0fs)"
                      % (e, n_q, n_tot, n_pos, el, el * (n_q - e) / max(e, 1)), flush=True)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    np.save(args.out + "_X.npy", X)
    np.save(args.out + "_y.npy", y)
    np.save(args.out + "_g.npy", g)
    print("saved %s_{X,y,g}.npy : %d pairs, %d positives (%.2f%%), %.0fs"
          % (args.out, len(y), int(y.sum()), 100 * y.mean(), time.time() - t0))


if __name__ == "__main__":
    main()
