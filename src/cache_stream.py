"""Featurise a large labelled entity set, streaming the matrix to disk.

Unlike cache_features.py this never holds the full matrix in RAM: feature
chunks are appended to a raw float32 file and memory-mapped afterwards.
Candidate ids are stored as int64 codes (source * 1e11 + numeric id) instead of
strings, so 30M candidates cost 240 MB rather than several GB of Python objects.
"""
import argparse
import os
import sys
import time

import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from features2 import N_FEATURES
from pairs import Pools, load_queries
from engine import featurize_chunk

CODE_BASE = 10 ** 11


def id_code(eid):
    return int(eid[1]) * CODE_BASE + int(eid[3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--truth", default=os.path.join(WORK, "cv_truth.tsv"))
    ap.add_argument("--out", default=os.path.join(WORK, "cvcache"))
    ap.add_argument("--kn", type=int, default=60)
    ap.add_argument("--ka", type=int, default=60)
    ap.add_argument("--bn", type=int, default=8000)
    ap.add_argument("--ba", type=int, default=8000)
    ap.add_argument("--chunk", type=int, default=1500)
    ap.add_argument("--procs", type=int, default=6)
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
    G, Y, C = [], [], []
    n_rows = 0
    t0 = time.time()
    xpath = args.out + "_X.f32"
    with open(xpath, "wb") as fx, Pool(args.procs) as pool:
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
            codes = np.fromiter((id_code(x) for x in ids), dtype=np.int64,
                                count=len(ids))
            np.ascontiguousarray(X, dtype=np.float32).tofile(fx)
            G.append(gid)
            Y.append(y)
            C.append(codes)
            n_rows += len(ids)
            if (s // args.chunk) % 5 == 0:
                el = time.time() - t0
                print("  %d/%d  rows %d  %.0fs (eta %.0fs)"
                      % (e, n_q, n_rows, el, el * (n_q - e) / max(e, 1)), flush=True)
    np.savez(args.out + "_meta.npz", gid=np.concatenate(G), label=np.concatenate(Y),
             cand=np.concatenate(C), n_true=n_true, n_q=np.int32(n_q),
             n_rows=np.int64(n_rows), n_feat=np.int32(N_FEATURES),
             entity=np.array(q["entity_id"], dtype=object))
    print("saved %s (%d x %d float32, %.1f GB) + meta in %.0fs"
          % (xpath, n_rows, N_FEATURES, n_rows * N_FEATURES * 4 / 1e9,
             time.time() - t0))


if __name__ == "__main__":
    main()
