"""Generate a labelled pair dataset for the matcher, from the `fit` split.

Writes, per pool source, a memmapped feature matrix plus labels:
    work/train_X_s{2,3}.npy   float32 (n_pairs, N_FEATURES)
    work/train_y_s{2,3}.npy   uint8   (n_pairs,)
    work/train_q_s{2,3}.npy   int32   (n_pairs,)  query row index

The `fit` entities are disjoint from `val` by construction (see split.py), so the
classifier never sees a validation entity.
"""
import os
import sys
import time
import argparse
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from block3 import FieldIndex, retrieve_union
from features import featurize_block, N_FEATURES


def _feat_worker(a):
    return featurize_block(*a)


def load_queries(split, subset_ids):
    tb = pq.read_table(os.path.join(WORK, "%s_s1_norm.parquet" % split))
    ids = tb["entity_id"].to_pylist()
    tb = tb.take([i for i, e in enumerate(ids) if e in subset_ids])
    return {c: tb[c].to_pylist() for c in tb.column_names}


def run_source(split, src, q, truth, args):
    t0 = time.time()
    fi = FieldIndex(split, src)
    pt = pq.read_table(os.path.join(WORK, "%s_s%d_norm.parquet" % (split, src)),
                       columns=["entity_id", "core", "addr"])
    pids = pt["entity_id"].to_pylist()
    pcore = pt["core"].to_pylist()
    paddr = pt["addr"].to_pylist()
    del pt
    n_q = len(q["entity_id"])
    Xs, ys, qs = [], [], []
    n_pos = n_tot = 0
    with Pool(args.procs) as pool:
        for s in range(0, n_q, args.chunk):
            e = min(s + args.chunk, n_q)
            (qa_, ca_, sa_), (qb_, cb_, sb_) = retrieve_union(
                fi, q["country"][s:e], q["core"][s:e], q["addr"][s:e],
                k_name=args.kn, k_addr=args.ka, budget_name=args.bn, budget_addr=args.ba)
            m = {}
            for a, b, c in zip(qa_.tolist(), ca_.tolist(), sa_.tolist()):
                m[(a, b)] = (c, 0.0)
            for a, b, c in zip(qb_.tolist(), cb_.tolist(), sb_.tolist()):
                k = (a, b)
                prev = m.get(k)
                m[k] = (prev[0], c) if prev else (0.0, c)
            if not m:
                continue
            keys = list(m)
            qi = np.fromiter((k[0] for k in keys), dtype=np.int64, count=len(keys))
            pi = np.fromiter((k[1] for k in keys), dtype=np.int64, count=len(keys))
            bn = np.fromiter((m[k][0] for k in keys), dtype=np.float32, count=len(keys))
            ba = np.fromiter((m[k][1] for k in keys), dtype=np.float32, count=len(keys))
            qc_l = [q["core"][s + i] for i in qi]
            qa_l = [q["addr"][s + i] for i in qi]
            pc_l = [pcore[j] for j in pi]
            pa_l = [paddr[j] for j in pi]
            nb = max(1, len(keys) // (args.procs * 4))
            tasks = [(qc_l[i:i + nb], qa_l[i:i + nb], pc_l[i:i + nb], pa_l[i:i + nb],
                      bn[i:i + nb], ba[i:i + nb]) for i in range(0, len(keys), nb)]
            X = np.concatenate(pool.map(_feat_worker, tasks, chunksize=1))
            y = np.fromiter((1 if pids[pi[i]] in truth[q["entity_id"][s + qi[i]]] else 0
                             for i in range(len(keys))), dtype=np.uint8, count=len(keys))
            # keep every positive; subsample negatives to control size
            if args.neg_rate < 1.0:
                rng = np.random.default_rng(1000 + s)
                keep = (y == 1) | (rng.random(len(y)) < args.neg_rate)
                X, y, qi = X[keep], y[keep], qi[keep]
            Xs.append(X)
            ys.append(y)
            qs.append((qi + s).astype(np.int32))
            n_pos += int(y.sum())
            n_tot += len(y)
            if (s // args.chunk) % 10 == 0:
                print("    S%d %d/%d  pairs=%d pos=%d  %.0fs"
                      % (src, e, n_q, n_tot, n_pos, time.time() - t0), flush=True)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    qq = np.concatenate(qs)
    np.save(os.path.join(WORK, "train_X_s%d.npy" % src), X)
    np.save(os.path.join(WORK, "train_y_s%d.npy" % src), y)
    np.save(os.path.join(WORK, "train_q_s%d.npy" % src), qq)
    print("  S%d: %d pairs, %d positives (%.3f%%), %.0fs"
          % (src, len(y), int(y.sum()), 100 * y.mean(), time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--truth", default=os.path.join(WORK, "fit_truth.tsv"))
    ap.add_argument("--kn", type=int, default=30)
    ap.add_argument("--ka", type=int, default=30)
    ap.add_argument("--bn", type=int, default=4000)
    ap.add_argument("--ba", type=int, default=4000)
    ap.add_argument("--chunk", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--neg-rate", type=float, default=0.35)
    ap.add_argument("--only-src", type=int, default=0)
    args = ap.parse_args()

    t = read_tsv(args.truth)
    tids = t["source1_entity_id"].to_pylist()
    cells = t["matched_entity_ids"].fill_null("").to_pylist()
    truth = {i: (set(c.split(",")) if c else set()) for i, c in zip(tids, cells)}
    q = load_queries(args.split, set(tids))
    print("fit queries: %d" % len(q["entity_id"]), flush=True)
    for src in ((args.only_src,) if args.only_src else (2, 3)):
        run_source(args.split, src, q, truth, args)


if __name__ == "__main__":
    main()
