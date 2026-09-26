"""Blocking v2 recall / reduction-ratio measurement on the frozen val split."""
import os
import sys
import time
import argparse
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from block2 import TokIndex, retrieve


def load_val_queries(split="train", truth_file="val_truth.tsv"):
    t = read_tsv(os.path.join(WORK, truth_file))
    vids = t["source1_entity_id"].to_pylist()
    cells = t["matched_entity_ids"].fill_null("").to_pylist()
    truth = {i: set(c.split(",")) if c else set() for i, c in zip(vids, cells)}
    vset = set(vids)
    tb = pq.read_table(os.path.join(WORK, "%s_s1_norm.parquet" % split))
    ids = tb["entity_id"].to_pylist()
    tb = tb.take([i for i, e in enumerate(ids) if e in vset])
    q = {c: tb[c].to_pylist() for c in tb.column_names}
    return q, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--no-ngrams", action="store_true")
    args = ap.parse_args()

    q, truth = load_val_queries()
    n_q = len(q["entity_id"])
    meta = read_tsv(os.path.join(WORK, "val_meta.tsv"))
    mmap = dict(zip(meta["source1_entity_id"].to_pylist(),
                    zip(meta["country"].to_pylist(), meta["bucket"].to_pylist())))
    cands = [[] for _ in range(n_q)]
    pool_total = 0
    for src in (2, 3):
        t0 = time.time()
        idx = TokIndex(os.path.join(WORK, "tok_train_s%d.npz" % src))
        pool_total += idx.n_rec
        pids = pq.read_table(os.path.join(WORK, "train_s%d_norm.parquet" % src),
                             columns=["entity_id"])["entity_id"].to_pylist()
        for s in range(0, n_q, args.chunk):
            e = min(s + args.chunk, n_q)
            qq, cc, ss = retrieve(idx, q["country"][s:e], q["core"][s:e], q["addr"][s:e],
                                  k=args.k, budget=args.budget, ngrams=not args.no_ngrams)
            for a, b, c in zip(qq.tolist(), cc.tolist(), ss.tolist()):
                cands[s + a].append((c, pids[b]))
        print("  S%d retrieved in %.0fs" % (src, time.time() - t0))
        del idx, pids

    n_true = sum(len(v) for v in truth.values())
    print("\nK=%d budget=%d ngrams=%s" % (args.k, args.budget, not args.no_ngrams))
    print("%-6s %-10s %-12s %-14s" % ("topK", "recall", "cand/query", "reduction"))
    for k in (10, 20, 30, 40, 60, 80):
        if k > 2 * args.k:
            break
        got = ncand = 0
        for i, eid in enumerate(q["entity_id"]):
            top = set(e for _, e in sorted(cands[i], reverse=True)[:k])
            ncand += len(top)
            got += len(truth[eid] & top)
        print("%-6d %-10.4f %-12.1f %-14.8f"
              % (k, got / n_true, ncand / n_q, 1 - ncand / (n_q * pool_total)))

    kk = 2 * args.k
    groups = {}
    ent_miss = 0
    for i, eid in enumerate(q["entity_id"]):
        cs = set(e for _, e in cands[i])
        tr = truth[eid]
        if tr - cs:
            ent_miss += 1
        if not tr:
            continue
        g = mmap.get(eid, ("?", "?"))
        a, b = groups.setdefault(g, [0, 0])
        groups[g] = [a + len(tr & cs), b + len(tr)]
    print("\nbreakdown (full candidate union, <=%d per query):" % kk)
    for g in sorted(groups):
        a, b = groups[g]
        print("  %-16s recall %.4f  (%d/%d)" % (str(g), a / b, a, b))
    print("entities with >=1 missed match: %d / %d (%.2f%%)"
          % (ent_miss, n_q, 100 * ent_miss / n_q))


if __name__ == "__main__":
    main()
