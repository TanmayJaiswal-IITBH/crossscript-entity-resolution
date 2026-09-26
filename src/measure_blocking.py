"""Measure blocking recall and reduction ratio on the frozen validation split.

Recall here is a hard ceiling on everything downstream, so it is measured against
the FULL 5M-record pools, exactly as the test run will see them.
"""
import os
import sys
import time
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from blocking import Index, query_keys, topk_candidates

KS = (10, 20, 30, 50, 80)


def load_queries(split, id_subset=None):
    tb = pq.read_table(os.path.join(WORK, "%s_s1_norm.parquet" % split))
    ids = tb["entity_id"].to_pylist()
    if id_subset is not None:
        keep = [i for i, e in enumerate(ids) if e in id_subset]
        tb = tb.take(keep)
        ids = tb["entity_id"].to_pylist()
    return dict(ids=ids,
                country=tb["country"].to_pylist(), core=tb["core"].to_pylist(),
                addr=tb["addr"].to_pylist(), digits=tb["digits"].to_pylist(),
                state=tb["state"].to_pylist())


def run(split="train", truth_file="val_truth.tsv", meta_file="val_meta.tsv",
        kmax=max(KS), chunk=5000):
    t = read_tsv(os.path.join(WORK, truth_file))
    vids = t["source1_entity_id"].to_pylist()
    vcells = t["matched_entity_ids"].fill_null("").to_pylist()
    truth = {i: set(c.split(",")) if c else set() for i, c in zip(vids, vcells)}
    meta = read_tsv(os.path.join(WORK, meta_file))
    mmap = dict(zip(meta["source1_entity_id"].to_pylist(),
                    zip(meta["country"].to_pylist(), meta["bucket"].to_pylist())))

    q = load_queries(split, id_subset=set(vids))
    n_q = len(q["ids"])
    print("queries: %d" % n_q)

    # per-query candidate sets, accumulated across the two pools
    cand_by_q = [dict() for _ in range(n_q)]   # rec-id string -> shared-key count
    pool_sizes = {}
    for src in (2, 3):
        t0 = time.time()
        idx = Index(os.path.join(WORK, "idx_%s_s%d.npz" % (split, src)))
        pool_ids = pq.read_table(os.path.join(WORK, "%s_s%d_norm.parquet" % (split, src)),
                                 columns=["entity_id"])["entity_id"].to_pylist()
        pool_sizes[src] = len(pool_ids)
        n_post = 0
        for s in range(0, n_q, chunk):
            e = min(s + chunk, n_q)
            qh, qr = query_keys(q["country"][s:e], q["core"][s:e], q["addr"][s:e],
                                q["digits"][s:e], q["state"][s:e])
            qq, cc, kk = topk_candidates(idx, qh, qr, e - s, kmax)
            n_post += len(qq)
            for a, b, c in zip(qq.tolist(), cc.tolist(), kk.tolist()):
                cand_by_q[s + a][pool_ids[b]] = c
        print("  S%d: %.1fs, %d candidate pairs (%.1f per query)"
              % (src, time.time() - t0, n_post, n_post / n_q))
        del idx, pool_ids

    total_pool = pool_sizes[2] + pool_sizes[3]
    # ---- recall at several K
    print("\n%-6s %-10s %-12s %-14s" % ("K", "recall", "cand/query", "reduction"))
    n_true = sum(len(v) for v in truth.values())
    for k in KS:
        got = 0
        ncand = 0
        for i, eid in enumerate(q["ids"]):
            d = cand_by_q[i]
            if len(d) > k:
                top = set(sorted(d, key=lambda x: -d[x])[:k])
            else:
                top = set(d)
            ncand += len(top)
            got += len(truth[eid] & top)
        rr = 1.0 - ncand / (n_q * total_pool)
        print("%-6d %-10.4f %-12.1f %-14.8f" % (k, got / n_true, ncand / n_q, rr))

    # ---- breakdown at kmax
    print("\nbreakdown at K=%d (union over both pools):" % kmax)
    groups = {}
    for i, eid in enumerate(q["ids"]):
        tr = truth[eid]
        if not tr:
            continue
        cand = set(cand_by_q[i])
        g = mmap.get(eid, ("?", "?"))
        a, b = groups.setdefault(g, [0, 0])
        groups[g] = [a + len(tr & cand), b + len(tr)]
    for g in sorted(groups):
        a, b = groups[g]
        print("  %-16s recall %.4f  (%d/%d)" % (str(g), a / b, a, b))

    # entities where blocking lost at least one match
    lost = [(q["ids"][i], truth[q["ids"][i]] - set(cand_by_q[i]))
            for i in range(n_q) if truth[q["ids"][i]] - set(cand_by_q[i])]
    print("\nentities with >=1 missed match: %d / %d (%.2f%%)"
          % (len(lost), n_q, 100 * len(lost) / n_q))
    np.save(os.path.join(WORK, "blocking_missed.npy"),
            np.array([a for a, _ in lost], dtype=object), allow_pickle=True)
    return cand_by_q, q, truth


if __name__ == "__main__":
    run()
