"""Blocking v3 recall measurement on the frozen val split."""
import os
import sys
import time
import argparse
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from block3 import FieldIndex, retrieve_union


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
    ap.add_argument("--kn", type=int, default=25)
    ap.add_argument("--ka", type=int, default=25)
    ap.add_argument("--bn", type=int, default=2500)
    ap.add_argument("--ba", type=int, default=2500)
    ap.add_argument("--chunk", type=int, default=4000)
    args = ap.parse_args()

    q, truth = load_val_queries()
    n_q = len(q["entity_id"])
    meta = read_tsv(os.path.join(WORK, "val_meta.tsv"))
    mmap = dict(zip(meta["source1_entity_id"].to_pylist(),
                    zip(meta["country"].to_pylist(), meta["bucket"].to_pylist())))

    cands = [dict() for _ in range(n_q)]     # eid -> (name_score, addr_score)
    pool_total = 0
    for src in (2, 3):
        t0 = time.time()
        fi = FieldIndex("train", src)
        pool_total += fi.n_rec
        pids = pq.read_table(os.path.join(WORK, "train_s%d_norm.parquet" % src),
                             columns=["entity_id"])["entity_id"].to_pylist()
        for s in range(0, n_q, args.chunk):
            e = min(s + args.chunk, n_q)
            (qa, ca, sa), (qb, cb, sb) = retrieve_union(
                fi, q["country"][s:e], q["core"][s:e], q["addr"][s:e],
                k_name=args.kn, k_addr=args.ka, budget_name=args.bn, budget_addr=args.ba)
            for a, b, c in zip(qa.tolist(), ca.tolist(), sa.tolist()):
                d = cands[s + a]
                r = d.get(pids[b], (0.0, 0.0))
                d[pids[b]] = (c, r[1])
            for a, b, c in zip(qb.tolist(), cb.tolist(), sb.tolist()):
                d = cands[s + a]
                r = d.get(pids[b], (0.0, 0.0))
                d[pids[b]] = (r[0], c)
        print("  S%d retrieved in %.0fs" % (src, time.time() - t0))
        del fi, pids

    n_true = sum(len(v) for v in truth.values())
    ncand = sum(len(d) for d in cands)
    got = sum(len(truth[e] & set(cands[i])) for i, e in enumerate(q["entity_id"]))
    print("\nkn=%d ka=%d bn=%d ba=%d" % (args.kn, args.ka, args.bn, args.ba))
    print("UNION recall %.4f   cand/query %.1f   reduction %.8f"
          % (got / n_true, ncand / n_q, 1 - ncand / (n_q * pool_total)))

    # recall of each field alone, and of a combined-rank truncation
    for field, j in (("name-only", 0), ("addr-only", 1)):
        g = sum(len(truth[e] & set(k for k, v in cands[i].items() if v[j] > 0))
                for i, e in enumerate(q["entity_id"]))
        print("  %-10s recall %.4f" % (field, g / n_true))
    for cap in (20, 30, 40, 50, 60):
        g = nc = 0
        for i, e in enumerate(q["entity_id"]):
            d = cands[i]
            top = set(sorted(d, key=lambda x: -max(d[x]))[:cap])
            nc += len(top)
            g += len(truth[e] & top)
        print("  cap %-4d recall %.4f  cand/query %.1f" % (cap, g / n_true, nc / n_q))

    groups = {}
    ent_miss = 0
    for i, eid in enumerate(q["entity_id"]):
        cs = set(cands[i])
        tr = truth[eid]
        if tr - cs:
            ent_miss += 1
        if not tr:
            continue
        gkey = mmap.get(eid, ("?", "?"))
        a, b = groups.setdefault(gkey, [0, 0])
        groups[gkey] = [a + len(tr & cs), b + len(tr)]
    print("\nbreakdown:")
    for gkey in sorted(groups):
        a, b = groups[gkey]
        print("  %-16s recall %.4f  (%d/%d)" % (str(gkey), a / b, a, b))
    print("entities with >=1 missed match: %d / %d (%.2f%%)"
          % (ent_miss, n_q, 100 * ent_miss / n_q))


if __name__ == "__main__":
    main()
