"""Print missed true pairs from blocking, with the keys each side generated."""
import os
import sys
import random
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from blocking import Index, query_keys, topk_candidates
from keys import gen_keys

N_SHOW = 22
KMAX = 80


def main():
    t = read_tsv(os.path.join(WORK, "val_truth.tsv"))
    vids = t["source1_entity_id"].to_pylist()
    vcells = t["matched_entity_ids"].fill_null("").to_pylist()
    truth = {i: set(c.split(",")) if c else set() for i, c in zip(vids, vcells)}
    vset = set(vids)

    tb = pq.read_table(os.path.join(WORK, "train_s1_norm.parquet"))
    ids = tb["entity_id"].to_pylist()
    keep = [i for i, e in enumerate(ids) if e in vset]
    tb = tb.take(keep)
    q = {c: tb[c].to_pylist() for c in tb.column_names}
    n_q = len(q["entity_id"])
    qpos = {e: i for i, e in enumerate(q["entity_id"])}

    cand = [set() for _ in range(n_q)]
    pools = {}
    for src in (2, 3):
        idx = Index(os.path.join(WORK, "idx_train_s%d.npz" % src))
        pt = pq.read_table(os.path.join(WORK, "train_s%d_norm.parquet" % src))
        pids = pt["entity_id"].to_pylist()
        pools[src] = (pt, {e: i for i, e in enumerate(pids)})
        for s in range(0, n_q, 5000):
            e = min(s + 5000, n_q)
            qh, qr = query_keys(q["country"][s:e], q["core"][s:e], q["addr"][s:e],
                                q["digits"][s:e], q["state"][s:e])
            qq, cc, _ = topk_candidates(idx, qh, qr, e - s, KMAX)
            for a, b in zip(qq.tolist(), cc.tolist()):
                cand[s + a].add(pids[b])
        del idx

    missed = []
    for i, eid in enumerate(q["entity_id"]):
        m = truth[eid] - cand[i]
        if m:
            missed.append((eid, m, len(cand[i])))
    print("entities with misses: %d" % len(missed))
    random.seed(3)
    for eid, m, ncand in random.sample(missed, min(N_SHOW, len(missed))):
        i = qpos[eid]
        print("\n=== %s  (%s)  ncand=%d  missed %d of %d ==="
              % (eid, q["country"][i], ncand, len(m), len(truth[eid])))
        print("  S1 name=%r" % q["name"][i])
        print("  S1 core=%r  addr=%r  dig=%r st=%r"
              % (q["core"][i], q["addr"][i], q["digits"][i], q["state"][i]))
        qk = set(gen_keys(q["country"][i], q["core"][i], q["addr"][i],
                          q["digits"][i], q["state"][i]))
        for mid in sorted(m):
            src = 2 if mid[1] == "2" else 3
            pt, pmap = pools[src]
            j = pmap.get(mid)
            if j is None:
                print("   MISS %s  <not in pool!>" % mid)
                continue
            row = {c: pt[c][j].as_py() for c in pt.column_names}
            mk = set(gen_keys(row["country"], row["core"], row["addr"],
                              row["digits"], row["state"]))
            print("   MISS %s name=%r" % (mid, row["name"]))
            print("        core=%r addr=%r dig=%r st=%r"
                  % (row["core"], row["addr"], row["digits"], row["state"]))
            shared = qk & mk
            print("        shared keys: %s" % (sorted(shared)[:4] if shared
                                               else "NONE  (blocking cannot see this pair)"))


if __name__ == "__main__":
    main()
