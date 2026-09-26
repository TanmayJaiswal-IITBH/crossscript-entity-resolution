"""Where do true matches actually rank under each retrieval score?

If true matches sit at rank 300+, no affordable K will save us and the scoring
function is wrong.  If they sit at rank 30-80, K (or the budget) is the knob.
"""
import os
import sys
import collections
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from block3 import FieldIndex, retrieve_field, split_tokens
from block2 import gen_tokens

N_Q = 1500
BIG_K = 600


def main():
    t = read_tsv(os.path.join(WORK, "val_truth.tsv"))
    vids = t["source1_entity_id"].to_pylist()
    cells = t["matched_entity_ids"].fill_null("").to_pylist()
    rng = np.random.default_rng(5)
    pick = rng.choice(len(vids), N_Q, replace=False)
    truth = {vids[i]: (set(cells[i].split(",")) if cells[i] else set()) for i in pick}
    vset = set(truth)

    tb = pq.read_table(os.path.join(WORK, "train_s1_norm.parquet"))
    ids = tb["entity_id"].to_pylist()
    tb = tb.take([i for i, e in enumerate(ids) if e in vset])
    q = {c: tb[c].to_pylist() for c in tb.column_names}
    n_q = len(q["entity_id"])
    del tb, ids

    nm_toks, ad_toks = [], []
    for i in range(n_q):
        a, b = split_tokens(gen_tokens(q["country"][i], q["core"][i], q["addr"][i]))
        nm_toks.append(a)
        ad_toks.append(b)

    rank_n = collections.Counter()
    rank_a = collections.Counter()
    rank_best = collections.Counter()
    n_pairs = 0
    for src in (2, 3):
        fi = FieldIndex("train", src)
        pids = pq.read_table(os.path.join(WORK, "train_s%d_norm.parquet" % src),
                             columns=["entity_id"])["entity_id"].to_pylist()
        res = {}
        for field, toks in (("name", nm_toks), ("addr", ad_toks)):
            qq, cc, ss = retrieve_field(fi, toks, field, n_q, k=BIG_K, budget=6000)
            d = collections.defaultdict(dict)
            cur_q = -1
            r = 0
            for a, b in zip(qq.tolist(), cc.tolist()):
                if a != cur_q:
                    cur_q, r = a, 0
                d[a][pids[b]] = r
                r += 1
            res[field] = d
        for i, eid in enumerate(q["entity_id"]):
            for m in truth[eid]:
                if (m[1] == "2") != (src == 2):
                    continue
                n_pairs += 1
                rn = res["name"].get(i, {}).get(m, 10 ** 6)
                ra = res["addr"].get(i, {}).get(m, 10 ** 6)
                rank_n[_b(rn)] += 1
                rank_a[_b(ra)] += 1
                rank_best[_b(min(rn, ra))] += 1
        del fi, pids, res

    print("true pairs: %d   (BIG_K=%d, budget=6000)" % (n_pairs, BIG_K))
    for label, ctr in (("name-cosine", rank_n), ("addr-cosine", rank_a),
                       ("best-of-two", rank_best)):
        print("\n%s rank of the true match:" % label)
        cum = 0
        for k in ["0", "1-4", "5-9", "10-24", "25-49", "50-99", "100-299",
                  "300-599", "NOT RETRIEVED"]:
            cum += ctr.get(k, 0)
            print("  %-14s %6d  %5.2f%%   cum %5.2f%%"
                  % (k, ctr.get(k, 0), 100 * ctr.get(k, 0) / n_pairs, 100 * cum / n_pairs))


def _b(r):
    if r >= 10 ** 5:
        return "NOT RETRIEVED"
    if r == 0:
        return "0"
    for lo, hi, lab in ((1, 4, "1-4"), (5, 9, "5-9"), (10, 24, "10-24"),
                        (25, 49, "25-49"), (50, 99, "50-99"), (100, 299, "100-299"),
                        (300, 599, "300-599")):
        if lo <= r <= hi:
            return lab
    return "NOT RETRIEVED"


if __name__ == "__main__":
    main()
