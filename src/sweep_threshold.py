"""Sweep decision rules over cached candidate scores (no re-retrieval).

Reads the per-source candidate streams written by run_pipeline (ids + scores,
one line per query per source) and evaluates macro F_0.5 for:
  * a global absolute threshold
  * absolute threshold + a relative "keep anything within r of this entity's best"
  * top-1 rescue: if nothing clears the bar, still emit the best candidate when
    it clears a lower bar
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from evaluate import f_beta_half


def load_stream(tag, split, n_q):
    per_q = []
    files = [open(os.path.join(WORK, "cand_%s_%s_s%d.tsv" % (tag, split, s)),
                  encoding="utf-8") for s in (2, 3)]
    for _ in range(n_q):
        ids, scs = [], []
        for fh in files:
            line = fh.readline().rstrip("\n")
            a, _, b = line.partition("\t")
            if a:
                ids.extend(a.split(","))
                scs.extend(float(x) for x in b.split(","))
        per_q.append((ids, np.asarray(scs, dtype=np.float32)))
    for fh in files:
        fh.close()
    return per_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="val")
    ap.add_argument("--split", default="train")
    ap.add_argument("--truth", default=os.path.join(WORK, "val_truth.tsv"))
    ap.add_argument("--order-file", default=os.path.join(WORK, "val_out", "candidate_pairs.tsv"))
    args = ap.parse_args()

    t = read_tsv(args.truth)
    tmap = dict(zip(t["source1_entity_id"].to_pylist(),
                    t["matched_entity_ids"].fill_null("").to_pylist()))
    # the cached streams are in S1-file order, not truth-file order -- take the
    # authoritative ordering from the candidate file the same run wrote.
    c = read_tsv(args.order_file)
    qids = c["source1_entity_id"].to_pylist()
    truth = [set(tmap[q].split(",")) if tmap[q] else set() for q in qids]
    n_q = len(qids)
    per_q = load_stream(args.tag, args.split, n_q)
    print("loaded %d query candidate lists" % n_q)

    best = (0.0, None)
    print("\n%-8s %-8s %-8s %-9s %-9s %-9s" %
          ("abs", "rel", "rescue", "F0.5", "emptyfrac", "pred/q"))
    for thr in np.arange(0.52, 0.921, 0.02):
        for rel in (0.0, 0.88, 0.94):
            for rescue in (0.0, 0.50, 0.60):
                tot = 0.0
                n_empty = 0
                n_pred = 0
                for i in range(n_q):
                    ids, scs = per_q[i]
                    if len(ids) == 0:
                        tot += 1.0 if not truth[i] else 0.0
                        n_empty += 1
                        continue
                    top = scs.max()
                    keep = scs >= thr
                    if rel > 0:
                        keep &= scs >= rel * top
                    if not keep.any() and rescue > 0 and top >= rescue:
                        keep = scs >= top - 1e-9
                    pred = {ids[j] for j in np.flatnonzero(keep)}
                    n_pred += len(pred)
                    if not pred:
                        n_empty += 1
                    tot += f_beta_half(pred, truth[i])
                f = tot / n_q
                if f > best[0]:
                    best = (f, (thr, rel, rescue))
                print("%-8.3f %-8.2f %-8.2f %-9.5f %-9.4f %-9.2f"
                      % (thr, rel, rescue, f, n_empty / n_q, n_pred / n_q))
    print("\nBEST: F0.5=%.5f  (abs=%.3f rel=%.2f rescue=%.2f)"
          % (best[0], best[1][0], best[1][1], best[1][2]))


if __name__ == "__main__":
    main()
