"""Build the frozen validation split.

Design notes
------------
* The split is taken at the Source-1 entity level and carries every one of that
  entity's true S2/S3 matches with it.
* It is stratified by country x match-cardinality bucket (0 / 1 / 2 / 3+), so the
  singleton rate and the many-match tail are both represented faithfully.
* S2/S3 are NOT subsampled.  Validation queries are searched against the *full*
  5.0M/5.3M training pool, so blocking recall and precision measured here are
  directly comparable to what the same pipeline will do on the test set
  (test: 1.73M S1 vs 4.89M S2 / 5.08M S3 -- essentially the same ratio).
* SEED is frozen.  Every experiment for the rest of the challenge is compared on
  this one split.
"""
import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv, source_path, load_gt

SEED = 20260925
N_VAL = 40_000          # entities scored for every experiment
N_FIT = 200_000         # disjoint entities used to train the pair classifier


def bucket_of(n):
    return "0" if n == 0 else ("1" if n == 1 else ("2" if n == 2 else "3+"))


def main():
    rng = np.random.default_rng(SEED)

    s1 = read_tsv(source_path("train", 1))
    ids = np.asarray(s1["entity_id"].to_pylist())
    country = np.asarray(s1["country"].fill_null("").to_pylist())
    cmap = dict(zip(ids.tolist(), country.tolist()))

    gt_ids, gt_cells = load_gt()
    n_match = np.fromiter((0 if not c else c.count(",") + 1 for c in gt_cells),
                          dtype=np.int32, count=len(gt_cells))
    gt_ids_arr = np.asarray(gt_ids)
    buckets = np.asarray([bucket_of(n) for n in n_match])
    countries = np.asarray([cmap.get(i, "") for i in gt_ids])

    strata = np.char.add(np.char.add(countries, "|"), buckets)
    uniq, inv = np.unique(strata, return_inverse=True)
    N = len(gt_ids)
    print("strata (proportional allocation):")

    val_idx, fit_idx = [], []
    for k, s in enumerate(uniq):
        members = np.flatnonzero(inv == k)
        rng.shuffle(members)
        share = len(members) / N
        nv = max(1, int(round(N_VAL * share)))
        nf = max(1, int(round(N_FIT * share)))
        nv = min(nv, len(members))
        nf = min(nf, len(members) - nv)
        val_idx.append(members[:nv])
        fit_idx.append(members[nv:nv + nf])
        print("  %-14s pool=%-9d val=%-7d fit=%-7d" % (s, len(members), nv, nf))

    val_idx = np.sort(np.concatenate(val_idx))
    fit_idx = np.sort(np.concatenate(fit_idx))
    assert not (set(val_idx.tolist()) & set(fit_idx.tolist())), "val/fit overlap"

    def dump(idx, tag):
        with open(os.path.join(WORK, "%s_truth.tsv" % tag), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
            w.writerow(["source1_entity_id", "matched_entity_ids"])
            for i in idx:
                w.writerow([gt_ids_arr[i], gt_cells[i]])
        with open(os.path.join(WORK, "%s_meta.tsv" % tag), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
            w.writerow(["source1_entity_id", "country", "bucket"])
            for i in idx:
                w.writerow([gt_ids_arr[i], countries[i], buckets[i]])
        print("wrote %s: %d entities, %d true matches"
              % (tag, len(idx), int(n_match[idx].sum())))

    dump(val_idx, "val")
    dump(fit_idx, "fit")
    np.save(os.path.join(WORK, "val_idx.npy"), val_idx)
    np.save(os.path.join(WORK, "fit_idx.npy"), fit_idx)
    print("\nSEED=%d frozen. Singleton rate in val: %.4f"
          % (SEED, float((n_match[val_idx] == 0).mean())))


if __name__ == "__main__":
    main()
