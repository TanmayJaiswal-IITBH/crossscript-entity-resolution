"""Measure real blocking recall when dense top-K is unioned with TF-IDF candidates.

Uses the full-pool dense retrieval (embed_retrieve.py) for the validation
entities and the TF-IDF candidate sets cached in work/valcache_meta.npz.
Also reports how many NEW candidates each K adds per entity -- the inference
cost of the change. Runs inside the E: venv.
"""
import os
import numpy as np
import pyarrow as pa
import pyarrow.csv as pv

ROOT = os.environ.get("ER_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
WORK = ROOT + "/work"


def read_col(path, cols):
    t = pv.read_csv(path, parse_options=pv.ParseOptions(delimiter="\t"),
                    convert_options=pv.ConvertOptions(
                        column_types={c: pa.string() for c in cols},
                        strings_can_be_null=True, null_values=[], include_columns=cols))
    return [t[c].fill_null("").to_pylist() for c in cols]


def main():
    d = np.load(WORK + "/emb/val_dense.npz", allow_pickle=True)
    ids = {s: read_col(ROOT + "/dataset/train/train_source%d.tsv" % s, ["entity_id"])[0]
           for s in (2, 3)}
    z = np.load(WORK + "/valcache_meta.npz", allow_pickle=True)
    z_ent = list(z["entity"])                      # materialise once, not per row
    tfidf = {}
    for g, c in zip(z["gid"].tolist(), z["cand"].tolist()):
        tfidf.setdefault(z_ent[g], set()).add(c)
    ent_ids, cells = read_col(WORK + "/val_truth.tsv",
                              ["source1_entity_id", "matched_entity_ids"])
    truth = {e: (set(c.split(",")) if c else set()) for e, c in zip(ent_ids, cells)}
    n_true = sum(len(v) for v in truth.values())
    base = sum(len(truth[e] & tfidf.get(e, set())) for e in truth)
    print("TF-IDF blocking recall            %.4f   (%d / %d)" % (base / n_true, base, n_true))

    ents = list(d["entity"])
    # load once: indexing an NpzFile inside the loop re-reads the array from disk
    dense_idx = {s: np.asarray(d["idx%d" % s]) for s in (2, 3)}
    for K in (10, 25, 50, 100):
        dense_hit = union_hit = added = 0
        for qi, e in enumerate(ents):
            dset = set()
            for s in (2, 3):
                for r in dense_idx[s][qi, :K]:
                    if r >= 0:
                        dset.add(ids[s][r])
            t = tfidf.get(e, set())
            tr = truth[e]
            dense_hit += len(tr & dset)
            union_hit += len(tr & (t | dset))
            added += len(dset - t)
        print("K=%-3d per source: dense-only %.4f   UNION %.4f   new candidates/entity %.1f"
              % (K, dense_hit / n_true, union_hit / n_true, added / len(ents)))


if __name__ == "__main__":
    main()
