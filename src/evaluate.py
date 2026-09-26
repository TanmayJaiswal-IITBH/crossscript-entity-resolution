"""Macro F_0.5 scorer, matching the challenge spec exactly.

Per Source-1 entity:
    precision = |pred & true| / |pred|,  recall = |pred & true| / |true|
    F_0.5     = (1.25 * P * R) / (0.25 * P + R)
Singleton (true set empty): 1.0 if prediction is empty, else 0.0.
An entity with true matches but an empty prediction scores 0.0.
The reported score is the mean over *all* Source-1 entities in the eval set.

Usage:
    python src/evaluate.py --pred <pred.tsv> --truth <truth.tsv> [--breakdown <meta.tsv>]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_tsv


def _parse(cell):
    if not cell:
        return frozenset()
    return frozenset(x for x in (p.strip() for p in cell.split(",")) if x)


def f_beta_half(pred, true):
    """F_0.5 for one entity, given two sets of ids."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    return (1.25 * p * r) / (0.25 * p + r)


def score_dicts(pred_map, true_map, return_per_entity=False):
    """Average over every key of true_map (the eval universe)."""
    scores = {}
    for eid, true in true_map.items():
        scores[eid] = f_beta_half(pred_map.get(eid, frozenset()), true)
    macro = sum(scores.values()) / len(scores) if scores else 0.0
    return (macro, scores) if return_per_entity else macro


def load_pairs(path, id_col=None, list_col=None):
    t = read_tsv(path)
    cols = t.column_names
    id_col = id_col or cols[0]
    list_col = list_col or cols[1]
    ids = t[id_col].to_pylist()
    cells = t[list_col].fill_null("").to_pylist()
    return {i: _parse(c) for i, c in zip(ids, cells)}


def report(pred_map, true_map, meta=None, label=""):
    """meta: optional {entity_id: (country, bucket)} for a stratified breakdown."""
    macro, per = score_dicts(pred_map, true_map, return_per_entity=True)
    n = len(true_map)
    tp = fp = fn = 0
    n_empty_pred = n_singleton = n_singleton_hit = 0
    for eid, true in true_map.items():
        pred = pred_map.get(eid, frozenset())
        tp += len(pred & true)
        fp += len(pred - true)
        fn += len(true - pred)
        if not pred:
            n_empty_pred += 1
        if not true:
            n_singleton += 1
            if not pred:
                n_singleton_hit += 1
    mp = tp / (tp + fp) if tp + fp else 0.0
    mr = tp / (tp + fn) if tp + fn else 0.0
    print("=" * 66)
    print("MACRO F_0.5 %s: %.5f   (n=%d entities)" % (label, macro, n))
    print("-" * 66)
    print("  micro precision %.4f   micro recall %.4f   (pairs tp=%d fp=%d fn=%d)"
          % (mp, mr, tp, fp, fn))
    print("  predicted-empty %d (%.2f%%)   singletons %d   singleton hit %d (%.2f%%)"
          % (n_empty_pred, 100 * n_empty_pred / n, n_singleton, n_singleton_hit,
             100 * n_singleton_hit / n_singleton if n_singleton else 0.0))
    if meta:
        print("-" * 66)
        groups = {}
        for eid, sc in per.items():
            key = meta.get(eid)
            if key is None:
                continue
            groups.setdefault(key, []).append(sc)
        for key in sorted(groups):
            v = groups[key]
            print("  %-22s n=%-8d F0.5=%.5f" % (str(key), len(v), sum(v) / len(v)))
    print("=" * 66)
    return macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--breakdown", default=None,
                    help="optional tsv with columns source1_entity_id, country, bucket")
    args = ap.parse_args()
    pred_map = load_pairs(args.pred)
    true_map = load_pairs(args.truth)
    meta = None
    if args.breakdown:
        t = read_tsv(args.breakdown)
        meta = dict(zip(t["source1_entity_id"].to_pylist(),
                        zip(t["country"].to_pylist(), t["bucket"].to_pylist())))
    report(pred_map, true_map, meta)


if __name__ == "__main__":
    main()
