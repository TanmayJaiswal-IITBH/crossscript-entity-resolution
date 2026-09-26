"""Block 5: read the false merges.

In a precision-weighted metric the false positives are the whole game, so this
classifies every FP produced by the tuned rule into recognisable patterns and
prints worked examples of each.
"""
import os
import sys
import time
import argparse
import collections
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from features2 import base_score, N_ABS, _IX
from pairs import Pools, load_queries
from engine import featurize_chunk

GENERIC_DF = 3000        # a name token this common is not identifying


def token_df(path, col="core"):
    t = pq.read_table(path, columns=[col])[col].to_pylist()
    df = collections.Counter()
    for s in t:
        for w in set(s.split()):
            df[w] += 1
    return df


def classify(f, qcore, df, has_true):
    """-> one pattern label for a false merge.

    Reads the candidate side off the feature vector rather than raw strings:
    a_empty_p/a_empty_q already encode the missing-address cases, and the
    similarity features encode the rest."""
    nt = f[_IX["n_tokset"]]
    at = f[_IX["a_tokset"]]
    nj = f[_IX["n_jacc"]]
    nc = f[_IX["n_contain"]]
    legal = f[_IX["legal_state"]]
    p_empty = f[_IX["a_empty_p"]] > 0.5
    q_empty = f[_IX["a_empty_q"]] > 0.5
    both_addr = not p_empty and not q_empty
    rare = [w for w in qcore.split() if df.get(w, 0) < GENERIC_DF]
    if qcore and not rare:
        return "generic_name"
    if p_empty or q_empty:
        return "empty_address_name_only"
    if at >= 0.85 and nt < 0.60:
        return "same_address_different_name"
    if nt >= 0.90 and at < 0.45 and both_addr:
        return "same_name_different_address(chain)"
    if legal == 2.0 and nt >= 0.80:
        return "legal_form_mismatch"
    if nc >= 0.95 and nj < 0.60:
        return "name_containment(parent/subsidiary)"
    if nt < 0.70 and at < 0.70:
        return "weak_on_both_fields"
    if has_true:
        return "sibling_of_a_true_match"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--truth", default=os.path.join(WORK, "val_truth.tsv"))
    ap.add_argument("--model", default=os.path.join(WORK, "matcher_lgb.txt"))
    ap.add_argument("--rule", default=os.path.join(WORK, "best_rule.npy"))
    ap.add_argument("--kn", type=int, default=60)
    ap.add_argument("--ka", type=int, default=60)
    ap.add_argument("--bn", type=int, default=8000)
    ap.add_argument("--ba", type=int, default=8000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--max-entities", type=int, default=20000)
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    tau, delta, tau2, tsing, kmax = np.load(args.rule)
    kmax = int(kmax)
    print("rule: tau=%.2f delta=%.2f tau2=%.2f tau_single=%.2f kmax=%d"
          % (tau, delta, tau2, tsing, kmax))

    import lightgbm as lgb
    booster = lgb.Booster(model_file=args.model)
    iso = None
    cp = os.path.join(WORK, "matcher_calib.pkl")
    if os.path.exists(cp):
        import pickle
        with open(cp, "rb") as fh:
            iso = pickle.load(fh)

    t = read_tsv(args.truth)
    tids = t["source1_entity_id"].to_pylist()
    cells = t["matched_entity_ids"].fill_null("").to_pylist()
    truth = {i: (set(c.split(",")) if c else set()) for i, c in zip(tids, cells)}
    q = load_queries(args.split, set(tids))
    n_q = min(len(q["entity_id"]), args.max_entities)

    print("building name-token df table ...", flush=True)
    df = token_df(os.path.join(WORK, "%s_s1_norm.parquet" % args.split))

    pools = Pools(args.split)
    pat = collections.Counter()
    examples = collections.defaultdict(list)
    n_fp = n_tp = n_fn = 0
    t0 = time.time()
    with Pool(args.procs) as pool:
        for s in range(0, n_q, args.chunk):
            e = min(s + args.chunk, n_q)
            X, ids, starts, ends = featurize_chunk(pools, q, s, e, args, pool)
            if len(ids) == 0:
                continue
            sc = booster.predict(X)
            if iso is not None:
                sc = iso.predict(sc)
            for i, (a, b) in enumerate(zip(starts, ends)):
                if a >= b:
                    continue
                qi = s + i
                eid = q["entity_id"][qi]
                tr = truth[eid]
                loc = np.argsort(-sc[a:b], kind="stable")
                best = sc[a + loc[0]]
                if best < tsing:
                    n_fn += len(tr)
                    continue
                for rank, j in enumerate(loc[:kmax]):
                    k = a + j
                    if sc[k] < tau or sc[k] < best - delta:
                        continue
                    if rank > 0 and sc[k] < tau2:
                        continue
                    if ids[k] in tr:
                        n_tp += 1
                        continue
                    n_fp += 1
                    lab = classify(X[k], q["core"][qi], df, bool(tr))
                    pat[lab] += 1
                    if len(examples[lab]) < args.examples:
                        examples[lab].append(
                            (eid, q["name"][qi], q["addr"][qi], ids[k],
                             float(sc[k]), float(X[k][_IX["n_tokset"]]),
                             float(X[k][_IX["a_tokset"]]), len(tr)))
            if (s // args.chunk) % 5 == 0:
                print("  %d/%d  %.0fs" % (e, n_q, time.time() - t0), flush=True)

    print("\npredicted pairs: tp=%d fp=%d   micro precision %.4f"
          % (n_tp, n_fp, n_tp / max(n_tp + n_fp, 1)))
    print("\n=== FALSE-MERGE PATTERNS (%d total) ===" % n_fp)
    for lab, c in pat.most_common():
        print("  %-38s %7d  %5.1f%%" % (lab, c, 100 * c / max(n_fp, 1)))

    # resolve pool ids -> readable records for the examples
    want = set()
    for v in examples.values():
        for row in v:
            want.add(row[3])
    disp = {}
    for src in (2, 3):
        pt = pq.read_table(os.path.join(WORK, "%s_s%d_norm.parquet" % (args.split, src)),
                           columns=["entity_id", "name", "addr"])
        pid = pt["entity_id"].to_pylist()
        keep = [i for i, x in enumerate(pid) if x in want]
        for i in keep:
            disp[pid[i]] = (pt["name"][i].as_py(), pt["addr"][i].as_py())
        del pt, pid
    print("\n=== EXAMPLES ===")
    for lab, c in pat.most_common():
        print("\n--- %s (%d) ---" % (lab, c))
        for (eid, qn, qa, cid, sc_, nt, at, ntrue) in examples[lab]:
            pn, pa = disp.get(cid, ("?", "?"))
            print("  S1 %-14s %r" % (eid, qn))
            print("     addr %r   (entity has %d true matches)" % (qa, ntrue))
            print("  FP %-14s %r" % (cid, pn))
            print("     addr %r" % pa)
            print("     score=%.3f  n_tokset=%.2f  a_tokset=%.2f" % (sc_, nt, at))


if __name__ == "__main__":
    main()
