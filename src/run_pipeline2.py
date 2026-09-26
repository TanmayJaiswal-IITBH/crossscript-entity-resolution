"""Day-2 end-to-end inference: candidates -> 52 features -> model -> per-entity rule.

Shard with --shard i/N to run several processes over disjoint query ranges, then
--merge to concatenate them in query order into the two submission files.

    python src/run_pipeline2.py --split test --shard 0/2 &
    python src/run_pipeline2.py --split test --shard 1/2 &
    python src/run_pipeline2.py --split test --merge --out-dir output
"""
import os
import sys
import time
import pickle
import argparse
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from pairs import Pools, load_queries
from engine import featurize_chunk


def shard_path(tag, split, i):
    return os.path.join(WORK, "out_%s_%s_sh%d.tsv" % (tag, split, i))


def apply_rule(sc, tau, delta, tau2, tsing, kmax):
    """-> indices of the accepted candidates, best first."""
    loc = np.argsort(-sc, kind="stable")
    best = sc[loc[0]]
    if best < tsing:
        return []
    out = []
    for rank, j in enumerate(loc[:kmax]):
        v = sc[j]
        if v < tau or v < best - delta:
            break
        if rank > 0 and v < tau2:
            break
        out.append(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--truth", default=None)
    ap.add_argument("--model", default=os.path.join(WORK, "matcher_lgb.txt"))
    ap.add_argument("--rule", default=os.path.join(WORK, "best_rule.npy"))
    ap.add_argument("--kn", type=int, default=30)
    ap.add_argument("--ka", type=int, default=30)
    ap.add_argument("--bn", type=int, default=4000)
    ap.add_argument("--ba", type=int, default=4000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    subset = None
    if args.truth:
        subset = set(read_tsv(args.truth)["source1_entity_id"].to_pylist())
    q = load_queries(args.split, subset)
    n_q = len(q["entity_id"])
    si, sn = (int(x) for x in args.shard.split("/"))

    if args.merge:
        out_dir = args.out_dir or WORK
        os.makedirs(out_dir, exist_ok=True)
        mp = os.path.join(out_dir, "matching_results.tsv")
        cp = os.path.join(out_dir, "candidate_pairs.tsv")
        n_pred = n_empty = n_cand = 0
        with open(mp, "w", encoding="utf-8", newline="\n") as fm, \
                open(cp, "w", encoding="utf-8", newline="\n") as fc:
            fm.write("source1_entity_id\tmatched_entity_ids\n")
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
            written = 0
            for i in range(sn):
                with open(shard_path(args.tag, args.split, i), encoding="utf-8") as fh:
                    for line in fh:
                        eid, matched, cands = line.rstrip("\n").split("\t")
                        fm.write(eid + "\t" + matched + "\n")
                        fc.write(eid + "\t" + cands + "\n")
                        written += 1
                        n_cand += cands.count(",") + 1 if cands else 0
                        if matched:
                            n_pred += matched.count(",") + 1
                        else:
                            n_empty += 1
        print("merged %d rows (expected %d)" % (written, n_q))
        assert written == n_q, "shard rows %d != queries %d" % (written, n_q)
        print("  candidates/query %.1f  matches/query %.2f  empty %.2f%%"
              % (n_cand / n_q, n_pred / n_q, 100 * n_empty / n_q))
        print("wrote %s\nwrote %s" % (mp, cp))
        return

    tau, delta, tau2, tsing, kmax = np.load(args.rule)
    kmax = int(kmax)
    import lightgbm as lgb
    booster = lgb.Booster(model_file=args.model)
    iso = None
    cpk = os.path.join(WORK, "matcher_calib.pkl")
    if os.path.exists(cpk):
        with open(cpk, "rb") as fh:
            iso = pickle.load(fh)
    lo = n_q * si // sn
    hi = n_q * (si + 1) // sn
    print("shard %d/%d: entities [%d, %d)  rule tau=%.2f delta=%.2f tau2=%.2f "
          "tsing=%.2f kmax=%d" % (si, sn, lo, hi, tau, delta, tau2, tsing, kmax),
          flush=True)

    pools = Pools(args.split)
    t0 = time.time()
    with open(shard_path(args.tag, args.split, si), "w", encoding="utf-8",
              newline="\n") as fh, Pool(args.procs) as pool:
        for s in range(lo, hi, args.chunk):
            e = min(s + args.chunk, hi)
            X, ids, starts, ends = featurize_chunk(pools, q, s, e, args, pool)
            if len(ids) == 0:
                for i in range(e - s):
                    fh.write(q["entity_id"][s + i] + "\t\t\n")
                continue
            sc = booster.predict(X)
            if iso is not None:
                sc = iso.predict(sc)
            for i, (a, b) in enumerate(zip(starts, ends)):
                eid = q["entity_id"][s + i]
                if a >= b:
                    fh.write(eid + "\t\t\n")
                    continue
                sub = sc[a:b]
                sel = apply_rule(sub, tau, delta, tau2, tsing, kmax)
                fh.write(eid + "\t" + ",".join(ids[a + j] for j in sel)
                         + "\t" + ",".join(ids[a:b]) + "\n")
            if (s - lo) // args.chunk % 20 == 0:
                el = time.time() - t0
                done = e - lo
                print("  %d/%d  %.0fs (eta %.0fs)"
                      % (done, hi - lo, el, el * (hi - lo - done) / max(done, 1)),
                      flush=True)
    print("shard %d done in %.0fs" % (si, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
