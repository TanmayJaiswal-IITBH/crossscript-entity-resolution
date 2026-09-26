"""End-to-end: blocking -> pair features -> scoring -> submission files.

Runs one pool source at a time (S2, then S3) so only one 5M-record pool is ever
resident.  Per-source candidate lists are streamed to disk in query order and
zip-merged at the end, which keeps peak memory flat regardless of query count.

    python src/run_pipeline.py --split train --truth work/val_truth.tsv
    python src/run_pipeline.py --split test --out-dir output
"""
import os
import sys
import time
import argparse
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, OUT, read_tsv
from block3 import FieldIndex, retrieve_union
from features import featurize_block, baseline_score, FEATURE_NAMES

_MODEL = {"booster": None}


def score_pairs(feats):
    """LightGBM probability when a model is loaded, else the hand-weighted blend."""
    b = _MODEL["booster"]
    if b is None:
        return baseline_score(feats)
    return b.predict(feats, num_iteration=b.best_iteration).astype(np.float32)

_POOL = {}


def _feat_worker(args):
    qc, qa, pc, pa, bn, ba = args
    return featurize_block(qc, qa, pc, pa, bn, ba)


def load_queries(split, subset_ids=None):
    tb = pq.read_table(os.path.join(WORK, "%s_s1_norm.parquet" % split))
    if subset_ids is not None:
        ids = tb["entity_id"].to_pylist()
        tb = tb.take([i for i, e in enumerate(ids) if e in subset_ids])
    return {c: tb[c].to_pylist() for c in tb.column_names}


def pass_over_source(split, src, q, args, tmp_path, procs=8):
    """Retrieve + score candidates from one pool; stream one line per query."""
    t0 = time.time()
    fi = FieldIndex(split, src)
    pt = pq.read_table(os.path.join(WORK, "%s_s%d_norm.parquet" % (split, src)),
                       columns=["entity_id", "core", "addr"])
    pids = pt["entity_id"].to_pylist()
    pcore = pt["core"].to_pylist()
    paddr = pt["addr"].to_pylist()
    del pt
    n_q = len(q["entity_id"])
    n_written = 0
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh, Pool(procs) as pool:
        for s in range(0, n_q, args.chunk):
            e = min(s + args.chunk, n_q)
            (qa_, ca_, sa_), (qb_, cb_, sb_) = retrieve_union(
                fi, q["country"][s:e], q["core"][s:e], q["addr"][s:e],
                k_name=args.kn, k_addr=args.ka,
                budget_name=args.bn, budget_addr=args.ba)
            # merge the two rankings into one candidate table with both scores
            m = {}
            for a, b, c in zip(qa_.tolist(), ca_.tolist(), sa_.tolist()):
                m[(a, b)] = (c, 0.0)
            for a, b, c in zip(qb_.tolist(), cb_.tolist(), sb_.tolist()):
                k = (a, b)
                prev = m.get(k)
                m[k] = (prev[0], c) if prev else (0.0, c)
            if not m:
                for _ in range(e - s):
                    fh.write("\t\n")
                continue
            keys = list(m)
            qi = np.fromiter((k[0] for k in keys), dtype=np.int64, count=len(keys))
            pi = np.fromiter((k[1] for k in keys), dtype=np.int64, count=len(keys))
            bn = np.fromiter((m[k][0] for k in keys), dtype=np.float32, count=len(keys))
            ba = np.fromiter((m[k][1] for k in keys), dtype=np.float32, count=len(keys))
            qc_l = [q["core"][s + i] for i in qi]
            qa_l = [q["addr"][s + i] for i in qi]
            pc_l = [pcore[j] for j in pi]
            pa_l = [paddr[j] for j in pi]
            nb = max(1, len(keys) // (procs * 4))
            tasks = [(qc_l[i:i + nb], qa_l[i:i + nb], pc_l[i:i + nb], pa_l[i:i + nb],
                      bn[i:i + nb], ba[i:i + nb]) for i in range(0, len(keys), nb)]
            feats = np.concatenate(pool.map(_feat_worker, tasks, chunksize=1))
            score = score_pairs(feats)
            # group by query row
            order = np.lexsort((-score, qi))
            qi, pi, score = qi[order], pi[order], score[order]
            first = np.searchsorted(qi, np.arange(e - s, dtype=np.int64), side="left")
            last = np.searchsorted(qi, np.arange(e - s, dtype=np.int64), side="right")
            for r in range(e - s):
                a, b = first[r], last[r]
                if a == b:
                    fh.write("\t\n")
                    continue
                cap = min(b - a, args.cap_per_src)
                ids = [pids[j] for j in pi[a:a + cap]]
                scs = score[a:a + cap]
                n_written += cap
                fh.write(",".join(ids) + "\t"
                         + ",".join("%.5f" % v for v in scs) + "\n")
            if args.verbose and (s // args.chunk) % 20 == 0:
                print("    S%d %d/%d  %.0fs" % (src, e, n_q, time.time() - t0), flush=True)
    print("  S%d pass: %.0fs, %d candidates written" % (src, time.time() - t0, n_written),
          flush=True)
    del fi, pids, pcore, paddr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--truth", default=None, help="restrict queries to this truth file")
    ap.add_argument("--kn", type=int, default=25)
    ap.add_argument("--ka", type=int, default=25)
    ap.add_argument("--bn", type=int, default=2500)
    ap.add_argument("--ba", type=int, default=2500)
    ap.add_argument("--cap-per-src", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--threshold", type=float, default=0.82)
    ap.add_argument("--rel", type=float, default=0.88,
                    help="also require score >= rel * this entity's best score")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only-src", type=int, default=0,
                    help="run just this pool pass and exit (2 or 3)")
    ap.add_argument("--model", default=None,
                    help="path to a LightGBM model; omit to use the hand-weighted blend")
    ap.add_argument("--merge-only", action="store_true",
                    help="skip retrieval, merge existing candidate streams")
    args = ap.parse_args()

    if args.model:
        import lightgbm as lgb
        _MODEL["booster"] = lgb.Booster(model_file=args.model)
        print("loaded model %s" % args.model, flush=True)

    subset = None
    if args.truth:
        t = read_tsv(args.truth)
        subset = set(t["source1_entity_id"].to_pylist())
    q = load_queries(args.split, subset)
    n_q = len(q["entity_id"])
    print("queries: %d (split=%s)" % (n_q, args.split), flush=True)

    tmps = [os.path.join(WORK, "cand_%s_%s_s%d.tsv" % (args.tag, args.split, s))
            for s in (2, 3)]
    if not args.merge_only:
        todo = (args.only_src,) if args.only_src else (2, 3)
        for src in todo:
            pass_over_source(args.split, src, q, args,
                             tmps[0] if src == 2 else tmps[1], procs=args.procs)
        if args.only_src:
            print("only-src %d done; run --merge-only to produce the outputs" % args.only_src)
            return

    out_dir = args.out_dir or WORK
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "matching_results.tsv")
    cpath = os.path.join(out_dir, "candidate_pairs.tsv")
    n_pred = n_empty = n_cand = 0
    with open(tmps[0], encoding="utf-8") as f2, open(tmps[1], encoding="utf-8") as f3, \
            open(mpath, "w", encoding="utf-8", newline="\n") as fm, \
            open(cpath, "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(n_q):
            eid = q["entity_id"][i]
            cands, scores = [], []
            for fh in (f2, f3):
                line = fh.readline().rstrip("\n")
                ids_s, _, sc_s = line.partition("\t")
                if ids_s:
                    cands.extend(ids_s.split(","))
                    scores.extend(float(x) for x in sc_s.split(","))
            n_cand += len(cands)
            fc.write(eid + "\t" + ",".join(cands) + "\n")
            top = max(scores) if scores else 0.0
            lo = max(args.threshold, args.rel * top) if args.rel > 0 else args.threshold
            sel = [c for c, s in zip(cands, scores) if s >= lo]
            if sel:
                n_pred += len(sel)
            else:
                n_empty += 1
            fm.write(eid + "\t" + ",".join(sel) + "\n")
    print("wrote %s and %s" % (mpath, cpath))
    print("  candidates/query %.1f   predicted matches %d (%.2f/query)   empty preds %d (%.2f%%)"
          % (n_cand / n_q, n_pred, n_pred / n_q, n_empty, 100 * n_empty / n_q))


if __name__ == "__main__":
    main()
