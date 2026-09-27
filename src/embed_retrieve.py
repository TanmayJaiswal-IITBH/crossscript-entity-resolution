"""Dense retrieval: top-K pool records per S1 entity, per source, same country only.

Exact (brute-force) cosine search on the GPU, one (source, country) partition at a
time so only one partition's embeddings are resident. Output:
  work/emb/<tag>_dense.npz
    entity      S1 entity ids, in the order given
    idx2, cos2  (n_q, K) row indices into source 2 (TSV / parquet / index order)
    idx3, cos3  same for source 3
Rows beyond a partition's size, or for a country absent from a pool, are -1.
Runs inside the E: venv.
"""
import argparse
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.csv as pv
import torch

ROOT = os.environ.get("ER_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
EMB = ROOT + "/work/emb"
DIM = 384
QBATCH = 256


def read_col(path, col):
    t = pv.read_csv(path, parse_options=pv.ParseOptions(delimiter="\t"),
                    convert_options=pv.ConvertOptions(
                        column_types={col: pa.string()}, strings_can_be_null=True,
                        null_values=[], include_columns=[col]))
    return t[col].fill_null("").to_pylist()


def load_rows(path, n_total, rows):
    """Read the given (sorted) rows of a raw float16 embedding file."""
    out = np.empty((len(rows), DIM), dtype=np.float16)
    pos = 0
    step = 500_000
    with open(path, "rb") as f:
        for s in range(0, n_total, step):
            e = min(s + step, n_total)
            lo, hi = np.searchsorted(rows, [s, e])
            if lo == hi:
                continue
            f.seek(s * DIM * 2)
            buf = np.fromfile(f, dtype=np.float16, count=(e - s) * DIM).reshape(e - s, DIM)
            out[pos:pos + hi - lo] = buf[rows[lo:hi] - s]
            pos += hi - lo
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--queries", default=None,
                    help="TSV with source1_entity_id; default = every S1 entity")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()
    t0 = time.time()
    data = "%s/dataset/%s/%s_source%%d.tsv" % (ROOT, args.split, args.split)

    s1_ids = read_col(data % 1, "entity_id")
    s1_country = np.load("%s/%s_s1_country.npy" % (EMB, args.split))
    if args.queries:
        want = set(read_col(args.queries, "source1_entity_id"))
        q_rows = np.array([i for i, e in enumerate(s1_ids) if e in want], dtype=np.int64)
    else:
        q_rows = np.arange(len(s1_ids), dtype=np.int64)
    n_q = len(q_rows)
    Q = load_rows("%s/%s_s1.f16" % (EMB, args.split), len(s1_ids), q_rows)
    qc = s1_country[q_rows]
    print("queries: %d  (%.0fs)" % (n_q, time.time() - t0), flush=True)

    out = {"entity": np.array([s1_ids[i] for i in q_rows], dtype=object)}
    for src in (2, 3):
        pc = np.load("%s/%s_s%d_country.npy" % (EMB, args.split, src))
        idx = np.full((n_q, args.k), -1, dtype=np.int32)
        cos = np.full((n_q, args.k), -1.0, dtype=np.float16)
        for c in np.unique(qc):
            p_rows = np.flatnonzero(pc == c)
            q_sel = np.flatnonzero(qc == c)
            if len(p_rows) == 0 or len(q_sel) == 0:
                continue
            P = torch.from_numpy(load_rows("%s/%s_s%d.f16" % (EMB, args.split, src),
                                           len(pc), p_rows)).cuda()
            k = min(args.k, len(p_rows))
            for b in range(0, len(q_sel), QBATCH):
                qi = q_sel[b:b + QBATCH]
                s = torch.from_numpy(Q[qi]).cuda() @ P.T
                v, i = s.topk(k, dim=1)
                idx[qi, :k] = p_rows[i.cpu().numpy()]
                cos[qi, :k] = v.cpu().numpy()
            del P, s
            torch.cuda.empty_cache()
            print("  S%d %-7s pool %8d  queries %8d  (%.0fs)"
                  % (src, c, len(p_rows), len(q_sel), time.time() - t0), flush=True)
        out["idx%d" % src] = idx
        out["cos%d" % src] = cos
    np.savez("%s/%s_dense.npz" % (EMB, args.tag), **out)
    print("saved %s/%s_dense.npz  (%.0fs)" % (EMB, args.tag, time.time() - t0))


if __name__ == "__main__":
    main()
