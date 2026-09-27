"""Blocking v3: per-field, length-normalised TF-IDF retrieval, unioned.

Two changes over v2, both driven by the miss diagnostics:

1. **Length normalisation.**  v2 ranked by a raw sum of IDF, so a pool record
   with a sprawling address scored well against everything.  v3 divides by the
   record's field norm (binary-tf cosine), which is what pushes genuinely
   similar records above merely verbose ones.

2. **Per-field retrieval, unioned.**  A true pair often agrees on exactly one
   field: the address is empty (3.3% of the pool) so only the name can match, or
   the name is transliterated into another script so only the address can match.
   Summing both fields into one score buries those pairs.  Retrieving top-K on
   the name and top-K on the address separately, then taking the union, recovers
   them.

The postings index from block2 is reused unchanged; only the scoring and the
query-side token selection differ, plus two precomputed norm vectors per source.
"""
import os
import sys
import time
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK
from block2 import gen_tokens, TokIndex

NAME_PREFIX = ("n", "g")
ADDR_PREFIX = ("a", "d")


def split_tokens(toks):
    nm = [t for t in toks if t[0] in NAME_PREFIX]
    ad = [t for t in toks if t[0] in ADDR_PREFIX]
    return nm, ad


def _hash_arr(toks):
    return np.fromiter((hash(t) & 0x7FFFFFFFFFFFFFFF for t in toks),
                       dtype=np.uint64, count=len(toks))


def _norm_chunk(args):
    """Worker: token hashes + row + field flag for one slice of records.

    Only the token generation is farmed out; the parent does the vectorised
    searchsorted, which is cheap by comparison."""
    country, core, addr = args
    hs, rows, isname = [], [], []
    for i in range(len(core)):
        for t in gen_tokens(country[i], core[i], addr[i]):
            hs.append(hash(t) & 0x7FFFFFFFFFFFFFFF)
            rows.append(i)
            isname.append(t[0] in NAME_PREFIX)
    return (np.array(hs, dtype=np.uint64), np.array(rows, dtype=np.int64),
            np.array(isname, dtype=bool))


def build_norms(split, src, chunk=100_000, procs=8, force=False):
    """Per-record ||d|| for the name field and the address field separately."""
    tag = "%s_s%d" % (split, src)
    dst = os.path.join(WORK, "norm_%s.npz" % tag)
    if os.path.exists(dst) and not force:
        print("  skip (exists):", os.path.basename(dst))
        return dst
    t0 = time.time()
    idx = TokIndex(os.path.join(WORK, "tok_%s.npz" % tag))
    tb = pq.read_table(os.path.join(WORK, "%s_norm.parquet" % tag),
                       columns=["country", "core", "addr"])
    n = tb.num_rows
    country = tb["country"].to_pylist()
    core = tb["core"].to_pylist()
    addr = tb["addr"].to_pylist()
    del tb
    nn = np.zeros(n, dtype=np.float32)
    na = np.zeros(n, dtype=np.float32)
    starts = list(range(0, n, chunk))
    tasks = [(country[s:s + chunk], core[s:s + chunk], addr[s:s + chunk]) for s in starts]
    del country, core, addr
    with Pool(procs) as p:
        for s, (hs, rows, isname) in zip(starts, p.imap(_norm_chunk, tasks, chunksize=1)):
            if len(hs) == 0:
                continue
            e = min(s + chunk, n)
            pos = np.searchsorted(idx.uniq, hs)
            np.minimum(pos, len(idx.uniq) - 1, out=pos)
            hit = idx.uniq[pos] == hs
            pos, rows, isname = pos[hit], rows[hit], isname[hit]
            w = idx.idf[pos] ** 2
            nn[s:e] = np.bincount(rows[isname], weights=w[isname], minlength=e - s)
            na[s:e] = np.bincount(rows[~isname], weights=w[~isname], minlength=e - s)
    np.sqrt(nn, out=nn)
    np.sqrt(na, out=na)
    np.maximum(nn, 1e-3, out=nn)
    np.maximum(na, 1e-3, out=na)
    np.savez(dst, name=nn, addr=na)
    print("  norms %s in %.0fs" % (tag, time.time() - t0), flush=True)
    return dst


class FieldIndex(object):
    def __init__(self, split, src):
        tag = "%s_s%d" % (split, src)
        self.idx = TokIndex(os.path.join(WORK, "tok_%s.npz" % tag))
        z = np.load(os.path.join(WORK, "norm_%s.npz" % tag))
        self.norm = {"name": z["name"], "addr": z["addr"]}
        self.n_rec = self.idx.n_rec


QDF_CAP = 8000          # a query never spends postings on a token commoner than this


def _select(idx, qh, qr, n_q, budget):
    """Walk each query's tokens rarest-first, keeping them while the posting
    budget lasts.  Both caps matter: without QDF_CAP a single very common token
    can blow the budget by an order of magnitude in one step, because the budget
    is only checked *before* a token is admitted."""
    pos = np.searchsorted(idx.uniq, qh)
    np.minimum(pos, len(idx.uniq) - 1, out=pos)
    hit = idx.uniq[pos] == qh
    pos, qr = pos[hit], qr[hit]
    if len(pos) == 0:
        return pos, qr, None
    # float64: the running sum below spans a whole chunk of queries and passes
    # float32's exact-integer range (~1.7e7), which made the budget cutoff for a
    # query depend on which other queries shared its chunk.
    df = idx.df[pos].astype(np.float64)
    ok = df <= QDF_CAP
    pos, qr, df = pos[ok], qr[ok], df[ok]
    if len(pos) == 0:
        return pos, qr, None
    order = np.lexsort((df, qr))
    pos, qr, df = pos[order], qr[order], df[order]
    run_start = np.searchsorted(qr, np.arange(n_q, dtype=np.int64), side="left")
    csum = np.cumsum(df)
    prev = np.where(run_start[qr] > 0, csum[run_start[qr] - 1], 0.0)
    spent = csum - prev
    is_first = np.arange(len(qr), dtype=np.int64) == run_start[qr]
    keep = (spent <= budget) | is_first        # always keep the single rarest token
    return pos[keep], qr[keep], None


def retrieve_field(fidx, qtoks, field, n_q, k=30, budget=2500):
    """qtoks: list (per query row) of that field's token strings."""
    idx = fidx.idx
    hs, rw = [], []
    for i, ts in enumerate(qtoks):
        for t in ts:
            hs.append(hash(t) & 0x7FFFFFFFFFFFFFFF)
            rw.append(i)
    if not hs:
        return np.empty(0, np.int64), np.empty(0, np.int32), np.empty(0, np.float32)
    qh = np.array(hs, dtype=np.uint64)
    qr = np.array(rw, dtype=np.int64)
    pos, qr, _ = _select(idx, qh, qr, n_q, budget)
    if len(pos) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int32), np.empty(0, np.float32)
    w = idx.idf[pos] ** 2
    cnt = (idx.offs[pos + 1] - idx.offs[pos]).astype(np.int64)
    total = int(cnt.sum())
    qrep = np.repeat(qr, cnt)
    wrep = np.repeat(w, cnt)
    base = np.repeat(idx.offs[pos], cnt)
    cum0 = np.zeros(len(cnt), dtype=np.int64)
    np.cumsum(cnt[:-1], out=cum0[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(cum0, cnt)
    cand = idx.post[base + within]
    del base, within, cum0, cnt, pos, w
    comb = qrep * np.int64(idx.n_rec) + cand.astype(np.int64)
    del qrep, cand
    o = np.argsort(comb, kind="stable")
    comb = comb[o]
    wrep = wrep[o]
    del o
    brk = np.empty(len(comb), dtype=bool)
    brk[0] = True
    np.not_equal(comb[1:], comb[:-1], out=brk[1:])
    starts = np.flatnonzero(brk)
    score = np.add.reduceat(wrep, starts).astype(np.float32)
    pair = comb[starts]
    del comb, wrep, brk, starts
    q = pair // idx.n_rec
    c = (pair - q * idx.n_rec).astype(np.int32)
    del pair
    score /= fidx.norm[field][c]
    o = np.lexsort((-score, q))
    q, c, score = q[o], c[o], score[o]
    first = np.searchsorted(q, np.arange(n_q, dtype=np.int64), side="left")
    rank = np.arange(len(q), dtype=np.int64) - first[q]
    sel = rank < k
    return q[sel], c[sel], score[sel]


def retrieve_union(fidx, country, core, addr, k_name=30, k_addr=30,
                   budget_name=2500, budget_addr=2500):
    n_q = len(core)
    nm_toks, ad_toks = [], []
    for i in range(n_q):
        a, b = split_tokens(gen_tokens(country[i], core[i], addr[i]))
        nm_toks.append(a)
        ad_toks.append(b)
    qa, ca, sa = retrieve_field(fidx, nm_toks, "name", n_q, k_name, budget_name)
    qb, cb, sb = retrieve_field(fidx, ad_toks, "addr", n_q, k_addr, budget_addr)
    return (qa, ca, sa), (qb, cb, sb)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    for src in (2, 3):
        build_norms(split, src)
