"""Blocking v2: IDF-weighted token retrieval with a per-query posting budget.

Why this replaces the composite-key version
-------------------------------------------
Composite keys (sorted name tokens, digit signature + state, ...) are brittle:
one inserted word, one leading zero or one reordered address component and the
key changes completely.  Diagnostics on the missed pairs showed exactly that,
plus a df cutoff that was throwing away the only token a pair had in common.

Here every record is a bag of tokens:
    n<country><token>   core name tokens
    a<country><token>   address tokens (alphabetic, len >= 3)
    d<country><run>     digit runs of the address, leading zeros stripped
    g<country><4gram>   char 4-grams of the joined core name (typo tolerance)

A query scores every pool record that shares a token, weighted by IDF, and keeps
the top K.  Cost is bounded per query by walking the query's tokens rarest-first
and stopping once the posting budget is spent -- so a query never pays for the
token 'private', but always pays for 'byadarahalli'.
"""
import os
import re
import sys
import time
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK

DIGRUN = re.compile(r"\d+")
MAX_DF = 60_000          # tokens more common than this are not indexed at all
NGRAM_N = 4
MAX_NGRAM = 16
MAX_NAME_TOK = 10
MAX_ADDR_TOK = 12
MAX_DIG = 5


def gen_tokens(country, core, addr, ngrams=True):
    c = country[0] if country else "?"
    out = []
    ct = core.split()
    seen = set()
    for t in ct[:MAX_NAME_TOK]:
        if len(t) >= 2:
            k = "n" + c + t
            if k not in seen:
                seen.add(k)
                out.append(k)
    if ngrams:
        j = "".join(ct)
        if len(j) >= NGRAM_N:
            for i in range(min(len(j) - NGRAM_N + 1, MAX_NGRAM)):
                k = "g" + c + j[i:i + NGRAM_N]
                if k not in seen:
                    seen.add(k)
                    out.append(k)
    na = 0
    for t in addr.split():
        if na >= MAX_ADDR_TOK:
            break
        if len(t) >= 3 and not t.isdigit():
            k = "a" + c + t
            if k not in seen:
                seen.add(k)
                out.append(k)
                na += 1
    nd = 0
    for r in DIGRUN.findall(addr):
        if nd >= MAX_DIG:
            break
        r = r.lstrip("0") or "0"
        if len(r) >= 2:
            k = "d" + c + r
            if k not in seen:
                seen.add(k)
                out.append(k)
                nd += 1
    return out


def _tok_chunk(args):
    start, country, core, addr = args
    hs, ix = [], []
    for i in range(len(core)):
        r = start + i
        for k in gen_tokens(country[i], core[i], addr[i]):
            hs.append(hash(k) & 0x7FFFFFFFFFFFFFFF)
            ix.append(r)
    return np.array(hs, dtype=np.uint64), np.array(ix, dtype=np.int32)


def build_index(split, src, procs=8, chunk=100_000, force=False):
    tag = "%s_s%d" % (split, src)
    dst = os.path.join(WORK, "tok_%s.npz" % tag)
    if os.path.exists(dst) and not force:
        print("  skip (exists):", os.path.basename(dst))
        return dst
    t0 = time.time()
    tb = pq.read_table(os.path.join(WORK, "%s_norm.parquet" % tag),
                       columns=["country", "core", "addr"])
    n_rec = tb.num_rows
    country = tb["country"].to_pylist()
    core = tb["core"].to_pylist()
    addr = tb["addr"].to_pylist()
    del tb
    tasks = [(s, country[s:s + chunk], core[s:s + chunk], addr[s:s + chunk])
             for s in range(0, n_rec, chunk)]
    del country, core, addr
    with Pool(procs) as p:
        res = p.map(_tok_chunk, tasks, chunksize=1)
    del tasks
    hs = np.concatenate([r[0] for r in res])
    ix = np.concatenate([r[1] for r in res])
    del res
    print("  %s: %d records -> %d postings (%.0fs)" % (tag, n_rec, len(hs), time.time() - t0))
    order = np.argsort(hs, kind="stable")
    hs = hs[order]
    ix = ix[order]
    del order
    uniq, starts, counts = np.unique(hs, return_index=True, return_counts=True)
    del hs
    keep = counts <= MAX_DF
    n_drop = int((~keep).sum())
    uniq, starts, counts = uniq[keep], starts[keep], counts[keep]
    total = int(counts.sum())
    cum0 = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=cum0[1:])
    take = (np.repeat(starts.astype(np.int64), counts)
            + np.arange(total, dtype=np.int64) - np.repeat(cum0, counts))
    post = ix[take]
    del ix, take, cum0
    offs = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offs[1:])
    np.savez(dst, uniq=uniq, offs=offs, post=post, n_rec=np.int64(n_rec))
    print("  -> %d tokens (%d too common, dropped), %d postings, %.0fs, %.0f MB"
          % (len(uniq), n_drop, total, time.time() - t0, os.path.getsize(dst) / 1e6))
    return dst


class TokIndex(object):
    def __init__(self, path):
        z = np.load(path)
        self.uniq = z["uniq"]
        self.offs = z["offs"]
        self.post = z["post"]
        self.n_rec = int(z["n_rec"])
        self.df = (self.offs[1:] - self.offs[:-1]).astype(np.float32)
        self.idf = np.log(self.n_rec / np.maximum(self.df, 1.0)).astype(np.float32)


def _q_tokens(country, core, addr, ngrams=True):
    hs, rw = [], []
    for i in range(len(core)):
        for k in gen_tokens(country[i], core[i], addr[i], ngrams=ngrams):
            hs.append(hash(k) & 0x7FFFFFFFFFFFFFFF)
            rw.append(i)
    return np.array(hs, dtype=np.uint64), np.array(rw, dtype=np.int64)


def retrieve(idx, country, core, addr, k=40, budget=1200, ngrams=True):
    """-> (query_row, pool_row, score) arrays, top-k per query by IDF overlap."""
    n_q = len(core)
    qh, qr = _q_tokens(country, core, addr, ngrams=ngrams)
    if len(qh) == 0:
        return (np.empty(0, np.int64), np.empty(0, np.int32), np.empty(0, np.float32))
    pos = np.searchsorted(idx.uniq, qh)
    np.minimum(pos, len(idx.uniq) - 1, out=pos)
    hit = idx.uniq[pos] == qh
    pos, qr = pos[hit], qr[hit]
    df = idx.df[pos]
    w = idx.idf[pos]

    # rarest-first within each query, then keep tokens while the budget lasts
    order = np.lexsort((df, qr))
    pos, qr, df, w = pos[order], qr[order], df[order], w[order]
    run_start = np.searchsorted(qr, np.arange(n_q, dtype=np.int64), side="left")
    csum = np.cumsum(df)
    prev = np.where(run_start[qr] > 0, csum[run_start[qr] - 1], 0.0)
    spent = csum - prev
    keep = (spent - df) < budget
    pos, qr, w = pos[keep], qr[keep], w[keep]
    if len(pos) == 0:
        return (np.empty(0, np.int64), np.empty(0, np.int32), np.empty(0, np.float32))

    cnt = (idx.offs[pos + 1] - idx.offs[pos]).astype(np.int64)
    total = int(cnt.sum())
    qrep = np.repeat(qr, cnt)
    wrep = np.repeat(w, cnt)
    base = np.repeat(idx.offs[pos], cnt)
    cum0 = np.zeros(len(cnt), dtype=np.int64)
    np.cumsum(cnt[:-1], out=cum0[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(cum0, cnt)
    cand = idx.post[base + within]
    del base, within, cum0, cnt

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
    score = np.add.reduceat(wrep, starts)
    pair = comb[starts]
    del comb, wrep, brk, starts
    q = pair // idx.n_rec
    c = (pair - q * idx.n_rec).astype(np.int32)
    o = np.lexsort((-score, q))
    q, c, score = q[o], c[o], score[o]
    first = np.searchsorted(q, np.arange(n_q, dtype=np.int64), side="left")
    rank = np.arange(len(q), dtype=np.int64) - first[q]
    sel = rank < k
    return q[sel], c[sel], score[sel].astype(np.float32)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    for src in (2, 3):
        build_index(split, src)
