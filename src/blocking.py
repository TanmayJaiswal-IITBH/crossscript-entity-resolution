"""Inverted-index blocking over hashed composite keys.

Index layout (per pool source): all (key_hash, record_idx) pairs sorted by
key_hash, plus the unique key boundaries.  Buckets larger than MAX_DF are
dropped -- a key shared by thousands of records carries no evidence and would
dominate the cost.

Retrieval: a query emits the same key families; every pool record sharing at
least one key is a candidate, ranked by how many distinct keys it shares.
"""
import os
import sys
import time
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK
from keys import gen_keys

MAX_DF = 400           # buckets bigger than this are dropped from the index
MASK = np.uint64(0x7FFFFFFFFFFFFFFF)


def _keys_chunk(args):
    start, country, core, addr, digits, state = args
    hs, ix = [], []
    for i in range(len(core)):
        ks = gen_keys(country[i], core[i], addr[i], digits[i], state[i])
        if not ks:
            continue
        r = start + i
        for k in ks:
            hs.append(hash(k) & 0x7FFFFFFFFFFFFFFF)
            ix.append(r)
    return np.array(hs, dtype=np.uint64), np.array(ix, dtype=np.int32)


def gen_key_arrays(tb, procs=8, chunk=100_000):
    country = tb["country"].to_pylist()
    core = tb["core"].to_pylist()
    addr = tb["addr"].to_pylist()
    digits = tb["digits"].to_pylist()
    state = tb["state"].to_pylist()
    n = len(core)
    tasks = [(s, country[s:s + chunk], core[s:s + chunk], addr[s:s + chunk],
              digits[s:s + chunk], state[s:s + chunk]) for s in range(0, n, chunk)]
    del country, core, addr, digits, state
    with Pool(procs) as p:
        res = p.map(_keys_chunk, tasks, chunksize=1)
    hs = np.concatenate([r[0] for r in res])
    ix = np.concatenate([r[1] for r in res])
    return hs, ix


def build_index(split, src, procs=8, force=False):
    tag = "%s_s%d" % (split, src)
    dst = os.path.join(WORK, "idx_%s.npz" % tag)
    if os.path.exists(dst) and not force:
        print("  skip (exists):", os.path.basename(dst))
        return dst
    t0 = time.time()
    tb = pq.read_table(os.path.join(WORK, "%s_norm.parquet" % tag),
                       columns=["country", "core", "addr", "digits", "state"])
    n_rec = tb.num_rows
    hs, ix = gen_key_arrays(tb, procs=procs)
    del tb
    print("  %s: %d records -> %d (key,rec) pairs (%.1fs)"
          % (tag, n_rec, len(hs), time.time() - t0))
    order = np.argsort(hs, kind="stable")
    hs = hs[order]
    ix = ix[order]
    del order
    # unique key boundaries
    uniq, starts, counts = np.unique(hs, return_index=True, return_counts=True)
    del hs
    keep = counts <= MAX_DF
    dropped = int((~keep).sum())
    dropped_post = int(counts[~keep].sum())
    uniq = uniq[keep]
    starts = starts[keep]
    counts = counts[keep]
    # compact the postings so they are contiguous per surviving key
    total = int(counts.sum())
    cum0 = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=cum0[1:])
    take = (np.repeat(starts.astype(np.int64), counts)
            + np.arange(total, dtype=np.int64) - np.repeat(cum0, counts))
    post = ix[take]
    del ix, take
    offs = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offs[1:])
    np.savez(dst, uniq=uniq, offs=offs, post=post, n_rec=np.int64(n_rec))
    print("  -> %d keys kept (%d dropped, %d postings dropped), %d postings, %.1fs, %.0f MB"
          % (len(uniq), dropped, dropped_post, len(post), time.time() - t0,
             os.path.getsize(dst) / 1e6))
    return dst


class Index(object):
    def __init__(self, path):
        z = np.load(path)
        self.uniq = z["uniq"]
        self.offs = z["offs"]
        self.post = z["post"]
        self.n_rec = int(z["n_rec"])

    def lookup(self, qhash, qrow):
        """qhash/qrow: parallel arrays (one entry per (query, key)).
        -> (query_row, candidate_rec) posting arrays."""
        pos = np.searchsorted(self.uniq, qhash)
        pos = np.minimum(pos, len(self.uniq) - 1)
        hit = self.uniq[pos] == qhash
        pos = pos[hit]
        qrow = qrow[hit]
        cnt = (self.offs[pos + 1] - self.offs[pos]).astype(np.int64)
        total = int(cnt.sum())
        if total == 0:
            return np.empty(0, np.int64), np.empty(0, np.int32)
        qrep = np.repeat(qrow, cnt)
        base = np.repeat(self.offs[pos], cnt)
        within = np.arange(total, dtype=np.int64) - np.repeat(
            np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
        return qrep, self.post[base + within]


def query_keys(country, core, addr, digits, state):
    """-> (hash array, query-row array) for a list of query records."""
    hs, rw = [], []
    for i in range(len(core)):
        for k in gen_keys(country[i], core[i], addr[i], digits[i], state[i]):
            hs.append(hash(k) & 0x7FFFFFFFFFFFFFFF)
            rw.append(i)
    return np.array(hs, dtype=np.uint64), np.array(rw, dtype=np.int64)


def topk_candidates(index, qh, qr, n_q, k):
    """Rank each query's candidates by number of distinct shared keys, keep top k."""
    qrep, cand = index.lookup(qh, qr)
    if len(qrep) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int32), np.empty(0, np.int32)
    comb = qrep * np.int64(index.n_rec) + cand.astype(np.int64)
    comb.sort()
    # run-length encode -> (pair, shared-key count)
    brk = np.empty(len(comb), dtype=bool)
    brk[0] = True
    np.not_equal(comb[1:], comb[:-1], out=brk[1:])
    starts = np.flatnonzero(brk)
    pair = comb[starts]
    cnt = np.diff(np.append(starts, len(comb))).astype(np.int32)
    q = (pair // index.n_rec)
    c = (pair - q * index.n_rec).astype(np.int32)
    # order by (query asc, count desc) and take the first k of each query
    order = np.lexsort((-cnt, q))
    q, c, cnt = q[order], c[order], cnt[order]
    rank = np.arange(len(q), dtype=np.int64)
    first = np.searchsorted(q, np.arange(n_q, dtype=np.int64), side="left")
    rank = rank - first[q]
    sel = rank < k
    return q[sel], c[sel], cnt[sel]


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    for src in (2, 3):
        build_index(split, src)
