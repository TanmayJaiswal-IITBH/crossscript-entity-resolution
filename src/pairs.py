"""Candidate generation over BOTH pools in a single pass.

Day 1 ran S2 and S3 as separate processes, which was fine for absolute features
but makes per-entity relative features impossible -- an entity's candidates were
split across two processes that never saw each other. Here one pass holds both
indexes and both pools, so every entity's full candidate set is materialised
together. Parallelism moves to query sharding instead (--shard i/N).
"""
import os
import sys
import collections
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, LEGAL
from block3 import FieldIndex, retrieve_union

# legal forms as a bitmask so the pools carry 2 bytes/record instead of a set
_LEGAL_ORDER = sorted(set(LEGAL.values()))
LEGAL_BIT = {v: 1 << i for i, v in enumerate(_LEGAL_ORDER)}


def legal_mask(name):
    """Legal forms present in a normalised name, as a bitmask.

    Runs of single letters are glued back together first: 'L.L.C.' normalises to
    'l l c', and without this the most common US legal form in the data would
    never be detected. Same for 'P.L.C.', 'S.A.S.' and friends."""
    toks = name.split()
    glued, run = [], []
    for w in toks:
        if len(w) == 1 and w.isalpha():
            run.append(w)
            continue
        if run:
            glued.append("".join(run))
            run = []
        glued.append(w)
    if run:
        glued.append("".join(run))
    m = 0
    for w in glued:
        lv = LEGAL.get(w)
        if lv is not None:
            m |= LEGAL_BIT[lv]
    return m


def _masks(names):
    return np.fromiter((legal_mask(n) for n in names), dtype=np.uint32,
                       count=len(names))


def _rarity(cores, tdf):
    """log1p(df) of a name's rarest core token. Low = distinctive name."""
    out = np.zeros(len(cores), dtype=np.float32)
    for i, c in enumerate(cores):
        m = 0
        first = True
        for w in c.split():
            d = tdf.get(w, 0)
            if first or d < m:
                m = d
                first = False
        out[i] = np.log1p(m) if not first else 0.0
    return out


class Pools(object):
    """Both pool sources, resident. ~4 GB for the 5M-record training pools.

    Also precomputes two corpus statistics per record, which is what the Day-2
    error analysis said was missing:

      name_rarity  log1p of the rarest core token's document frequency. Half of
                   the remaining false merges involve a name with no identifying
                   token, where the address wins by default.
      addr_sat     log1p of how many pool records share this exact normalised
                   address. In dense commercial buildings dozens of distinct
                   businesses collapse to the same address string, so address
                   agreement there is nearly worthless evidence.

    Both are per-record scalars, computed once at load, so they cost nothing per
    pair."""

    def __init__(self, split, srcs=(2, 3)):
        self.split = split
        self.src = {}
        for s in srcs:
            tb = pq.read_table(os.path.join(WORK, "%s_s%d_norm.parquet" % (split, s)),
                               columns=["entity_id", "name", "core", "addr", "state"])
            names = tb["name"].to_pylist()
            core = tb["core"].to_pylist()
            addr = tb["addr"].to_pylist()
            d = dict(ids=tb["entity_id"].to_pylist(),
                     core=core, addr=addr,
                     state=tb["state"].to_pylist(),
                     legal=_masks(names))
            del tb, names
            tdf = collections.Counter()
            for c in core:
                for w in set(c.split()):
                    tdf[w] += 1
            d["token_df"] = tdf
            d["name_rarity"] = _rarity(core, tdf)
            acnt = collections.Counter(a for a in addr if a)
            d["addr_count"] = acnt
            d["addr_sat"] = np.fromiter(
                (np.log1p(acnt.get(a, 0)) if a else 0.0 for a in addr),
                dtype=np.float32, count=len(addr))
            d["index"] = FieldIndex(split, s)
            self.src[s] = d
        # only the reference source's lookup tables are consulted afterwards
        # (query_scalars uses the lowest-numbered source); the others' dicts are
        # several hundred MB each and would otherwise sit idle in every shard.
        ref = min(self.src)
        for s, d in self.src.items():
            if s != ref:
                d.pop("token_df", None)
                d.pop("addr_count", None)

    def query_scalars(self, q, s, e):
        """Per-query rarity/saturation, measured against the S2 pool.

        Computed once per chunk rather than once per candidate."""
        ref = self.src[min(self.src)]
        tdf = ref["token_df"]
        acnt = ref["addr_count"]
        cores = q["core"][s:e]
        addrs = q["addr"][s:e]
        rar = _rarity(cores, tdf)
        sat = np.fromiter((np.log1p(acnt.get(a, 0)) if a else 0.0 for a in addrs),
                          dtype=np.float32, count=len(addrs))
        return rar, sat

    def close(self):
        self.src.clear()


def load_queries(split, subset_ids=None):
    tb = pq.read_table(os.path.join(WORK, "%s_s1_norm.parquet" % split))
    if subset_ids is not None:
        ids = tb["entity_id"].to_pylist()
        tb = tb.take([i for i, e in enumerate(ids) if e in subset_ids])
    q = {c: tb[c].to_pylist() for c in tb.column_names}
    q["legal"] = _masks(q["name"])
    return q


def candidates_for_chunk(pools, q, s, e, kn, ka, bn, ba):
    """-> rows (feature tuples), group boundaries, and the candidate ids.

    Rows come out grouped contiguously by query, which is what
    features2.add_relative expects."""
    n = e - s
    q_rar, q_sat = pools.query_scalars(q, s, e)
    per_q = [[] for _ in range(n)]
    for src, d in pools.src.items():
        (qa_, ca_, sa_), (qb_, cb_, sb_) = retrieve_union(
            d["index"], q["country"][s:e], q["core"][s:e], q["addr"][s:e],
            k_name=kn, k_addr=ka, budget_name=bn, budget_addr=ba)
        m = {}
        for a, b, c in zip(qa_.tolist(), ca_.tolist(), sa_.tolist()):
            m[(a, b)] = (c, 0.0)
        for a, b, c in zip(qb_.tolist(), cb_.tolist(), sb_.tolist()):
            k = (a, b)
            prev = m.get(k)
            m[k] = (prev[0], c) if prev else (0.0, c)
        is_s2 = 1.0 if src == 2 else 0.0
        for (a, b), (bn_, ba_) in m.items():
            per_q[a].append((src, b, bn_, ba_, is_s2))

    rows, ids, starts, ends = [], [], [], []
    pos = 0
    for i in range(n):
        starts.append(pos)
        qi = s + i
        qcore = q["core"][qi]
        qaddr = q["addr"][qi]
        qlegal = int(q["legal"][qi])
        qstate = q["state"][qi]
        qcountry = q["country"][qi]
        for (src, j, bn_, ba_, is_s2) in per_q[i]:
            d = pools.src[src]
            rows.append((qcore, qaddr, qlegal, qstate, qcountry,
                         d["core"][j], d["addr"][j], int(d["legal"][j]), d["state"][j],
                         bn_, ba_, is_s2,
                         float(q_rar[i]), float(d["name_rarity"][j]),
                         float(q_sat[i]), float(d["addr_sat"][j])))
            ids.append(d["ids"][j])
            pos += 1
        ends.append(pos)
    return rows, np.array(starts), np.array(ends), ids
