"""Day-2 pairwise features: absolute similarity + per-entity relative features.

The relative block is the important part. The question a matcher has to answer is
not "is this pair similar in absolute terms" but "is this pair clearly the best
explanation for this S1 entity" -- so every candidate also carries its rank, its
gap to the entity's best candidate, and its gap to the runner-up. Those require
the entity's candidates from BOTH pools to be present at once, which is why the
pipeline was restructured to score S2 and S3 in a single pass.
"""
import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, LCSseq

# ---------------------------------------------------------------- absolute
ABS_FEATURES = [
    # name
    "n_jw", "n_lev", "n_tokset", "n_toksort", "n_partial", "n_jacc", "n_contain",
    "n_lcs", "n_3gram", "n_idfw", "n_len_ratio", "n_ntok_q", "n_ntok_p",
    "n_first_tok", "n_last_tok", "n_empty", "n_abbrev",
    # legal form
    "legal_state",
    # address
    "a_lev", "a_tokset", "a_toksort", "a_jacc", "a_contain", "a_3gram",
    "a_dig_jacc", "a_dig_any", "a_streetnum", "a_postal", "a_city", "a_state",
    "a_len_ratio", "a_empty_p", "a_empty_q", "a_short_p", "a_landmark",
    # context
    "blk_name", "blk_addr", "src_s2", "country_id",
    # corpus statistics -- added Day 3 from the false-merge analysis
    "q_name_rarity", "p_name_rarity", "q_addr_sat", "p_addr_sat",
    "streetnum_conflict", "dig_conflict", "generic_and_addr_only",
]
REL_FEATURES = [
    "n_cands", "n_near_best",
    "s0", "s0_rank", "s0_d_best", "s0_r_best", "s0_d_second",
    "nt_rank", "nt_d_best", "at_rank", "at_d_best",
    "s0_rank_src", "s0_is_best", "s0_z",
]
FEATURE_NAMES = ABS_FEATURES + REL_FEATURES
N_ABS = len(ABS_FEATURES)
N_FEATURES = len(FEATURE_NAMES)
_IX = {k: i for i, k in enumerate(FEATURE_NAMES)}

_LANDMARK = re.compile(r"\b(near|opp|opposite|behind|beside|next|above|below|infront)\b")
_DIGRUN = re.compile(r"\d+")
COUNTRY_ID = {"US": 0, "India": 1, "France": 2}

# legal forms, mirrored from common.LEGAL values
_LEGAL_VALUES = {"pvt", "ltd", "llp", "llc", "inc", "corp", "co", "sa", "sas",
                 "sarl", "sci", "gmbh", "bv", "nv", "plc", "ag"}


def _ngrams(s, n=3):
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _cos(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / ((len(a) * len(b)) ** 0.5)


def _jacc(a, b):
    if not a or not b:
        return 0.0
    i = len(a & b)
    return i / (len(a) + len(b) - i)


def _contain(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _idf_proxy(a, b):
    """Rare-token overlap, with token length standing in for IDF.

    Long tokens are overwhelmingly the discriminative ones here ('byadarahalli'
    vs 'road'), and this avoids shipping the 2M-entry df table to every worker."""
    if not a or not b:
        return 0.0
    sh = sum(len(t) for t in (a & b))
    un = sum(len(t) for t in (a | b))
    return sh / un if un else 0.0


def _digs(s):
    return {d.lstrip("0") or "0" for d in _DIGRUN.findall(s)}


def _postal(s):
    for d in _DIGRUN.findall(s):
        if len(d) in (5, 6):
            return d
    return ""


def _streetnum(s):
    m = _DIGRUN.search(s)
    return (m.group(0).lstrip("0") or "0") if m else ""


def _legal_state(lq, lp):
    """lq/lp are legal-form bitmasks (see pairs.LEGAL_BIT).

    Kept as a 4-way categorical rather than a similarity: 'Acme Corp' vs
    'Acme Ltd' is a genuine mismatch signal, while 'Acme' vs 'Acme Ltd' is only
    a missing suffix and carries far less evidence."""
    if not lq and not lp:
        return 0.0
    if not lq or not lp:
        return 3.0
    return 1.0 if (lq & lp) else 2.0


def _abbrev(qt, pt):
    """One side's tokens are initials of the other ('IBM' vs 'International ...')."""
    if not qt or not pt:
        return 0.0
    for a, b in ((qt, pt), (pt, qt)):
        if len(a) == 1 and len(a[0]) >= 2 and len(a[0]) == len(b) >= 2:
            if all(a[0][i] == b[i][0] for i in range(len(b))):
                return 1.0
    return 0.0


def abs_features(qcore, qaddr, qlegal, qstate, qcountry,
                 pcore, paddr, plegal, pstate, blk_n, blk_a, src_s2,
                 q_rarity=0.0, p_rarity=0.0, q_sat=0.0, p_sat=0.0):
    qt_l = qcore.split()
    pt_l = pcore.split()
    qt, pt = set(qt_l), set(pt_l)
    n_empty = 1.0 if (not qcore or not pcore) else 0.0
    if qcore and pcore:
        n_jw = JaroWinkler.similarity(qcore, pcore)
        n_lev = fuzz.ratio(qcore, pcore) / 100.0
        n_tokset = fuzz.token_set_ratio(qcore, pcore) / 100.0
        n_toksort = fuzz.token_sort_ratio(qcore, pcore) / 100.0
        n_partial = fuzz.partial_ratio(qcore, pcore) / 100.0
        n_lcs = LCSseq.normalized_similarity(qcore, pcore)
        n_3gram = _cos(_ngrams(qcore), _ngrams(pcore))
    else:
        n_jw = n_lev = n_tokset = n_toksort = n_partial = n_lcs = n_3gram = 0.0
    n_jacc = _jacc(qt, pt)
    n_contain = _contain(qt, pt)
    n_idfw = _idf_proxy(qt, pt)
    lq, lp = len(qcore), len(pcore)
    n_len_ratio = min(lq, lp) / max(lq, lp) if lq and lp else 0.0
    n_first = 1.0 if (qt_l and pt_l and qt_l[0] == pt_l[0]) else 0.0
    n_last = 1.0 if (qt_l and pt_l and qt_l[-1] == pt_l[-1]) else 0.0

    qa_l = qaddr.split()
    pa_l = paddr.split()
    qa, pa = set(qa_l), set(pa_l)
    if qaddr and paddr:
        a_lev = fuzz.ratio(qaddr, paddr) / 100.0
        a_tokset = fuzz.token_set_ratio(qaddr, paddr) / 100.0
        a_toksort = fuzz.token_sort_ratio(qaddr, paddr) / 100.0
        a_3gram = _cos(_ngrams(qaddr), _ngrams(paddr))
    else:
        a_lev = a_tokset = a_toksort = a_3gram = 0.0
    a_jacc = _jacc(qa, pa)
    a_contain = _contain(qa, pa)
    dq, dp = _digs(qaddr), _digs(paddr)
    a_dig_jacc = _jacc(dq, dp)
    a_dig_any = 1.0 if (dq & dp) else 0.0
    sq, sp = _streetnum(qaddr), _streetnum(paddr)
    a_streetnum = 1.0 if (sq and sq == sp) else 0.0
    # an explicit *conflict* is different evidence from a missing match:
    # 'frye santana inc' at 108 Tower St vs 19 Tower St was the single most
    # common false merge on Day 2.
    streetnum_conflict = 1.0 if (sq and sp and sq != sp) else 0.0
    zq, zp = _postal(qaddr), _postal(paddr)
    a_postal = 1.0 if (zq and zq == zp) else (0.0 if (zq and zp) else 0.5)
    a_city = _contain(set(qa_l[-3:]), set(pa_l[-3:]))
    a_state = 1.0 if (qstate and qstate == pstate) else (0.0 if (qstate and pstate) else 0.5)
    la, lb = len(qaddr), len(paddr)
    a_len_ratio = min(la, lb) / max(la, lb) if la and lb else 0.0
    a_landmark = 1.0 if _LANDMARK.search(paddr) or _LANDMARK.search(qaddr) else 0.0
    union = dq | dp
    dig_conflict = (len(union - (dq & dp)) / len(union)) if union else 0.0
    # the exact shape of failure pattern #2: uninformative name, and the only
    # thing carrying the pair is an address that thousands of records share
    generic_and_addr_only = 1.0 if (q_rarity > 8.0 and p_sat > 3.0) else 0.0

    return [n_jw, n_lev, n_tokset, n_toksort, n_partial, n_jacc, n_contain,
            n_lcs, n_3gram, n_idfw, n_len_ratio, float(len(qt)), float(len(pt)),
            n_first, n_last, n_empty, _abbrev(qt_l, pt_l),
            _legal_state(qlegal, plegal),
            a_lev, a_tokset, a_toksort, a_jacc, a_contain, a_3gram,
            a_dig_jacc, a_dig_any, a_streetnum, a_postal, a_city, a_state,
            a_len_ratio, 1.0 if not paddr else 0.0, 1.0 if not qaddr else 0.0,
            1.0 if len(paddr) < 12 else 0.0, a_landmark,
            blk_n, blk_a, src_s2, float(COUNTRY_ID.get(qcountry, 3)),
            q_rarity, p_rarity, q_sat, p_sat,
            streetnum_conflict, dig_conflict, generic_and_addr_only]


def abs_block(rows):
    out = np.empty((len(rows), N_ABS), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i] = abs_features(*r)
    return out


# ---------------------------------------------------------------- relative
def base_score(A):
    """Unsupervised blend used only as the anchor for the relative features."""
    ix = _IX
    name = (0.42 * A[:, ix["n_tokset"]] + 0.22 * A[:, ix["n_toksort"]]
            + 0.18 * A[:, ix["n_jw"]] + 0.18 * A[:, ix["n_jacc"]])
    addr = (0.40 * A[:, ix["a_tokset"]] + 0.24 * A[:, ix["a_jacc"]]
            + 0.20 * A[:, ix["a_dig_jacc"]] + 0.16 * A[:, ix["a_lev"]])
    s = 0.62 * name + 0.38 * addr
    return np.where(A[:, ix["a_empty_p"]] > 0.5, 0.92 * name, s).astype(np.float32)


def _grouped_rel(vals, starts, ends):
    """-> (rank, delta_to_best, ratio_to_best, delta_to_second) per row."""
    n = len(vals)
    rank = np.zeros(n, dtype=np.float32)
    d_best = np.zeros(n, dtype=np.float32)
    r_best = np.zeros(n, dtype=np.float32)
    d_second = np.zeros(n, dtype=np.float32)
    for a, b in zip(starts, ends):
        if a >= b:
            continue
        v = vals[a:b]
        order = np.argsort(-v, kind="stable")
        rk = np.empty(b - a, dtype=np.float32)
        rk[order] = np.arange(b - a, dtype=np.float32)
        best = v[order[0]]
        second = v[order[1]] if b - a > 1 else 0.0
        rank[a:b] = rk
        d_best[a:b] = v - best
        r_best[a:b] = v / best if best > 1e-6 else 0.0
        d_second[a:b] = v - second
    return rank, d_best, r_best, d_second


def add_relative(A, group_starts, group_ends, src_s2):
    """A: (n, N_ABS) absolute features, rows grouped contiguously per S1 entity."""
    ix = _IX
    n = A.shape[0]
    R = np.zeros((n, len(REL_FEATURES)), dtype=np.float32)
    s0 = base_score(A)
    sizes = np.zeros(n, dtype=np.float32)
    for a, b in zip(group_starts, group_ends):
        sizes[a:b] = b - a
    rank, d_best, r_best, d_second = _grouped_rel(s0, group_starts, group_ends)
    # how crowded the top of this entity's candidate list is: many near-tied
    # candidates means the evidence does not single anything out
    near = np.zeros(n, dtype=np.float32)
    for a, b in zip(group_starts, group_ends):
        if a < b:
            v = s0[a:b]
            near[a:b] = float((v >= v.max() - 0.05).sum())
    nt_rank, nt_d_best, _, _ = _grouped_rel(
        np.ascontiguousarray(A[:, ix["n_tokset"]]), group_starts, group_ends)
    at_rank, at_d_best, _, _ = _grouped_rel(
        np.ascontiguousarray(A[:, ix["a_tokset"]]), group_starts, group_ends)

    # rank within the same pool source, and a within-entity z-score
    rank_src = np.zeros(n, dtype=np.float32)
    z = np.zeros(n, dtype=np.float32)
    for a, b in zip(group_starts, group_ends):
        if a >= b:
            continue
        v = s0[a:b]
        m, sd = v.mean(), v.std()
        z[a:b] = (v - m) / sd if sd > 1e-6 else 0.0
        for which in (0.0, 1.0):
            sel = np.flatnonzero(src_s2[a:b] == which)
            if len(sel) == 0:
                continue
            o = sel[np.argsort(-v[sel], kind="stable")]
            rank_src[a + o] = np.arange(len(o), dtype=np.float32)

    cols = [sizes, near, s0, rank, d_best, r_best, d_second,
            nt_rank, nt_d_best, at_rank, at_d_best,
            rank_src, (rank == 0).astype(np.float32), z]
    for j, c in enumerate(cols):
        R[:, j] = c
    return np.hstack([A, R])
