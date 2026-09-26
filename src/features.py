"""Pairwise similarity features for candidate (S1, S2/S3) pairs.

Everything here operates on the pre-normalised fields (unidecoded, punctuation
stripped, repeated characters collapsed), which is what lets a Devanagari name
and its Latin counterpart land on comparable strings at all.
"""
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES = [
    "blk_name", "blk_addr",
    "n_ratio", "n_tokset", "n_toksort", "n_partial", "n_jw",
    "n_jacc", "n_contain", "n_len_ratio", "n_empty",
    "a_ratio", "a_tokset", "a_toksort", "a_jacc", "a_contain",
    "a_dig_jacc", "a_dig_any", "a_empty_p", "a_len_ratio",
    "both_strong", "n_first_tok", "n_ntok_q", "n_ntok_p",
]
N_FEATURES = len(FEATURE_NAMES)


def _jacc(a, b):
    if not a or not b:
        return 0.0
    i = len(a & b)
    return i / (len(a) + len(b) - i)


def _contain(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _digset(s):
    out = set()
    cur = []
    for ch in s:
        if ch.isdigit():
            cur.append(ch)
        elif cur:
            out.add("".join(cur).lstrip("0") or "0")
            cur = []
    if cur:
        out.add("".join(cur).lstrip("0") or "0")
    return out


def pair_features(qcore, qaddr, pcore, paddr, blk_name, blk_addr):
    """One pair -> a list of floats, ordered as FEATURE_NAMES."""
    qt = set(qcore.split())
    pt = set(pcore.split())
    n_empty = 1.0 if (not qcore or not pcore) else 0.0
    if qcore and pcore:
        n_ratio = fuzz.ratio(qcore, pcore) / 100.0
        n_tokset = fuzz.token_set_ratio(qcore, pcore) / 100.0
        n_toksort = fuzz.token_sort_ratio(qcore, pcore) / 100.0
        n_partial = fuzz.partial_ratio(qcore, pcore) / 100.0
        n_jw = JaroWinkler.similarity(qcore, pcore)
    else:
        n_ratio = n_tokset = n_toksort = n_partial = n_jw = 0.0
    n_jacc = _jacc(qt, pt)
    n_contain = _contain(qt, pt)
    lq, lp = len(qcore), len(pcore)
    n_len_ratio = min(lq, lp) / max(lq, lp) if lq and lp else 0.0

    qa = set(qaddr.split())
    pa = set(paddr.split())
    a_empty_p = 1.0 if not paddr else 0.0
    if qaddr and paddr:
        a_ratio = fuzz.ratio(qaddr, paddr) / 100.0
        a_tokset = fuzz.token_set_ratio(qaddr, paddr) / 100.0
        a_toksort = fuzz.token_sort_ratio(qaddr, paddr) / 100.0
    else:
        a_ratio = a_tokset = a_toksort = 0.0
    a_jacc = _jacc(qa, pa)
    a_contain = _contain(qa, pa)
    dq, dp = _digset(qaddr), _digset(paddr)
    a_dig_jacc = _jacc(dq, dp)
    a_dig_any = 1.0 if (dq & dp) else 0.0
    la, lb = len(qaddr), len(paddr)
    a_len_ratio = min(la, lb) / max(la, lb) if la and lb else 0.0

    both_strong = 1.0 if (n_tokset > 0.85 and a_tokset > 0.85) else 0.0
    qf = qcore.split()[:1]
    pf = pcore.split()[:1]
    n_first_tok = 1.0 if (qf and pf and qf[0] == pf[0]) else 0.0

    return [blk_name, blk_addr,
            n_ratio, n_tokset, n_toksort, n_partial, n_jw,
            n_jacc, n_contain, n_len_ratio, n_empty,
            a_ratio, a_tokset, a_toksort, a_jacc, a_contain,
            a_dig_jacc, a_dig_any, a_empty_p, a_len_ratio,
            both_strong, n_first_tok, float(len(qt)), float(len(pt))]


def featurize_block(qcores, qaddrs, pcores, paddrs, blk_n, blk_a):
    """Vectorised over a block of pairs -> float32 (n, N_FEATURES)."""
    n = len(qcores)
    out = np.empty((n, N_FEATURES), dtype=np.float32)
    for i in range(n):
        out[i] = pair_features(qcores[i], qaddrs[i], pcores[i], paddrs[i],
                               blk_n[i], blk_a[i])
    return out


def baseline_score(f):
    """Hand-weighted blend used for the Day-1 submission (no learned model).

    f: (n, N_FEATURES) float32.  Name evidence dominates, address evidence
    corroborates, and either one alone can carry a pair when the other field is
    missing -- which is exactly the empty-address / transliterated-name case.
    """
    ix = {k: i for i, k in enumerate(FEATURE_NAMES)}
    name = (0.40 * f[:, ix["n_tokset"]] + 0.25 * f[:, ix["n_toksort"]]
            + 0.20 * f[:, ix["n_jw"]] + 0.15 * f[:, ix["n_jacc"]])
    addr = (0.45 * f[:, ix["a_tokset"]] + 0.25 * f[:, ix["a_jacc"]]
            + 0.20 * f[:, ix["a_dig_jacc"]] + 0.10 * f[:, ix["a_ratio"]])
    addr_missing = f[:, ix["a_empty_p"]] > 0.5
    s = 0.62 * name + 0.38 * addr
    s = np.where(addr_missing, 0.92 * name, s)
    return s.astype(np.float32)
