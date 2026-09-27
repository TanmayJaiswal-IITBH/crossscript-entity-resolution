"""What does blocking miss, and could a multilingual bi-encoder recover it?

Classifies every true pair absent from the candidate set:
  cross_script   the pool record's name is written in an Indic script while the
                 S1 name is Latin -- exactly what a multilingual encoder targets
  alias_name     same script, but the names share no token and almost no
                 character 3-grams (e.g. 'ir scan pvt ltd' vs 'cirahalo'); no
                 semantic model can connect these
  partial_name   same script with some overlap (typos, truncation, reordering)
  (each also split by whether the candidate's address is empty)

The cross_script share bounds what the bi-encoder can add to recall.
"""
import collections
import os
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv, source_path, norm_name

INDIC = ("DEVANAGARI", "TAMIL", "TELUGU", "KANNADA", "GUJARATI", "BENGALI",
         "GURMUKHI", "MALAYALAM", "ORIYA")


def has_indic(s):
    for ch in s or "":
        if ord(ch) > 0x0900:
            n = unicodedata.name(ch, "")
            if any(k in n for k in INDIC):
                return True
    return False


def grams(s, n=3):
    s = s.replace(" ", "")
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 0))}


def main():
    z = np.load(os.path.join(WORK, "valcache_meta.npz"), allow_pickle=True)
    ent, gid, label, cand = z["entity"], z["gid"], z["label"], z["cand"]
    got = collections.defaultdict(set)
    for k in np.flatnonzero(label == 1):
        got[ent[gid[k]]].add(cand[k])
    t = read_tsv(os.path.join(WORK, "val_truth.tsv"))
    missed, n_true = [], 0
    for e, c in zip(t["source1_entity_id"].to_pylist(),
                    t["matched_entity_ids"].fill_null("").to_pylist()):
        tr = set(c.split(",")) if c else set()
        n_true += len(tr)
        for m in tr - got[e]:
            missed.append((e, m))
    print("true pairs %d, missed by blocking %d  -> recall %.4f"
          % (n_true, len(missed), 1 - len(missed) / n_true))

    need1 = {e for e, _ in missed}
    need23 = {m for _, m in missed}
    raw = {}
    for s in (1, 2, 3):
        tb = read_tsv(source_path("train", s))
        ids = tb["entity_id"].to_pylist()
        nm = tb["business_name"].fill_null("").to_pylist()
        ad = tb["business_address"].fill_null("").to_pylist()
        want = need1 if s == 1 else need23
        for i, e in enumerate(ids):
            if e in want:
                raw[e] = (nm[i], ad[i])
        del tb, ids, nm, ad

    cat = collections.Counter()
    ex = collections.defaultdict(list)
    for e, m in missed:
        qn, _qa = raw[e]
        pn, pa = raw[m]
        empty = "empty_addr" if not pa.strip() else "has_addr"
        if has_indic(pn) and not has_indic(qn):
            c = "cross_script"
        else:
            _, qc, _ = norm_name(qn)
            _, pc, _ = norm_name(pn)
            shared = set(qc) & set(pc)
            g1, g2 = grams(" ".join(qc)), grams(" ".join(pc))
            j = len(g1 & g2) / max(len(g1 | g2), 1)
            c = "partial_name" if (shared or j >= 0.25) else "alias_name"
        cat[(c, empty)] += 1
        if len(ex[c]) < 4:
            ex[c].append((qn, pn, pa[:50]))

    n = len(missed)
    print("\n%-14s %-11s %8s %7s" % ("category", "address", "pairs", "share"))
    for c in ("cross_script", "partial_name", "alias_name"):
        for a in ("has_addr", "empty_addr"):
            v = cat[(c, a)]
            print("%-14s %-11s %8d %6.1f%%" % (c, a, v, 100 * v / n))
    tot = collections.Counter()
    for (c, _a), v in cat.items():
        tot[c] += v
    rec = 1 - n / n_true
    print("\ncurrent recall                              %.4f" % rec)
    print("if a bi-encoder recovered ALL cross_script  %.4f"
          % (rec + tot["cross_script"] / n_true))
    print("... plus ALL partial_name                   %.4f"
          % (rec + (tot["cross_script"] + tot["partial_name"]) / n_true))
    print("alias_name (unreachable by any similarity)  %.4f of all true pairs"
          % (tot["alias_name"] / n_true))
    for c in ("cross_script", "partial_name", "alias_name"):
        print("\n--- %s examples ---" % c)
        for qn, pn, pa in ex[c]:
            print("  S1 %-40r | pool %r  addr=%r" % (qn[:40], pn[:45], pa))


if __name__ == "__main__":
    main()
