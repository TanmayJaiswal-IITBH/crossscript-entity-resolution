"""Step 1 of the k-fold evaluation: pick a competition-closed set of train entities.

Conflict resolution only acts when two S1 entities competing for one record are
BOTH scored. A random sample almost never contains both sides of a conflict
(sampling a fraction f sees ~f^2 of them), so every variant would look identical.

Competing entities share an address, hence a state. So the evaluation set is
EVERY train S1 entity in a chosen set of states. The state comes from the raw
Source-1 address, whose comma-separated components are clean (it is the
deduplicated reference source). The closure assumption is verified on the real
test conflicts before it is relied on.
"""
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv, source_path, load_gt

US = {"AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL",
      "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
      "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
      "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"}
IN = {s.lower() for s in [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Orissa", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi",
    "Jammu & Kashmir", "Jammu and Kashmir", "Ladakh", "Puducherry", "Pondicherry",
    "Chandigarh", "Andaman & Nicobar Islands", "Lakshadweep", "Dadra & Nagar Haveli",
    "Daman & Diu", "Dadra and Nagar Haveli and Daman and Diu"]}

TARGET_PER_COUNTRY = 60_000
SEED = 20260927


def state_of(country, raw_addr):
    parts = [p.strip() for p in (raw_addr or "").split(",")]
    for p in reversed(parts):
        if country == "US" and p.upper() in US and len(p) == 2:
            return p.upper()
        if country == "India" and p.lower() in IN:
            return p.lower()
    return ""


def states_for(split):
    t = read_tsv(source_path(split, 1))
    ids = t["entity_id"].to_pylist()
    co = t["country"].fill_null("").to_pylist()
    ad = t["business_address"].fill_null("").to_pylist()
    return ids, co, [state_of(c, a) for c, a in zip(co, ad)]


def main():
    # ---- 1. coverage of the state extractor
    ids, co, st = states_for("train")
    cov = collections.Counter()
    tot = collections.Counter()
    for c, s in zip(co, st):
        tot[c] += 1
        cov[c] += bool(s)
    print("state extracted (train S1):")
    for c in tot:
        print("  %-6s %6.2f%% of %d" % (c, 100 * cov[c] / tot[c], tot[c]))

    # ---- 2. verify closure on REAL test conflicts
    tids, tco, tst = states_for("test")
    tkey = {e: (c, s) for e, c, s in zip(tids, tco, tst)}
    claim = collections.defaultdict(list)
    sub = os.path.join(os.path.dirname(WORK), "output", "variants",
                       "matching_results_lgb.tsv")
    with open(sub, encoding="utf-8") as f:
        next(f)
        for line in f:
            e, _, m = line.rstrip("\n").partition("\t")
            for c in (m.split(",") if m else []):
                claim[c].append(e)
    same = diff = unk = 0
    for c, es in claim.items():
        if len(es) < 2:
            continue
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                a, b = tkey[es[i]], tkey[es[j]]
                if not a[1] or not b[1]:
                    unk += 1
                elif a == b:
                    same += 1
                else:
                    diff += 1
    n = same + diff + unk
    print("\nclosure check on %d competing claimant pairs in the TEST submission:" % n)
    print("  same state        %6.2f%%" % (100 * same / n))
    print("  different state   %6.2f%%" % (100 * diff / n))
    print("  state unknown     %6.2f%%" % (100 * unk / n))

    # ---- 3. pick whole states until each country reaches its target
    size = collections.Counter((c, s) for c, s in zip(co, st) if s)
    rng = np.random.default_rng(SEED)
    chosen = []
    for c in ("US", "India"):
        cand = [k for k in size if k[0] == c and size[k] <= TARGET_PER_COUNTRY // 2]
        rng.shuffle(cand)
        n_c = 0
        for k in cand:
            if n_c >= TARGET_PER_COUNTRY:
                break
            chosen.append(k)
            n_c += size[k]
    chosen_set = set(chosen)
    print("\nchosen states (%d):" % len(chosen))
    for c in ("US", "India"):
        ks = sorted((k for k in chosen if k[0] == c), key=lambda k: -size[k])
        print("  %-6s %d entities: %s" % (c, sum(size[k] for k in ks),
                                         ", ".join("%s(%d)" % (k[1], size[k]) for k in ks)))

    # ---- 4. write the evaluation set with its ground truth
    gi, gc = load_gt()
    gmap = dict(zip(gi, gc))
    rows = [(e, gmap[e], c, s) for e, c, s in zip(ids, co, st) if (c, s) in chosen_set]
    with open(os.path.join(WORK, "cv_truth.tsv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for e, m, _c, _s in rows:
            w.writerow([e, m])
    with open(os.path.join(WORK, "cv_meta.tsv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        w.writerow(["source1_entity_id", "country", "state"])
        for e, _m, c, s in rows:
            w.writerow([e, c, s])
    nt = np.array([0 if not m else m.count(",") + 1 for _e, m, _c, _s in rows])
    print("\nevaluation set: %d entities, %d true pairs, singleton rate %.4f"
          % (len(rows), int(nt.sum()), float((nt == 0).mean())))
    print("wrote work/cv_truth.tsv and work/cv_meta.tsv")


if __name__ == "__main__":
    main()
