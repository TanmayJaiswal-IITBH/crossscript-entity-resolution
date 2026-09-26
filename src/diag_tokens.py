"""For true pairs, how much token evidence actually exists?

Distinguishes two very different failure modes:
  * representation failure -- the pair shares no (or almost no) token, so no
    token-based blocker can ever retrieve it;
  * ranking/budget failure  -- evidence exists but the pair was outranked.
"""
import os
import sys
import random
import collections
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, read_tsv
from block2 import gen_tokens

N_SAMPLE = 4000


def main():
    t = read_tsv(os.path.join(WORK, "val_truth.tsv"))
    vids = t["source1_entity_id"].to_pylist()
    cells = t["matched_entity_ids"].fill_null("").to_pylist()
    random.seed(11)
    pick = random.sample(range(len(vids)), N_SAMPLE)
    truth = {vids[i]: (set(cells[i].split(",")) if cells[i] else set()) for i in pick}
    want23 = set()
    for v in truth.values():
        want23 |= v
    vset = set(truth)

    tb = pq.read_table(os.path.join(WORK, "train_s1_norm.parquet"))
    ids = tb["entity_id"].to_pylist()
    tb = tb.take([i for i, e in enumerate(ids) if e in vset])
    qrec = {tb["entity_id"][i].as_py(): (tb["country"][i].as_py(), tb["core"][i].as_py(),
                                         tb["addr"][i].as_py(), tb["name"][i].as_py())
            for i in range(tb.num_rows)}
    del tb, ids

    prec = {}
    for src in (2, 3):
        pt = pq.read_table(os.path.join(WORK, "train_s%d_norm.parquet" % src))
        pids = pt["entity_id"].to_pylist()
        idxs = [i for i, e in enumerate(pids) if e in want23]
        pt = pt.take(idxs)
        for i in range(pt.num_rows):
            prec[pt["entity_id"][i].as_py()] = (pt["country"][i].as_py(), pt["core"][i].as_py(),
                                                pt["addr"][i].as_py(), pt["name"][i].as_py())
        del pt, pids, idxs

    # token df from the pool indexes, to weight shared tokens by informativeness
    from block2 import TokIndex
    idfs = {}
    for src in (2, 3):
        ix = TokIndex(os.path.join(WORK, "tok_train_s%d.npz" % src))
        idfs[src] = ix
    hist = collections.Counter()
    rare_hist = collections.Counter()
    examples = []
    n_pairs = 0
    for eid, ms in truth.items():
        if eid not in qrec:
            continue
        c, core, addr, nm = qrec[eid]
        qt = set(gen_tokens(c, core, addr))
        for mid in ms:
            if mid not in prec:
                continue
            n_pairs += 1
            c2, core2, addr2, nm2 = prec[mid]
            mt = set(gen_tokens(c2, core2, addr2))
            sh = qt & mt
            hist[min(len(sh), 12)] += 1
            # how many shared tokens are rare (df <= 2000) in the target pool?
            src = 2 if mid[1] == "2" else 3
            ix = idfs[src]
            nrare = 0
            for tk in sh:
                h = np.uint64(hash(tk) & 0x7FFFFFFFFFFFFFFF)
                p = np.searchsorted(ix.uniq, h)
                if p < len(ix.uniq) and ix.uniq[p] == h and ix.df[p] <= 2000:
                    nrare += 1
            rare_hist[min(nrare, 8)] += 1
            if len(sh) <= 1 and len(examples) < 14:
                examples.append((eid, nm, addr, mid, nm2, addr2, sorted(sh)))

    print("true pairs examined: %d" % n_pairs)
    print("\nshared-token count distribution (all tokens):")
    cum = 0
    for k in sorted(hist):
        cum += hist[k]
        print("  %2d%s : %6d  %5.2f%%   cum %5.2f%%"
              % (k, "+" if k == 12 else " ", hist[k], 100 * hist[k] / n_pairs,
                 100 * cum / n_pairs))
    print("\nshared tokens that are RARE (df<=2000) in the target pool:")
    cum = 0
    for k in sorted(rare_hist):
        cum += rare_hist[k]
        print("  %2d%s : %6d  %5.2f%%   cum %5.2f%%"
              % (k, "+" if k == 8 else " ", rare_hist[k], 100 * rare_hist[k] / n_pairs,
                 100 * cum / n_pairs))
    print("\n--- pairs with <=1 shared token (unreachable by this token space) ---")
    for eid, nm, addr, mid, nm2, addr2, sh in examples:
        print("\n  %s %r | %r" % (eid, nm, addr))
        print("  %s %r | %r   shared=%s" % (mid, nm2, addr2, sh))


if __name__ == "__main__":
    main()
