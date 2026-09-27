"""Variant: award each contested record to its highest-scoring claimant.

Takes the finished submission plus a re-scored file (entity -> id:score list) for
every entity touching a contested record, verifies the re-scored predictions
reproduce the submission exactly, then resolves conflicts by score.
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    order, pred = [], {}
    with open(args.matching, encoding="utf-8") as f:
        next(f)
        for line in f:
            e, _, m = line.rstrip("\n").partition("\t")
            order.append(e)
            pred[e] = m.split(",") if m else []

    scored = {}
    with open(args.scored, encoding="utf-8") as f:
        for line in f:
            e, _, m = line.rstrip("\n").partition("\t")
            lst = []
            for tok in (m.split(",") if m else []):
                c, _, s = tok.rpartition(":")
                lst.append((c, float(s)))
            scored[e] = lst

    # determinism check: re-scoring must reproduce the submitted accept sets
    mism = sum(1 for e, lst in scored.items()
               if set(c for c, _ in lst) != set(pred.get(e, [])))
    print("re-scored entities %d, accept-set mismatches vs submission: %d"
          % (len(scored), mism))
    if mism:
        print("WARNING: pipeline is not reproducing its own output for %d entities" % mism)

    claims = collections.defaultdict(list)
    for e, lst in scored.items():
        for c, s in lst:
            claims[c].append((s, e))
    lose = collections.defaultdict(set)
    n_contested = n_removed = 0
    for c, cl in claims.items():
        if len(cl) < 2:
            continue
        n_contested += 1
        cl.sort(reverse=True)
        for _s, e in cl[1:]:
            lose[e].add(c)
            n_removed += 1
    print("contested records %d, pairs removed %d (kept %d with the top claimant)"
          % (n_contested, n_removed, n_contested))

    n_empty = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for e in order:
            lst = pred[e]
            if e in lose:
                lst = [c for c in lst if c not in lose[e]]
            if not lst:
                n_empty += 1
            f.write(e + "\t" + ",".join(lst) + "\n")
    print("wrote %s  (%d rows, %d empty)" % (args.out, len(order), n_empty))


if __name__ == "__main__":
    main()
