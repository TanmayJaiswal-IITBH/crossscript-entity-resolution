"""Shared chunk engine: candidates -> absolute features -> relative features."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features2 import abs_block, add_relative, N_ABS, N_FEATURES, _IX
from pairs import candidates_for_chunk


def _abs_worker(rows):
    return abs_block(rows)


def featurize_chunk(pools, q, s, e, args, pool=None):
    """-> (X (n, N_FEATURES), candidate ids, group starts, group ends)"""
    rows, starts, ends, ids = candidates_for_chunk(
        pools, q, s, e, args.kn, args.ka, args.bn, args.ba)
    if not rows:
        return (np.empty((0, N_FEATURES), np.float32), [], starts, ends)
    if pool is not None and len(rows) > 4000:
        nb = max(1000, len(rows) // (args.procs * 3))
        tasks = [rows[i:i + nb] for i in range(0, len(rows), nb)]
        A = np.concatenate(pool.map(_abs_worker, tasks, chunksize=1))
    else:
        A = abs_block(rows)
    src_s2 = A[:, _IX["src_s2"]]
    X = add_relative(A, starts, ends, src_s2)
    return X, ids, starts, ends
