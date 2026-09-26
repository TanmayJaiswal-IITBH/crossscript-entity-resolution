"""Normalise every source file once and cache it as parquet.

Produces work/{split}_s{n}_norm.parquet with columns:
    entity_id, country, name (canonical), core (core tokens, space joined),
    addr (canonical address), digits (concatenated digit runs of the address)
"""
import os
import re
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, source_path, read_tsv, norm_name, norm_addr, STATE

_DIGITS = re.compile(r"\d+")


def _work(args):
    names, addrs = args
    out_name, out_core, out_addr, out_dig, out_state = [], [], [], [], []
    for nm, ad in zip(names, addrs):
        n, core, _legal = norm_name(nm)
        a, toks, _nums = norm_addr(ad)
        out_name.append(n)
        out_core.append(" ".join(core))
        out_addr.append(a)
        out_dig.append("".join(_DIGITS.findall(a)))
        st = ""
        for w in reversed(toks[-4:]):
            s = STATE.get(w)
            if s:
                st = s
                break
        out_state.append(st)
    return out_name, out_core, out_addr, out_dig, out_state


def run(split, src, procs=8, chunk=50_000):
    dst = os.path.join(WORK, "%s_s%d_norm.parquet" % (split, src))
    if os.path.exists(dst):
        print("  skip (exists):", os.path.basename(dst))
        return
    t0 = time.time()
    t = read_tsv(source_path(split, src))
    ids = t["entity_id"]
    ctry = t["country"].fill_null("")
    names = t["business_name"].fill_null("").to_pylist()
    addrs = t["business_address"].fill_null("").to_pylist()
    n = len(names)
    tasks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, n, chunk)]
    del names, addrs
    with Pool(procs) as p:
        res = p.map(_work, tasks, chunksize=1)
    cols = [[], [], [], [], []]
    for r in res:
        for k in range(5):
            cols[k].extend(r[k])
    del res
    tb = pa.table({"entity_id": ids, "country": ctry,
                   "name": pa.array(cols[0]), "core": pa.array(cols[1]),
                   "addr": pa.array(cols[2]), "digits": pa.array(cols[3]),
                   "state": pa.array(cols[4])})
    pq.write_table(tb, dst, compression="zstd", compression_level=3)
    print("  %s: %d rows in %.1fs -> %.0f MB"
          % (os.path.basename(dst), n, time.time() - t0, os.path.getsize(dst) / 1e6))


if __name__ == "__main__":
    which = sys.argv[1:] or ["train", "test"]
    for split in which:
        for src in (1, 2, 3):
            run(split, src)
