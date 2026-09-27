"""Encode every record of a split (S1, S2, S3) as 'name, address' with e5-small.

Output per source, on E:: work/emb/<split>_s<n>.f16 -- a raw float16 matrix
(rows x 384) in the same row order as the source TSV (and therefore the same
order as work/<split>_s<n>_norm.parquet and the blocking indexes), plus a
country array for partitioning retrieval by country.

Runs inside the E: venv on the GPU. Streams to disk in chunks, so RAM stays flat.
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.csv as pv
from sentence_transformers import SentenceTransformer

ROOT = os.environ.get("ER_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
EMB = ROOT + "/work/emb"
CONFIG = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "config", "embedding_models.json")))
MODEL = CONFIG["selected"]
DIM = 384
CHUNK = 200_000


def load_encoder(name, device="cuda"):
    """Load a pinned bi-encoder via the Hugging Face hub cache (HF_HOME).

    Resolving the snapshot explicitly (rather than passing the model name to
    SentenceTransformer, which looks in its own separate cache folder) makes the
    pinned revision load offline once cached, and download on a fresh machine."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(name, revision=CONFIG["models"][name]["revision"],
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                             "1_Pooling/*", "sentencepiece*"])
    return SentenceTransformer(path, device=device)


def read(path):
    cols = ["business_name", "business_address", "country"]
    opts = dict(parse_options=pv.ParseOptions(delimiter="\t"),
                convert_options=pv.ConvertOptions(
                    column_types={c: pa.string() for c in cols},
                    strings_can_be_null=True, null_values=[], include_columns=cols))
    t = pv.read_csv(path, **opts)
    return ([n.strip() + (", " + a.strip() if a.strip() else "")
             for n, a in zip(t["business_name"].fill_null("").to_pylist(),
                             t["business_address"].fill_null("").to_pylist())],
            t["country"].fill_null("").to_pylist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--sources", default="1,2,3")
    args = ap.parse_args()
    os.makedirs(EMB, exist_ok=True)
    rev = CONFIG["models"][MODEL]["revision"]
    model = load_encoder(MODEL)
    model.half()
    model.max_seq_length = 96
    for s in (int(x) for x in args.sources.split(",")):
        t0 = time.time()
        texts, country = read("%s/dataset/%s/%s_source%d.tsv" % (ROOT, args.split, args.split, s))
        out = "%s/%s_s%d.f16" % (EMB, args.split, s)
        with open(out, "wb") as f:
            for i in range(0, len(texts), CHUNK):
                e = model.encode(["query: " + x for x in texts[i:i + CHUNK]],
                                 batch_size=512, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
                e.astype(np.float16).tofile(f)
                if (i // CHUNK) % 5 == 0:
                    el = time.time() - t0
                    done = min(i + CHUNK, len(texts))
                    print("  %s S%d: %d/%d  %.0fs (eta %.0fs)"
                          % (args.split, s, done, len(texts), el,
                             el * (len(texts) - done) / done), flush=True)
        np.save("%s/%s_s%d_country.npy" % (EMB, args.split, s), np.array(country))
        print("%s S%d: %d rows -> %s  (%.1f GB, %.0fs)"
              % (args.split, s, len(texts), out, os.path.getsize(out) / 1e9,
                 time.time() - t0), flush=True)
    json.dump({"model": MODEL, "revision": rev, "dim": DIM,
               "text": "query: <business_name>, <business_address>", "dtype": "float16"},
              open(EMB + "/%s_manifest.json" % args.split, "w"), indent=1)


if __name__ == "__main__":
    main()
