"""Step 3: which bi-encoder configuration recovers what TF-IDF blocking misses?

For every true pair the current blocking misses on validation, rank the true
pool record against a large same-country distractor sample under each
configuration (model x text), then scale the rank to the full pool size.
A pair counts as recovered at K if its estimated full-pool rank is < K.

Runs inside the E: venv (CUDA torch). Self-contained: no project imports.
"""
import json
import os
import time
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.csv as pv
import torch
from sentence_transformers import SentenceTransformer

ROOT = os.environ.get("ER_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
WORK = ROOT + "/work"
DATA = ROOT + "/dataset/train"
CONFIG = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "config", "embedding_models.json")))
PINS = CONFIG["models"]
D_PER_COUNTRY = 400_000
KS = (20, 50, 100, 200)
INDIC = ("DEVANAGARI", "TAMIL", "TELUGU", "KANNADA", "GUJARATI", "BENGALI",
         "GURMUKHI", "MALAYALAM", "ORIYA")


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


def read(path, cols):
    opts = dict(parse_options=pv.ParseOptions(delimiter="\t"),
                convert_options=pv.ConvertOptions(
                    column_types={c: pa.string() for c in cols},
                    strings_can_be_null=True, null_values=[], include_columns=cols))
    t = pv.read_csv(path, **opts)
    return {c: t[c].fill_null("").to_pylist() for c in cols}


def has_indic(s):
    for ch in s:
        if ord(ch) > 0x0900 and any(k in unicodedata.name(ch, "") for k in INDIC):
            return True
    return False


def main():
    z = np.load(WORK + "/valcache_meta.npz", allow_pickle=True)
    ent, gid, label, cand = z["entity"], z["gid"], z["label"], z["cand"]
    got = {}
    for k in np.flatnonzero(label == 1):
        got.setdefault(ent[gid[k]], set()).add(cand[k])
    vt = read(WORK + "/val_truth.tsv", ["source1_entity_id", "matched_entity_ids"])
    missed, all_true = [], set()
    for e, c in zip(vt["source1_entity_id"], vt["matched_entity_ids"]):
        tr = set(c.split(",")) if c else set()
        all_true |= tr
        for m in tr - got.get(e, set()):
            missed.append((e, m))
    n_true_total = sum(len(c.split(",")) for c in vt["matched_entity_ids"] if c)
    print("missed true pairs: %d of %d" % (len(missed), n_true_total), flush=True)

    cols = ["entity_id", "business_name", "business_address", "country"]
    s1 = read(DATA + "/train_source1.tsv", cols)
    s1i = {e: i for i, e in enumerate(s1["entity_id"])}
    pool = {k: [] for k in cols}
    for s in (2, 3):
        t = read(DATA + "/train_source%d.tsv" % s, cols)
        for k in cols:
            pool[k].extend(t[k])
        del t
    pidx = {e: i for i, e in enumerate(pool["entity_id"])}
    pc = np.array(pool["country"])
    rng = np.random.default_rng(0)
    dist, scale = {}, {}
    for c in ("US", "India"):
        cand_rows = np.flatnonzero(pc == c)
        cand_rows = np.array([r for r in cand_rows
                              if pool["entity_id"][r] not in all_true])
        dist[c] = rng.choice(cand_rows, D_PER_COUNTRY, replace=False)
        scale[c] = (pc == c).sum() / D_PER_COUNTRY
    print("distractors per country: %d  (full-pool scale US x%.1f, India x%.1f)"
          % (D_PER_COUNTRY, scale["US"], scale["India"]), flush=True)

    q_rows = [s1i[e] for e, _ in missed]
    t_rows = [pidx[m] for _, m in missed]
    qc = np.array([s1["country"][r] for r in q_rows])
    cat = np.array(["cross_script" if has_indic(pool["business_name"][t])
                    and not has_indic(s1["business_name"][q]) else "same_script"
                    for q, t in zip(q_rows, t_rows)])

    def texts(src, rows, with_addr):
        if with_addr:
            return [(src["business_name"][r] + ", " + src["business_address"][r]).strip(", ")
                    for r in rows]
        return [src["business_name"][r] for r in rows]

    results = {}
    for mname in PINS:
        model = load_encoder(mname)
        model.half()
        prefix = "query: " if "e5" in mname else ""
        short = "e5-small" if "e5" in mname else "mMiniLM-L12"
        for with_addr in (False, True):
            model.max_seq_length = 96 if with_addr else 48
            tag = "%s | %s" % (short, "name+address" if with_addr else "name only")
            enc = lambda xs: model.encode([prefix + x for x in xs], batch_size=512,
                                          convert_to_tensor=True, normalize_embeddings=True,
                                          show_progress_bar=False)
            t0 = time.time()
            Q = enc(texts(s1, q_rows, with_addr))
            T = enc(texts(pool, t_rows, with_addr))
            D = {c: enc(texts(pool, dist[c], with_addr)) for c in dist}
            n_enc = len(q_rows) * 2 + 2 * D_PER_COUNTRY
            el = time.time() - t0
            sim_true = (Q * T).sum(1)
            est_rank = np.zeros(len(missed))
            for c in D:
                idx = np.flatnonzero(qc == c)
                for b in range(0, len(idx), 1024):
                    ii = torch.as_tensor(idx[b:b + 1024], device="cuda")
                    s = Q[ii] @ D[c].T
                    cnt = (s > sim_true[ii, None]).sum(1).float().cpu().numpy()
                    est_rank[idx[b:b + 1024]] = cnt * scale[c]
            results[tag] = est_rank
            print("\n%s   (%d texts encoded in %.0fs = %.0f texts/s)"
                  % (tag, n_enc, el, n_enc / el), flush=True)
            for cname in ("cross_script", "same_script", "ALL"):
                m = np.ones(len(missed), bool) if cname == "ALL" else (cat == cname)
                row = "  ".join("K=%-3d %5.1f%%" % (K, 100 * (est_rank[m] < K).mean())
                                for K in KS)
                print("   %-13s n=%-5d recovered: %s" % (cname, m.sum(), row), flush=True)
            del Q, T, D
            torch.cuda.empty_cache()
        del model

    print("\n=== projected blocking recall if dense top-K were unioned in ===")
    base_rec = 1 - len(missed) / n_true_total
    for tag, r in results.items():
        print("  %-30s " % tag + "  ".join(
            "K=%-3d %.4f" % (K, base_rec + (r < K).sum() / n_true_total) for K in KS))
    print("  (current TF-IDF blocking recall %.4f)" % base_rec)


if __name__ == "__main__":
    main()
