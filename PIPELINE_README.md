# Business Entity Resolution — reproduction guide

Reproduces `output/matching_results.tsv` (and `output/candidate_pairs.tsv`) from the
provided TSVs. **No external data and no lookups.** One pretrained model is used, frozen,
as a text encoder: `intfloat/multilingual-e5-small` (MIT, ~118M parameters), pinned in
`config/embedding_models.json`.

## Environments

Two, because the CUDA build of PyTorch conflicts with Anaconda's Intel OpenMP runtime
when both load in one process.

**Base** — everything except embeddings:

```bash
pip install -r requirements.txt
```

**GPU embedding** — its own isolated venv (do *not* use `--system-site-packages`):

```bash
python -m venv venv_embed
venv_embed/Scripts/python -m pip install -r requirements-embed.txt
```

`embed_env.sh` redirects every cache and temp directory (pip, Hugging Face, CUDA kernel
cache, temp) under one root so nothing lands in the user profile. Set `EMBED_ROOT` to
the folder holding `venv_embed/`, then `source embed_env.sh` before any embedding step.

**Always:**

```bash
export PYTHONHASHSEED=0          # blocking tokens are hashed with hash(); index and
                                 # queries must agree on the seed
export ER_ROOT=/path/to/root     # the directory that contains dataset/
```

Peak RSS is ~7 GB per process. A 6 GB+ GPU is needed for the embedding steps.

## Run order

```bash
# ---- 1. preprocessing                                          (~5 min)
python src/normalize.py train test
python src/split.py                        # frozen split, SEED=20260925

# ---- 2. sparse blocking index + per-field norms                (~20 min)
python src/block2.py train && python src/block3.py train
python src/block2.py test  && python src/block3.py test

# ---- 3. dense bi-encoder: encode every record, retrieve per entity   (GPU)
source embed_env.sh
$VENV/Scripts/python src/embed_encode.py --split train      # 12.5M records, ~36 min
$VENV/Scripts/python src/embed_encode.py --split test       # 11.7M records, ~35 min
$VENV/Scripts/python src/embed_retrieve.py --split train --queries work/fit_truth.tsv --tag fit --k 25
$VENV/Scripts/python src/embed_retrieve.py --split train --queries work/val_truth.tsv --tag val --k 100
$VENV/Scripts/python src/embed_retrieve.py --split test  --tag test --k 25    # ~10 min

# ---- 4. training set: sparse + dense candidates, 65 features  (~30 min)
python src/build_training2.py --out work/fitd --kn 60 --ka 60 --bn 8000 --ba 8000 \
    --neg-rate 0.22 --max-entities 120000 --dense work/emb/fit_dense.npz --dense-k 25

# ---- 5. matcher: LightGBM, 4-fold grouped CV + isotonic       (~25 min)
python src/train_matcher.py --data work/fitd --folds 4 --rounds 1500

# ---- 6. validate and choose the decision rule                 (~15 min)
python src/score_split.py --truth work/val_truth.tsv --out work/val_scored_dense.npz \
    --kn 60 --ka 60 --bn 8000 --ba 8000 --dense work/emb/val_dense.npz --dense-k 25
python src/decide.py --scored work/val_scored_dense.npz --write-rule
#   -> tau=0.90 delta=0.08 kmax=12, validation macro F0.5 0.96090

# ---- 7. test inference                         (~8.7 h as one process)
python src/run_pipeline2.py --split test --kn 60 --ka 60 --bn 8000 --ba 8000 \
    --dense work/emb/test_dense.npz --dense-k 25 --tag dense --shard 0/1
python src/run_pipeline2.py --split test --merge --shard 0/1 --tag dense --out-dir work/dense_out
#   To split the work across processes, add --start/--end (entity index range),
#   concatenate the outputs in order, and merge. Output is deterministic.

# ---- 8. conflict resolution -> the submission                 (~2 min)
python src/make_resolved.py --matching work/dense_out/matching_results.tsv \
    --scored work/dense_out/accepted_scores.tsv --out output/matching_results.tsv

# ---- 9. format check
python utils/validate_submission.py --matching output/matching_results.tsv --test-dir dataset/test
```

`make_resolved.py` awards each record claimed by more than one entity to its
highest-scoring claimant. `run_pipeline2.py` records the scores of accepted pairs in
`accepted_scores.tsv`, so no re-scoring pass is needed.

## Evidence behind the reported numbers (optional)

```bash
# blocking recall / reduction ratio (sparse only)
python src/measure_block3.py --kn 60 --ka 60 --bn 8000 --ba 8000

# validation feature cache -> what blocking misses, and what dense recovers
python src/cache_features.py --out work/valcache
python src/miss_profile.py
$VENV/Scripts/python src/embed_probe.py      # model x text selection
$VENV/Scripts/python src/embed_recall.py     # real recall of sparse ∪ dense

# 5-fold model x conflict-variant comparison on a competition-closed set
python src/cv_select.py
python src/cache_stream.py                   # 31.6M rows streamed to disk
python src/cv_train.py --kind lgb            # CPU
python src/cv_train.py --kind xgb            # GPU
python src/cv_train.py --kind cat            # GPU
python src/cv_variants.py

# false-merge patterns
python src/error_analysis.py --max-entities 20000 --dense work/emb/val_dense.npz
```

The reported k-fold results were produced with the sparse-only 60-feature model, before
the bi-encoder was added; pass `--dense` to `cache_stream.py` to repeat them on the
current feature set.

## Modules

| file | role |
|---|---|
| `src/common.py` | TSV IO; the normaliser; legal-form, address and state dictionaries |
| `src/normalize.py` | normalises all 24M records in parallel, caches parquet |
| `src/split.py` | S1-level split stratified by country x match cardinality |
| `src/evaluate.py` | macro F0.5 exactly as specified |
| `src/recon.py`, `src/recon_gt.py` | data profiling and ground-truth statistics |
| `src/block2.py` | sparse tokens and the postings index |
| `src/block3.py` | per-field cosine retrieval and per-record field norms |
| `src/embed_encode.py` | bi-encoder encoding of "name, address", float16, streamed to disk |
| `src/embed_retrieve.py` | exact GPU top-K per source within the same country |
| `src/pairs.py` | sparse ∪ dense candidates; corpus statistics |
| `src/features2.py` | the 65 pair features, absolute and per-entity relative |
| `src/engine.py` | chunk pipeline: candidates -> absolute -> relative features |
| `src/build_training2.py` | labelled pairs; negatives drawn from blocking |
| `src/train_matcher.py` | LightGBM, CV grouped by S1 entity, isotonic calibration |
| `src/score_split.py`, `src/decide.py` | scored validation cache; decision-rule sweep |
| `src/run_pipeline2.py` | test inference, sharding / explicit ranges, merge |
| `src/make_resolved.py` | one-to-many conflict resolution by score |
| `src/cv_select.py`, `src/cache_stream.py`, `src/cv_train.py`, `src/cv_variants.py` | the k-fold comparison |
| `src/cache_features.py`, `src/miss_profile.py`, `src/embed_probe.py`, `src/embed_recall.py` | bi-encoder evidence |
| `src/error_analysis.py`, `src/measure_block3.py`, `src/diag_tokens.py`, `src/diag_rank.py` | diagnostics |
| `src/make_package.py` | assembles the submission zip |

## Key results

| | |
|---|---|
| blocking recall (sparse ∪ dense) | 0.9855 |
| candidates / entity | 274.6 |
| validation macro F0.5 | **0.96090** |
| micro precision / recall | 0.9843 / 0.9277 |
| decision rule | `score ≥ 0.90`, `score ≥ best − 0.08`, at most 12 per entity |
