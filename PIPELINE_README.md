# Business Entity Resolution — pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the
provided TSVs. **No external data, no network access, no pretrained models.**

## Environment

```bash
pip install -r requirements.txt
```

**Paths.** The pipeline expects `dataset/` and a writable `work/` under a single
root. In this submission package the source sits at
`code/business_entity_resolution/src/` while the data stays at the challenge
root, so set `ER_ROOT` to the directory that contains `dataset/`:

```bash
export ER_ROOT=/path/to/student_resource
```

Without it the root defaults to the parent of `src/`, which is correct only when
the code is run in place.

`PYTHONHASHSEED=0` must be set for every step. Blocking tokens are hashed with
Python's built-in `hash()`, so the index and the queries must agree on the seed.
Peak RSS is ~7 GB per process; the full test run writes ~4 GB of intermediates.

## Run order

```bash
export PYTHONHASHSEED=0

# 1. normalise all 7 source files once            (~4 min)  -> work/*_norm.parquet
python src/normalize.py train test

# 2. frozen validation / fit split, SEED=20260925 (~1 min)  -> work/{val,fit}_*.tsv
python src/split.py

# 3. blocking index + per-field norms, per split  (~20 min)
python src/block2.py train && python src/block3.py train
python src/block2.py test  && python src/block3.py test

# 4. labelled pair dataset from the fit entities  (~30 min) -> work/fitw_{X,y,g}.npy
python src/build_training2.py --out work/fitw \
    --kn 60 --ka 60 --bn 8000 --ba 8000 --neg-rate 0.22 --max-entities 120000

# 5. train the matcher: grouped CV + isotonic     (~17 min) -> work/matcher_lgb.txt
python src/train_matcher.py --data work/fitw --folds 4 --rounds 1400

# 6. score validation and tune the decision rule  (~11 min) -> work/best_rule.npy
python src/score_split.py --truth work/val_truth.tsv --out work/val_scored.npz \
    --kn 60 --ka 60 --bn 8000 --ba 8000
python src/decide.py --scored work/val_scored.npz

# 7. test inference, sharded                      (~5 h with 2 shards)
python src/run_pipeline2.py --split test --shard 0/2 --kn 60 --ka 60 --bn 8000 --ba 8000 --tag v3 &
python src/run_pipeline2.py --split test --shard 1/2 --kn 60 --ka 60 --bn 8000 --ba 8000 --tag v3 &
wait
python src/run_pipeline2.py --split test --shard 0/2 --merge --tag v3 --out-dir output

# 8. format check
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Diagnostics (not required to reproduce the submission):

```bash
python src/measure_block3.py --kn 60 --ka 60 --bn 8000 --ba 8000   # recall / reduction ratio
python src/error_analysis.py --max-entities 20000                  # false-merge patterns
python src/evaluate.py --pred <pred.tsv> --truth work/val_truth.tsv --breakdown work/val_meta.tsv
```

## Modules

| file | role |
|---|---|
| `src/common.py` | TSV IO; the normaliser (unidecode → lowercase → strip punctuation → collapse repeated characters); legal-form, address-abbreviation and state dictionaries |
| `src/normalize.py` | applies the normaliser to all 24M records in parallel, caches parquet |
| `src/split.py` | S1-level split stratified by country × match-cardinality bucket |
| `src/block2.py` | token extraction (name tokens, name char-4-grams, address tokens, address digit runs; all country-prefixed) and the postings index |
| `src/block3.py` | per-field cosine retrieval — name and address scored and truncated separately, then unioned — and the per-record field norms |
| `src/pairs.py` | both-pool candidate generation; per-record corpus statistics (name rarity, address saturation) |
| `src/features2.py` | the 60 pair features: absolute similarity plus per-entity relative features |
| `src/engine.py` | chunk pipeline: candidates → absolute features → relative features |
| `src/build_training2.py` | labelled pair dataset; negatives come from blocking, matching the inference distribution |
| `src/train_matcher.py` | LightGBM with CV grouped by S1 entity, plus isotonic calibration |
| `src/decide.py` | decision-rule sweep, exact closed-form F_0.5 vectorised over the split |
| `src/run_pipeline2.py` | sharded test inference and the merge into both output TSVs |
| `src/evaluate.py` | macro F_0.5 scorer, per-stratum breakdown |
| `src/error_analysis.py` | false-merge pattern classification with worked examples |
| `src/measure_block3.py` | blocking recall and reduction ratio on the frozen split |

## Key results

| | |
|---|---|
| blocking recall | 0.9448 |
| candidates / entity | 235.8 |
| reduction ratio | 0.99997715 |
| validation macro F_0.5 | **0.94286** |
| micro precision / recall | 0.9855 / 0.8889 |
| decision rule | `score ≥ 0.88` and `score ≥ best − 0.08`, at most 12 per entity |
