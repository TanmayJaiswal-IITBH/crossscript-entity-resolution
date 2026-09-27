# crossscript-entity-resolution

Entity resolution across **10.3M noisy business records** in three scripts and three
countries, with no shared identifiers between sources.

Given a deduplicated reference source (Source 1) and two noisy sources (Source 2 and
Source 3), find every record across S2/S3 that refers to the same real-world business
as each S1 entity. An entity may have zero, one, or many matches.

| | |
|---|---|
| **Validation macro F₀.₅** | **0.9609** (+ conflict resolution) |
| Public leaderboard (previous TF-IDF version) | 0.925 |
| Blocking recall | **0.9855** |
| Micro precision / recall | 0.9843 / 0.9277 |
| Scale | 2.2M queries × 10.3M candidates (train) · 1.7M × 10.0M (test) |

Validation is a frozen, stratified held-out split of 39,999 Source-1 entities searched
against the **full** candidate pools. It tracks the leaderboard: the country mix of the
test set predicted 0.9253 for the TF-IDF version, and the portal returned 0.925.

---

## The interesting part of this problem

**Names are transliterated into entirely different scripts.** The same business appears
as `Tech Food Private Limited` and `टेक फूड प्राइवेट लिमिटेड`. Raw string similarity
scores these at ~0. The data contains Devanagari, Tamil, Telugu, Kannada, Gujarati and
Bengali.

**Addresses saturate.** In dense commercial buildings many distinct businesses normalise
to an identical address string — this was the root cause of over half of all false
merges.

**France appears only at test time** (15% of test entities), with no training data at all.

---

## Approach

**1 · Normalise** — `unidecode` → lowercase → strip punctuation → **collapse repeated
characters** (`limittedd → limited`). Legal forms and honorifics are folded out of the
core name.

**2 · Block — two retrievers, unioned**

* **Sparse:** country-prefixed tokens (name tokens, name char-4-grams, address tokens,
  digit runs), binary-tf cosine, name and address retrieved *separately*, top-60 each.
* **Dense:** a multilingual bi-encoder (`intfloat/multilingual-e5-small`, MIT, 118M
  parameters) embeds **name + address together**; exact GPU cosine search returns the
  top-25 per source within the same country.

| blocking | recall |
|---|---|
| composite keys | 0.829 |
| sparse, one combined field | 0.848 |
| sparse, per-field, unioned | 0.945 |
| **sparse ∪ dense** | **0.986** |

The two are complementary: dense alone reaches 0.962, sparse alone 0.945. Embedding the
name *with* its address is what makes the dense side work — name-only embeddings recover
0.5% of the cross-script pairs sparse misses; name + address recovers 68%.

**3 · Match** — LightGBM over **65 pair features**, cross-validated with folds grouped by
S1 entity and isotonically calibrated. The strongest features are **relative**: a
candidate's dense-similarity rank within its entity, its gap to the best candidate, its
rank by a similarity blend. The question that matters is not "is this pair similar" but
"is this the best explanation for this entity".

**4 · Decide** — accept when `score ≥ 0.90` and `score ≥ best − 0.08`, at most 12 per
entity.

**5 · Resolve conflicts** — Source 1 is deduplicated, so each S2/S3 record belongs to
**exactly one** S1 entity (verified: 0 of 7,638,365 ground-truth records are claimed
twice). A record accepted by several entities is awarded to its highest-scoring claimant.

---

## Results

| stage | blocking recall | validation F₀.₅ | leaderboard |
|---|---|---|---|
| hand-weighted similarity blend | 0.894 | 0.706 | 0.694 |
| + LightGBM matcher | 0.894 | 0.912 | |
| + wide sparse blocking | 0.945 | 0.930 | |
| + corpus-statistic & conflict features | 0.945 | 0.943 | **0.925** |
| **+ dense bi-encoder** | **0.986** | **0.961** | *pending* |

**Choosing the model — 5-fold cross-validation** on 133,780 train entities (every entity
in 8 whole states, so both sides of each conflict are scored), thresholds tuned nested:

| model + conflict handling | macro F₀.₅ | vs plain LightGBM | folds won |
|---|---|---|---|
| **LightGBM + keep-best** | **0.94841** | **+0.00219 ± 0.00028** | **5/5** |
| LightGBM+XGBoost blend + keep-best | 0.94841 | +0.00218 | 5/5 |
| LightGBM + drop-all | 0.94765 | +0.00142 | 5/5 |
| LightGBM (plain) | 0.94623 | — | — |
| XGBoost (plain) | 0.94547 | −0.00076 | 0/5 |
| CatBoost (plain) | 0.94318 | −0.00305 | 0/5 |

### What didn't work, and is worth knowing

* **An elaborate per-entity decision rule was worth +0.0004.** Separate thresholds for
  2nd+ matches and an explicit singleton gate both optimised to zero.
* **Model class barely matters.** Three gradient-boosting libraries land within 0.003;
  blending adds nothing.
* **Chains, franchises and holding-company overlap barely appear** among false merges.
  The real causes were address saturation and near-duplicate siblings.

### Known weakness: France

France is 15% of the test set and absent from training. It produces **52% of all
contested records** — ten times the US conflict rate — so the matcher over-matches French
records. Working back from the leaderboard score puts France near 0.85, against 0.96 for
US. Conflict resolution removes every collision; non-colliding false matches remain.

---

## Quickstart

Two environments: the base one for everything except embeddings, and a GPU one for the
bi-encoder. Full commands, in order, are in [PIPELINE_README.md](PIPELINE_README.md).

```bash
pip install -r requirements.txt                 # base environment
export PYTHONHASHSEED=0                         # blocking tokens are hashed with hash()
export ER_ROOT=/path/to/root                    # directory containing dataset/

python src/normalize.py train test
python src/split.py
python src/block2.py train && python src/block3.py train
# ... then embeddings, training, inference: see PIPELINE_README.md
```

## Layout

| path | role |
|---|---|
| `src/common.py`, `src/normalize.py` | IO, the script-folding normaliser, dictionaries |
| `src/split.py`, `src/evaluate.py` | frozen split; macro F₀.₅ scorer |
| `src/block2.py`, `src/block3.py` | sparse postings index; per-field cosine retrieval |
| `src/embed_encode.py`, `src/embed_retrieve.py` | bi-encoder encoding; dense GPU retrieval |
| `src/pairs.py`, `src/features2.py`, `src/engine.py` | candidate union, the 65 features |
| `src/build_training2.py`, `src/train_matcher.py` | training set; grouped-CV LightGBM + calibration |
| `src/score_split.py`, `src/decide.py` | validation scoring; decision-rule sweep |
| `src/run_pipeline2.py`, `src/make_resolved.py` | test inference; conflict resolution |
| `src/cv_*.py`, `src/cache_stream.py` | the 5-fold model × variant comparison |
| `src/embed_probe.py`, `src/embed_recall.py`, `src/miss_profile.py` | bi-encoder selection and recall evidence |
| `src/error_analysis.py`, `src/measure_block3.py`, `src/diag_*.py`, `src/recon*.py` | diagnostics behind every reported number |
| `config/embedding_models.json` | pinned bi-encoder revisions and licences |
| `work/matcher_lgb.txt`, `work/matcher_calib.pkl`, `work/best_rule.npy` | the submitted model |
| `output/matching_results.tsv` | the submitted predictions |
| [RESULTS.md](RESULTS.md) | full experiment log, including what failed |
| [Documentation_template.md](Documentation_template.md) | methodology write-up |

Superseded first iterations (composite-key blocking, the hand-weighted matcher, the
model-zoo scripts) are removed from the tree but remain in the git history.

## Data, models and provenance

The `dataset/` directory is **not committed**. `utils/validate_submission.py` is
organiser-provided.

**No external data is used** — no entity-resolution APIs, business registries,
geocoding services or lookups of any kind. Every statistic the pipeline uses is computed
from the provided TSVs.

**One pretrained model is used:** `intfloat/multilingual-e5-small` — **MIT licence,
~118M parameters** (the rules allow MIT/Apache-2.0 models up to 8B), pinned to revision
`614241f6` in `config/embedding_models.json`. It is used **frozen**, purely as a text
encoder over the provided business names and addresses; it is not fine-tuned and looks
nothing up. `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0) was evaluated for
comparison and not used.

The matcher is LightGBM (MIT), trained from scratch. `Unidecode` is GPL-2.0-or-later and
is used only for character transliteration during preprocessing.
