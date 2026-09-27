# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

Two retrievers — a sparse per-field TF-IDF index and a dense multilingual bi-encoder over
name + address — produce the candidates; a LightGBM matcher whose strongest signals are
**relative** (each candidate's rank and score gap within its own entity) scores them; and
a one-to-many constraint resolves records claimed by several entities. The decisive ideas
were (a) collapsing repeated characters after `unidecode`, which aligns Indic-script
transliterations with their Latin forms; (b) asking "is this the best explanation for
this entity" rather than "is this pair similar", worth ~0.2 F0.5 on its own; and
(c) embedding the **name together with its address**, which lifted blocking recall from
0.945 to 0.986.

**Validation macro F0.5: 0.96090** (39,999 held-out Source-1 entities, searched against
the full training pools).

---

## 2. Methodology

### 2.1 Problem Analysis

From EDA on the 2.21M / 5.03M / 5.29M training records:

* **Singletons are rare — 5.58%.** Predicting nothing scores 0.056. The mean entity has
  3.46 matches and 72% have three or more, so despite the precision-weighted metric this
  is a recall-hungry problem.
* **85% of non-singletons match records in both S2 and S3.**
* **Matches never cross country** — 0 of 692,336 sampled pairs. Used as a hard
  constraint in both retrievers.
* **Every S2/S3 record belongs to exactly one S1 entity** — 0 of 7,638,365 ground-truth
  records are claimed twice, because Source 1 is deduplicated. Used for conflict
  resolution (section 4.4).
* **Names are transliterated into another script entirely** (Devanagari, Tamil, Telugu,
  Kannada, Gujarati, Bengali). `unidecode` plus collapsing repeated characters
  (`limittedd -> limited`) recovers 70–95 similarity for most.
* Name noise: typos and transpositions, word-order changes, legal-suffix drift,
  decorative prefixes, DBA / "formerly known as", bare domains, and **alias names that
  share nothing with the business name** (`Upper Accounting PC` vs `Belocalo`).
* Address noise: abbreviations, state name vs code vs native script, reordering,
  leading zeros, perturbed house numbers, landmarks; **~3% of S2/S3 addresses are empty**.
* **France appears only in the test set** (15% of test entities). The pipeline is
  country-agnostic by construction, but France's accuracy cannot be measured locally
  (section 5.4).

### 2.2 Solution Strategy

**Approach type:** hybrid sparse + dense blocking -> GBDT pairwise classifier ->
per-entity decision rule -> one-to-many conflict resolution.

**Core innovations:** per-field retrieval with a union; relative (rank-normalised) pair
features; a bi-encoder over the name **and** address together; and conflict resolution
from the deduplication constraint.

---

## 3. Candidate Generation (Blocking)

### 3.1 Sparse retriever

Every record becomes a bag of country-prefixed tokens — core name tokens, name char
4-grams, address tokens, address digit runs. Tokens are walked rarest-first under a
posting budget (8,000 per field, per-token df cap 8,000); scoring is binary-tf cosine
(sum of idf² over shared tokens / the record's field norm). **Name and address are
retrieved separately**, top-60 each per source, and unioned: a true pair often agrees on
only one field (empty address, or a name in another script), and summing both fields
into one score buries those pairs.

### 3.2 Dense retriever

`intfloat/multilingual-e5-small` encodes every record as the text
`"query: <business_name>, <business_address>"` (384-d, float16, L2-normalised).
For each S1 entity, **exact** cosine search on the GPU returns the top-25 records per
source, restricted to the same country. 24.2M records were encoded (train 12.5M, test
11.7M) at ~5,900 records/s on an RTX 4060; test retrieval took 10 minutes.

**Why name + address.** What sparse blocking missed (7,654 validation pairs): 28.9%
cross-script names, 64.9% same-script names whose tokens are all too common to rank the
right record (`Johnson Holdings Group`), 6.1% alias names. A probe against 800,000
same-country distractors compared four configurations; recovery of the missed pairs at
K=50:

| encoder | name only | **name + address** |
|---|---|---|
| multilingual-e5-small | 25.3% | **73.6%** |
| paraphrase-multilingual-MiniLM-L12-v2 | 18.9% | 33.8% |

Name-only embeddings recovered **0.5%** of the cross-script pairs — e5 cannot align
`टेक फूड प्राइवेट लिमिटेड` with `Tech Food Private Limited` on the name alone. With the
address attached it recovers 68% of them: the model effectively performs fuzzy matching
over both fields at once, which per-field sparse retrieval cannot.

### 3.3 Union and measured recall

Measured against the **full** pools on validation:

| dense top-K per source | dense alone | **sparse ∪ dense** | new candidates / entity |
|---|---|---|---|
| — | — | 0.9448 | — |
| 10 | 0.9506 | 0.9821 | +12.2 |
| **25 (used)** | 0.9620 | **0.9855** | **+38.4** |
| 50 | 0.9684 | 0.9875 | +84.4 |
| 100 | 0.9741 | 0.9896 | +179.2 |

K=25 was chosen for cost: K=50 adds only 0.002 recall for more than double the extra
inference work. Each retriever catches matches the other misses, which is why the dense
side **augments** rather than replaces the sparse one.

* **Blocking recall: 0.9855** · **candidates per entity: 274.6** · reduction ratio ~0.99997

### 3.4 Iterations

| version | design | recall |
|---|---|---|
| v1 | composite keys (sorted tokens, digit signature + state) | 0.829 |
| v2 | IDF-weighted tokens, one combined field | 0.848 |
| v3 | per-field cosine, separate top-K, unioned | 0.894 |
| v4 | v3 widened to K=60, budget 8000 | 0.945 |
| **v5** | **v4 ∪ dense bi-encoder top-25** | **0.986** |

---

## 4. Matching Model

### 4.1 Features (65)

**Name (17):** Jaro-Winkler, Levenshtein ratio, token-set, token-sort, partial ratio,
token Jaccard, containment, LCS ratio, char-3-gram cosine, rare-token overlap, length
ratio, token counts, first/last-token match, empty flag, initialism detection.

**Legal form (1):** 4-way categorical — both missing / match / **mismatch** /
one-side-missing. "Acme Corp" vs "Acme Ltd" is real evidence against a match; "Acme" vs
"Acme Ltd" is only a missing suffix.

**Address (17):** Levenshtein, token-set, token-sort, Jaccard, containment, char-3-gram
cosine, digit-run Jaccard, any shared digit run, street-number match, postal match, city
overlap, state match, length ratio, empty flags, short-address flag, landmark flag.

**Corpus statistics & conflicts (7):** name rarity of each side, **address saturation**
of each side (how many pool records share that exact address), street-number conflict,
digit conflict, generic-name-with-saturated-address conjunction — added from the
false-merge analysis.

**Dense (3):** bi-encoder cosine, rank within the dense top-K, in-dense-top-K flag. For a
candidate outside the dense top-K the K-th cosine is used: its true cosine is known to be
at most that.

**Context (4):** the two sparse cosines, source indicator, country id.

**Relative, per entity (16):** candidate count, near-tied count, a similarity blend and
its rank / gap to best / ratio to best / gap to runner-up, name-similarity and
address-similarity rank and gap, rank within source, is-best flag, within-entity
z-score, **dense-cosine gap to the entity's best and rank**.

Top features by gain in the final model: `dcos_rank`, `dense_cos`, `dense_rank`, `s0`,
`s0_z` — the model reorganised itself around the bi-encoder's relative ranking.

### 4.2 Model and training

LightGBM binary classifier (127 leaves, lr 0.06, L2 2.0), **4-fold cross-validation
grouped by Source-1 entity** — pairs of one entity share the query side verbatim, so
letting them straddle folds would leak — and **isotonic calibration** on the out-of-fold
predictions. Training data: 120,000 entities from a split disjoint from validation;
7.56M candidate pairs from the same sparse ∪ dense blocking used at inference (so the
negative distribution matches production); 409,475 positives; negatives subsampled at
0.22, positives never.

### 4.3 Model selection — 5-fold cross-validation

Three gradient-boosting libraries and four conflict-handling variants were compared by
5-fold CV on 133,780 train entities. The evaluation set is **every** entity in eight
whole states (4 US, 4 Indian): conflict resolution only acts when both competing
entities are scored, and a random sample almost never contains both. Thresholds were
tuned **nested** (chosen on four folds, scored on the fifth). Run on the sparse-only
60-feature model:

| model + conflict handling | macro F0.5 | vs plain LightGBM | folds won |
|---|---|---|---|
| **LightGBM + keep-best** | **0.94841** | **+0.00219 ± 0.00028** | **5/5** |
| LightGBM+XGBoost blend + keep-best | 0.94841 | +0.00218 ± 0.00052 | 5/5 |
| LightGBM + margin | 0.94794 | +0.00172 | 5/5 |
| LightGBM + drop-all | 0.94765 | +0.00142 | 5/5 |
| LightGBM (plain) | 0.94623 | — | — |
| XGBoost (plain) | 0.94547 | −0.00076 | 0/5 |
| CatBoost (plain) | 0.94318 | −0.00305 | 0/5 |

Keep-best beat the plain rule for every model in every fold; the ordering keep-best >
margin > drop-all > none held for all models. LightGBM is the best single model; the
blend ties it at twice the cost. All five folds independently chose τ=0.88, δ=0.08.

### 4.4 Decision rule and conflict resolution

Accept a candidate when `score ≥ 0.90` **and** `score ≥ (entity's best) − 0.08`, at most
12 per entity. F0.5 weights precision twice and is macro-averaged per entity — a false
merge on a singleton costs a full 1.0 — which pushes τ up and makes the rule per-entity.
A wider rule family (a higher bar for 2nd+ matches, an explicit singleton gate) added
**+0.0004**; both extra parameters optimised to zero. Thresholds were chosen on local
validation only, never on the leaderboard.

Then **keep-best**: a record accepted by several entities goes to its highest-scoring
claimant. On test this resolved 56,531 contested records and removed 89,501 losing
claims.

---

## 5. Results & Error Analysis

### 5.1 Progression

| stage | blocking recall | validation F0.5 | leaderboard |
|---|---|---|---|
| hand-weighted similarity blend | 0.8944 | 0.70588 | 0.694 |
| + LightGBM matcher (52 features) | 0.8944 | 0.91164 | |
| + wide sparse blocking | 0.9448 | 0.93023 | |
| + retrained on the wide candidate distribution | 0.9448 | 0.93596 | |
| + corpus-statistic & conflict features (60) | 0.9448 | 0.94286 | **0.925** |
| **+ dense bi-encoder (65 features)** | **0.9855** | **0.96090** | |

At the final operating point: micro precision 0.9843, micro recall 0.9277; US 0.9707,
India 0.9462. A perfect matcher on these candidates would score 0.9952, so the remaining
headroom is now in the matcher, not in blocking.

### 5.2 Validation tracks the leaderboard

From per-country validation scores and the test set's country mix (US 38.3%, India 46.8%,
France 15.0%), the sparse model's leaderboard score was predicted at 0.9253 assuming
France ≈ 0.85; the portal returned **0.925**. The whole gap to validation (0.943) is the
harder country mix.

### 5.3 Common false merges

Over 20,000 validation entities (precision 0.982 before the corpus-statistic features):

| pattern | share | description |
|---|---|---|
| sibling of a true match | 44.7% | same name, same street, different house number, different entity |
| generic name | 24.8% | no identifying name token; the address carries the match |
| empty address | 14.9% | name-only evidence on a candidate with no address |
| same address, different business | 6.9% | distinct businesses in one building |
| name containment | 0.9% | parent/subsidiary or an added qualifier |

Patterns 1 and 4 are one problem — **address saturation** in dense commercial buildings
— and produced the address-saturation, name-rarity and street-number-conflict features.
Chain/franchise merging and holding-company overlap did not appear in any volume.

### 5.4 False negatives, singletons and France

* 1.45% of true pairs never enter the candidate set; alias names (0.34% of all true
  pairs) are unreachable by any similarity method.
* Singletons are 5.58% of entities; ~90% are correctly predicted empty; perfect singleton
  handling would add only ~0.005.
* **France** (no training data) produced **52% of all contested test records** — 5.15%
  of its predicted pairs sat in a conflict, ten times the US rate — and working back from
  the leaderboard puts France near 0.85. Adding the bi-encoder doubled France's conflicts
  (18,625 -> 39,328 records), i.e. the matcher over-matches French records; conflict
  resolution removes every collision but not isolated false matches. France's normaliser
  output is otherwise healthy (0% empty names, token counts in line with US/India, French
  legal forms recognised).

---

## 6. Conclusion

Candidate generation and relative features, not model sophistication, decided this
problem: three GBDT libraries land within 0.003, and an elaborate decision rule was
worth 0.0004, while the name-plus-address bi-encoder alone added 0.018. Every jump came
from a diagnostic that explained *why* pairs were missed or wrongly merged. The largest
remaining opportunity is France, the one slice with no training signal.

---

## 7. Fair Play and Model Provenance

* **No external data of any kind was used** — no entity-resolution APIs, business
  registries, government databases, geocoding services, internet data augmentation, or
  lookups of business identities. Every statistic the pipeline uses (token document
  frequencies, address-saturation counts, IDF weights, dictionaries) is computed from the
  provided training and test TSVs.
* **One pretrained model is used: `intfloat/multilingual-e5-small`.**
  * Licence: **MIT** (verified from the model card before download).
  * Size: **~118M parameters** — far below the 8-billion-parameter cap.
  * Source and pin: Hugging Face, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`
    (`config/embedding_models.json`).
  * Use: **frozen** (not fine-tuned), purely as a text encoder over the provided business
    names and addresses. It produces vectors for similarity search; it retrieves no
    information about any business.
* `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, ~118M
  parameters, revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`) was downloaded and
  evaluated for model selection only; it is **not** used in the submission.
* The matcher is **LightGBM (MIT)**, trained from scratch on the provided training data
  (1,722 trees, 127 leaves, 65 features, 24.3 MB serialised). XGBoost (Apache-2.0) and CatBoost
  (Apache-2.0) were used only in the model comparison.
* Other dependencies: NumPy, pandas, SciPy, scikit-learn (BSD-3); PyArrow (Apache-2.0);
  RapidFuzz (MIT); PyTorch (BSD-style); transformers / sentence-transformers
  (Apache-2.0); **Unidecode (GPL-2.0-or-later)**, used only for character
  transliteration in preprocessing. Pinned versions: `requirements.txt`,
  `requirements-embed.txt`.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains the runnable pipeline; its `README.md` gives
the exact command sequence (two environments: base, and a GPU one for the bi-encoder).

| file | role |
|---|---|
| `src/normalize.py`, `src/common.py` | the normaliser, cached for all 24M records |
| `src/split.py`, `src/evaluate.py` | frozen split (`SEED=20260925`); macro F0.5 scorer |
| `src/block2.py`, `src/block3.py` | sparse postings index; per-field cosine retrieval |
| `src/embed_encode.py`, `src/embed_retrieve.py` | bi-encoder encoding; exact GPU retrieval |
| `src/pairs.py`, `src/features2.py`, `src/engine.py` | sparse ∪ dense candidates; 65 features |
| `src/build_training2.py`, `src/train_matcher.py` | labelled pairs; grouped-CV LightGBM + isotonic |
| `src/score_split.py`, `src/decide.py` | validation scoring; decision-rule sweep |
| `src/run_pipeline2.py`, `src/make_resolved.py` | test inference; conflict resolution |
| `src/cv_select.py` ... `src/cv_variants.py` | the 5-fold model x variant comparison |

`PYTHONHASHSEED=0` is required at every step: sparse blocking tokens are hashed with
Python's built-in `hash()`, so index and queries must agree on the seed. Inference is
deterministic — two processes computing the same 999 test entities independently
produced byte-identical output.

### B. Additional Results

Rank of the true match under the sparse retrieval scores (1,500 sampled queries):

| rank | name cosine | address cosine | best of two |
|---|---|---|---|
| 0 | 20.2% | 30.7% | 42.5% |
| ≤ 4 | 48.0% | 67.7% | 83.0% |
| ≤ 24 | 63.1% | 77.0% | 91.0% |
| ≤ 99 | 73.9% | 82.0% | 94.9% |
| not retrieved | 17.2% | 12.0% | 2.3% |

This is why name and address are retrieved separately: either alone leaves 12–17% of
true pairs unreachable, while the union leaves 2.3%.
