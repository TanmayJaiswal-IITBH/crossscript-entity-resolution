# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

A three-stage pipeline: an aggressive script-folding normaliser, per-field TF-IDF
cosine blocking whose name and address rankings are computed separately and
unioned, and a LightGBM pairwise matcher whose strongest signals are *relative* —
each candidate's rank and score-gap within its own entity's candidate set. The
two decisive ideas were (a) collapsing runs of repeated characters after
`unidecode`, which is what makes Devanagari/Tamil/Telugu/Kannada/Gujarati/Bengali
transliterations line up with their Latin counterparts, and (b) asking "is this
the best explanation for this entity" rather than "is this pair similar", which
moved macro F_0.5 from 0.706 to 0.912 on its own.

**Validation macro F_0.5: 0.94286** (39,999 held-out Source-1 entities, searched
against the full 5.0M/5.3M training pools).

---

## 2. Methodology

### 2.1 Problem Analysis

From EDA on the 2.21M / 5.03M / 5.29M training records:

* **Singletons are rare — 5.58%.** The predict-nothing baseline scores only
  0.056, the mean entity has 3.46 matches and 72% have three or more. Despite
  the precision-weighted metric this is a recall-hungry problem; a conservative
  "predict few" strategy is capped very low.
* **85% of non-singletons match records in both S2 and S3**, so retrieval must
  be balanced across pools rather than favouring one.
* **Matches never cross country** — 0 violations in 692,336 sampled pairs.
  Exploited as a hard constraint by prefixing every blocking token with country.
* **Names are frequently transliterated into a different script entirely.** Raw
  string similarity is useless on these; `unidecode` plus collapsing repeated
  characters (`limittedd` → `limited`, `praaivett` → `praivet`) recovers 70–95
  similarity.
* Other name noise: typos and character transpositions (`Aclhnmmy`, `Nati0nal`),
  word-order changes, legal-suffix drift, decorative prefixes (`The`, `Sri`,
  `Smt`, `>>`), DBA / "formerly known as", and names collapsed to a bare domain.
* Address noise: abbreviation, state name vs postal code vs native script,
  component reordering, leading zeros, perturbed house numbers, landmark
  references, and **~3% of S2/S3 addresses are empty**.
* **France appears only in the test set** (15% of test S1 entities) with no
  training representation. The pipeline is country-agnostic by construction —
  country is used only as an equality constraint and a token prefix — so France
  needs no special casing. Verified explicitly (section 5.3).

### 2.2 Solution Strategy

**Approach Type:** Blocking + GBDT pairwise classifier + per-entity decision rule

**Core Innovation:** Per-field cosine retrieval with a union, and relative
(rank-normalised) pair features. The relative block required restructuring
inference so both pools are scored in a single pass — with S2 and S3 in separate
processes, an entity's candidates are split across processes that never see each
other and the features cannot be computed at all.

---

## 3. Candidate Generation (Blocking)

Every record becomes a bag of country-prefixed tokens:

| family | content |
|---|---|
| `n<country><token>` | core name tokens (legal forms and honorifics removed) |
| `g<country><4gram>` | char 4-grams of the joined core name (typo tolerance) |
| `a<country><token>` | address tokens, alphabetic, length ≥ 3 |
| `d<country><run>` | address digit runs, leading zeros stripped |

A postings index maps token → record list; tokens with df > 60,000 are not
indexed. At query time each field's tokens are walked **rarest-first** and
admitted while a posting budget lasts (8,000 per field), with a per-token cap of
df ≤ 8,000. Scoring is binary-tf cosine — Σ idf² over shared tokens, divided by
the pool record's field norm. **Name and address produce separate top-60 lists
per pool source, and the union is the candidate set.**

* **Blocking recall: 0.9448**
* **Candidates per S1 entity: 235.8**
* **Reduction ratio: 0.99997715**

Per stratum: US 1-match 0.9615, US 2 0.9587, US 3+ 0.9588; India 1 0.9023,
India 2 0.9194, India 3+ 0.9246. The India gap is the transliteration cost.

### How we verified true matches were not lost

Measured, not assumed, on a frozen split against the **full** pools. A
token-evidence study showed only **0.11%** of true pairs share ≤1 token and
97.3% share at least one token that is rare (df ≤ 2000) in the target pool — so
residual loss is a ranking/budget cost, not a representation failure. Rank of the
true match under the union score: 83.0% within top 5, 91.0% within top 25.

### Blocking iterations

| version | design | recall |
|---|---|---|
| v1 | composite keys (sorted name tokens, digit signature + state) | 0.829 |
| v2 | IDF-weighted token retrieval, one combined field | 0.848 |
| v3 | per-field cosine, separate top-K, unioned (K=30, budget 4000) | 0.894 |
| v4 | same, widened to K=60, budget 8000 | **0.945** |

v1 failed because composite keys are brittle — one inserted word, one leading
zero or one reordered component changes the key entirely. v2 failed because a
raw IDF sum rewards verbose records; length normalisation fixed it. Widening to
v4 was only affordable once the matcher was strong enough to reject the extra
candidates (it scores non-matches at 0.003 on average), which is why blocking
and matching were tuned jointly rather than in sequence.

---

## 4. Matching Model

### 4.1 Features (60)

**Name (17):** Jaro-Winkler, Levenshtein ratio, token-set ratio, token-sort
ratio, partial ratio, token Jaccard, containment, longest-common-subsequence
ratio, char-3-gram cosine, rare-token IDF-weighted overlap, length ratio, token
counts (both sides), first-token match, last-token match, empty flag,
abbreviation/initialism detection.

**Legal form (1):** 4-way categorical — both missing / match / **mismatch** /
one-side-missing. Kept categorical deliberately: "Acme Corp" vs "Acme Ltd" is a
genuine mismatch signal, whereas "Acme" vs "Acme Ltd" is only a missing suffix
and carries far less evidence.

**Address (17):** Levenshtein, token-set, token-sort, token Jaccard,
containment, char-3-gram cosine, digit-run Jaccard, any-shared-digit-run,
street-number match, **street-number conflict**, postal-code match, city-token
overlap, state match, length ratio, empty flags (both sides), short-address
flag, landmark flag.

**Corpus statistics (7):** name rarity of each side (log df of the rarest core
token), **address saturation** of each side (log count of pool records sharing
that exact normalised address), digit conflict, and a generic-name-with-
saturated-address conjunction.

**Context (4):** the two blocking cosines, source indicator (S2/S3), country id.

**Relative, per entity (14):** candidate count, count of near-tied candidates,
the unsupervised blend `s0`, `s0` rank, gap to best, ratio to best, gap to
runner-up, name-similarity rank and gap, address-similarity rank and gap, rank
within the same pool source, is-best indicator, and within-entity z-score.

**The relative features dominate.** Top features by gain in the final model:
`s0_rank`, `s0_r_best`, `s0`, `a_contain`, `s0_rank_src`, `at_rank`. Five of the
top six are relative.

### 4.2 Model

LightGBM binary classifier, 127 leaves, lr 0.06, L2 = 2.0, ~1,100 trees.
**4-fold cross-validation grouped by Source-1 entity** — pairs from one entity
share the query side verbatim, so letting them straddle folds would leak and
inflate every estimate. **Isotonic calibration** fitted on the out-of-fold
predictions, so the decision threshold is an actual probability.

Training data: 120,000 `fit` entities (disjoint from validation by construction),
candidates drawn from the same blocking configuration used at inference so the
negative distribution matches what the model meets in production. 6.5M pairs,
393k positives, negatives subsampled at 0.22 (positives never subsampled).

Out-of-fold AP 0.99857, AUC 0.99991. An AP that high warrants suspicion, so it
was checked against the held-out validation entities; the gap is consistent with
the candidate population being mostly easy negatives (median candidate scores
0.0000; positives average 0.971, negatives 0.003).

### 4.3 Decision rule and why F_0.5 motivates it

Accept a candidate when

```
score >= 0.88   AND   score >= (entity's best score) - 0.08   AND   rank < 12
```

F_0.5 weights precision twice recall, and the metric is macro-averaged per
entity, so a false merge on a singleton costs a full 1.0 and an entity with one
true match where two are predicted scores 0.83. That pushes the threshold
upward and makes the rule *per entity* rather than global: the δ term suppresses
the long tail of weak candidates on entities that already have one confident
match.

A full sweep over a larger rule family — global τ, relative δ, a separate higher
threshold τ₂ for the 2nd and later matches, an explicit singleton gate τ_single,
and k_max — found that **τ₂ and τ_single both optimise to zero**. The complete
family beat a plain global threshold by only **+0.0004**. This is worth
recording: with a score distribution this separable, the *shape* of the decision
stops carrying information. The optimum is flat from τ = 0.76 to 0.90, which is
reassurance that the threshold is not fitted to validation noise.

Threshold selection used **local validation only**. No leaderboard feedback was
used to tune anything, to avoid overfitting the public split.

---

## 5. Results & Error Analysis

### 5.1 Progression

| stage | blocking recall | macro F_0.5 |
|---|---|---|
| hand-weighted blend, K=30 | 0.8944 | 0.70588 |
| + LightGBM matcher (52 features) | 0.8944 | 0.91164 |
| + wide blocking (K=60, budget 8000) | 0.9448 | 0.93023 |
| + matcher retrained on the wide distribution | 0.9448 | 0.93596 |
| + corpus-statistic and conflict features (60) | 0.9448 | **0.94286** |

At the final operating point: micro precision **0.9855**, micro recall 0.8889,
3.13 predicted matches per entity, 6.7% of entities predicted empty.

**Oracle decomposition.** A perfect matcher over the current candidate set would
score 0.97609. The matcher therefore realises 96.6% of what blocking makes
available; the remaining 0.024 is unreachable without better recall. This
decomposition is what redirected effort from the matcher to blocking mid-project.

### 5.2 Common false merges

Measured over 20,000 validation entities (tp = 60,894, fp = 1,121):

| # | pattern | share | description |
|---|---|---|---|
| 1 | sibling of a true match | 44.7% | a near-duplicate belonging to a *different* S1 entity — same name, same street, different house number |
| 2 | generic name | 24.8% | the S1 name has no identifying token, so the address carries the match while names diverge entirely |
| 3 | empty address, name only | 14.9% | candidate has no address; name-only evidence, usually token-reordered |
| 4 | same address, different business | 6.9% | genuinely distinct businesses in one building |
| 5 | name containment | 0.9% | parent/subsidiary or an added qualifier |

Patterns 1 and 4 together (52%) are one underlying problem: **address evidence
saturates in dense commercial buildings**, where many distinct businesses
normalise to an identical address string. Pattern 2 is its mirror — the name is
uninformative so the address wins by default. This analysis directly produced
the address-saturation, name-rarity and street-number-conflict features, which
moved F_0.5 from 0.93596 to 0.94286. Notably, chain/franchise merging and
holding-company overlap — both anticipated — did **not** appear in any volume.

### 5.3 Common false negatives, and France

10.6% of true pairs never enter the candidate set and are unreachable at any
threshold; 3.27% of non-singleton entities have zero true matches retrieved.
These are cross-script names whose address is also heavily abbreviated or
reordered, and records with an empty address plus a truncated or replaced name.

**Singletons** are 5.59% of entities; 90.2% are correctly predicted empty; they
contribute 5.38% of total score, and perfect singleton handling would add only
+0.0055.

**France check.** France normalises comparably to the trained countries across
all three sources — 0% empty cores, 19.0–19.4 blocking tokens per record against
19.2–22.6 for US/India, and address lengths between the two. French legal forms
(`sarl`, `sas`, `sasu`, `eurl`, `sci`) are in the legal-form dictionary and are
correctly stripped from the core name. Every France S1 entity appears in the
submission.

---

## 6. Conclusion

The problem is won in candidate generation and in *relative* features, not in
model sophistication: a GBDT over well-chosen features realises 96.6% of the
available ceiling, while the elaborate per-entity decision rule that intuition
suggests should matter was worth +0.0004. The most useful engineering habit was
building diagnostics that explained *why* pairs were missed or wrongly merged —
every one of the three score jumps came from reading errors rather than from
trying a different model.

---

## 7. Fair Play and Model Provenance

* **No external data of any kind was used.** No commercial entity-resolution
  APIs, no business registries, no government databases, no geocoding APIs, no
  internet data augmentation, no scraped or purchased data. Every statistic used
  by the pipeline — token document frequencies, address saturation counts,
  IDF weights, state/legal-form dictionaries — is computed from the provided
  training and test TSVs alone.
* **No pretrained model is used.** The final model is a LightGBM gradient-boosted
  tree ensemble trained from scratch on the provided training data.
  LightGBM is **MIT licensed**. Parameter count is not meaningfully comparable to
  a neural model, but the serialised model is ~1,100 trees at 127 leaves
  (≈1.4 × 10⁵ leaf values, ~18 MB), far below the 8-billion-parameter cap.
* **No sentence-transformer or language-model embeddings are used.** Adding
  Apache-2.0/MIT embedding models (MiniLM, BGE, E5) was considered and
  deliberately *not* done, because obtaining their weights requires an internet
  download and the rules prohibit external augmentation from internet sources.
  That ambiguity was left unresolved rather than risked.
* Other dependencies, all permissive: NumPy (BSD-3), pandas (BSD-3), PyArrow
  (Apache-2.0), scikit-learn (BSD-3), RapidFuzz (MIT), Unidecode (GPL-2.0 —
  used only for character transliteration during preprocessing, not part of the
  model). Pinned versions are in `requirements.txt`.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` holds the runnable pipeline; its `README.md`
gives the exact end-to-end command sequence. Entry points:

| file | role |
|---|---|
| `src/common.py` | TSV IO, the normaliser, legal-form / address / state dictionaries |
| `src/normalize.py` | normalises all 7 files once, caches parquet |
| `src/split.py` | frozen validation / fit split, `SEED = 20260925` |
| `src/block2.py` | token extraction and the postings index |
| `src/block3.py` | per-field cosine retrieval and per-record field norms |
| `src/pairs.py` | both-pool candidate generation, corpus statistics |
| `src/features2.py` | the 60 pair features, absolute and relative |
| `src/engine.py` | chunk pipeline: candidates → absolute → relative |
| `src/build_training2.py` | labelled pair dataset from the `fit` split |
| `src/train_matcher.py` | grouped-CV LightGBM + isotonic calibration |
| `src/decide.py` | decision-rule sweep with exact closed-form F_0.5 |
| `src/run_pipeline2.py` | sharded test inference → both output TSVs |
| `src/evaluate.py` | macro F_0.5 scorer with per-stratum breakdown |
| `src/error_analysis.py` | false-merge pattern classification |

`PYTHONHASHSEED=0` is required at every step: blocking tokens are hashed with
Python's built-in `hash()`, so index and queries must agree on the seed.

### B. Additional Results

Rank of the true match under each retrieval score (1,500 sampled queries):

| rank | name-cosine | addr-cosine | best-of-two |
|---|---|---|---|
| 0 | 20.2% | 30.7% | 42.5% |
| ≤4 | 48.0% | 67.7% | 83.0% |
| ≤24 | 63.1% | 77.0% | 91.0% |
| ≤99 | 73.9% | 82.0% | 94.9% |
| not retrieved | 17.2% | 12.0% | 2.3% |

This table is why name and address are retrieved separately: either field alone
leaves 12–17% of true pairs unreachable, while the union leaves 2.3%.
