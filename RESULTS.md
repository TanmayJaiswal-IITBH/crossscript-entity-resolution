# Day 1 log — Business Entity Resolution

Frozen artefacts: `SEED = 20260925`, split in `src/split.py`, scorer in `src/evaluate.py`.
Every number below is on the **same** validation split (39,999 S1 entities) searched
against the **full** 5.03M S2 / 5.29M S3 training pools.

## Block 1 — Data recon

| file | rows | countries | null address |
|---|---|---|---|
| train_source1 | 2,206,821 | US 1.32M, India 883k | 0% |
| train_source2 | 5,034,616 | US 3.02M, India 2.02M | 3.36% |
| train_source3 | 5,285,603 | US 3.17M, India 2.12M | 3.33% |
| test_source1 | 1,732,544 | India 810k, US 663k, **France 259k** | 0% |
| test_source2 | 4,887,273 | India 2.31M, US 1.87M, France 703k | 2.65% |
| test_source3 | 5,082,316 | India 2.41M, US 1.95M, France 732k | 2.68% |

`entity_id` prefixes are clean (`S1-`/`S2-`/`S3-`), no duplicates in any file.

### Ground-truth profile (the number that drives everything)

| matches | entities | share |
|---|---|---|
| 0 (singleton) | 123,247 | **5.58%** |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3+ | 1,589,205 | 72.01% |

mean 3.46 matches per entity, max 11, 7,638,365 true pairs total.

> **The predict-nothing floor is 0.056, not 0.30.** Singletons are rare, so this is a
> recall-hungry problem despite the precision-weighted metric. 72% of entities have
> three or more matches, and 85% of non-singletons match records in *both* S2 and S3.
> Any strategy built around "be conservative, predict few" is capped very low.

**Matches never cross country** — 0 violations in 692,336 sampled pairs. Country is
therefore baked into every blocking key as a hard constraint.

### Noise patterns (from manual inspection of ~50 true groups)

* **Full-script transliteration.** Indian names appear in Devanagari, Tamil, Telugu,
  Kannada, Gujarati and Bengali. Raw string similarity is useless on these.
  `unidecode` + collapsing repeated characters bridges most of the gap
  (`limittedd` → `limited`, `praaivett` → `praivet`), lifting typical pair similarity
  from ~0 to 70–95.
* Name: typos and character swaps (`Aclhnmmy`, `Cotllere`, `Nati0nal`), word-order
  transposition, legal-suffix drift (Pvt/Private/प्रा.), decorative prefixes
  (`The`, `Sri`, `Smt`, `>>`, `--`), DBA / `formerly known as`, and names collapsed to
  a bare domain (`hrsinvestments.com`).
* Address: abbreviation (Rd/Road, St/Street), state name vs postal code vs native
  script, component reordering, leading zeros on house numbers (`1857` vs `01857`),
  perturbed house numbers (`7327` vs `7317`), landmark references, and ~3% empty.

## Block 2 — Scorer and split

`src/evaluate.py` implements the spec formula exactly and reproduces the spec's worked
example (0.7143). Singleton = 1.0 for an empty prediction, 0.0 otherwise.

Split is at the S1-entity level, stratified by country × match-cardinality bucket,
carrying every true match with the entity. S2/S3 are **not** subsampled, so blocking
recall measured on validation transfers to test (the query:pool ratio is nearly
identical: train 2.21M:10.3M, test 1.73M:9.97M). Validation singleton rate 5.59%
vs 5.58% in the population.

## Block 3 — Blocking

Three iterations, each driven by a diagnostic rather than a guess:

| version | design | recall |
|---|---|---|
| v1 | composite keys (sorted name tokens, digit-signature + state, token pairs), df cap 400 | 0.829 |
| v2 | IDF-weighted token retrieval, single combined field | 0.848 |
| v3 | **per-field cosine (name / address), retrieved separately and unioned** | **0.894** |

What the diagnostics showed:

1. Composite keys are brittle — one inserted word, one leading zero, one reordered
   component and the key changes completely.
2. Token evidence is *not* the bottleneck: only **0.11%** of true pairs share ≤1 token,
   and 97.3% share at least one token that is rare (df ≤ 2000) in the target pool.
   The failures were ranking failures.
3. Ranking by a raw IDF sum rewards verbose records. Dividing by the record's field
   norm (binary-tf cosine) fixed that.
4. A true pair often agrees on exactly **one** field — the address is empty so only
   the name can match, or the name is transliterated so only the address can match.
   Summing both fields into one score buries these; retrieving top-K per field and
   taking the union recovers them.

Rank of the true match under the final scoring (1,500 sampled queries, 5,201 pairs):

| rank | name-cosine | addr-cosine | best-of-two |
|---|---|---|---|
| 0 | 20.2% | 30.7% | 42.5% |
| ≤4 | 48.0% | 67.7% | **83.0%** |
| ≤24 | 63.1% | 77.0% | **91.0%** |
| ≤49 | 68.9% | 79.6% | 93.2% |
| ≤99 | 73.9% | 82.0% | 94.9% |
| not retrieved | 17.2% | 12.0% | 2.3% |

**Operating point:** K=30 per field per source, posting budget 4000, per-token df cap 8000.

* blocking recall **0.8944**
* candidates/query **117.0**
* reduction ratio **0.99998866**

Per stratum: US ≈ 0.90, India ≈ 0.88; the India gap is the transliteration cost.

This is below the 98% target and is the single biggest lever available. The rank table
says the headroom is real but expensive: reaching ~95% needs K≈100 per field, ~97.7%
needs K≈600. Day 2 should buy recall with a cheaper representation (a learned or
phonetic cross-script name key) rather than by raising K.

## Block 4 — Baseline matcher

24 pairwise features (rapidfuzz ratio / token-set / token-sort / partial /
Jaro-Winkler on the normalised name, token and digit-run Jaccard on the address,
containment, length ratios, missing-field flags, plus the two blocking cosines),
combined by a hand-weighted blend that falls back to name-only evidence when the
candidate's address is empty.

Decision rule tuned on validation: keep a candidate when
`score >= 0.82` **and** `score >= 0.88 * (this entity's best score)`.

| threshold rule | macro F_0.5 |
|---|---|
| 0.72 absolute (untuned) | 0.62454 |
| **0.82 absolute + 0.88 relative (tuned)** | **0.70588** |

Tuning moved micro precision from 0.548 to **0.797** (recall 0.794 → 0.607) — precision
was the binding constraint, exactly as F_0.5 implies. At the tuned point: 2.64 predicted
matches per entity, 7.9% of entities predicted empty.

Per-stratum F_0.5 at the tuned point:

| stratum | n | F_0.5 |
|---|---|---|
| India, 0 matches | 895 | 0.458 |
| India, 1 | 860 | 0.463 |
| India, 2 | 2,718 | 0.594 |
| India, 3+ | 11,536 | 0.690 |
| US, 0 matches | 1,339 | 0.506 |
| US, 1 | 1,299 | 0.576 |
| US, 2 | 4,083 | 0.721 |
| US, 3+ | 17,269 | 0.780 |

The weak cells are singletons (~0.48 — we still emit a false match for half of them)
and 1-match entities (~0.52), where a single false positive halves the score. India
trails US by ~0.09 across every bucket, which is the transliteration cost showing up
again downstream of blocking.

## Block 5 — Local vs leaderboard

| run | local val F_0.5 | public LB F_0.5 | gap |
|---|---|---|---|
| baseline (tuned blend) | 0.70588 | _(to fill after upload)_ | |

If the gap exceeds ~0.03 the split is suspect and is the first thing to fix on Day 2.
Note one structural difference the split cannot capture: **France is in test only**
(15% of test S1 entities) and has no training representation at all. The pipeline is
country-agnostic by construction — country is only ever used as an equality
constraint and as a token prefix — so France is handled, but its error rate is
unmeasurable locally.

## Day 2 priorities

1. **Blocking recall** (0.894 → target 0.95+). Biggest ceiling. Cross-script name
   representation is the specific gap.
2. **Learned pair classifier** (LightGBM on the 200k-entity `fit` split, already cut
   and disjoint from validation) to replace the hand-weighted blend.
3. **Singleton / no-match decision** as a separate per-entity model rather than a
   by-product of the threshold.

---

# Day 2 log — supervised matcher

Same frozen split (`SEED = 20260925`, 39,999 val entities, 138,584 true matches —
regenerated bit-identically after the move to E:, so every Day-1 number remains
directly comparable).

## Headline

| stage | blocking recall | macro F_0.5 |
|---|---|---|
| Day 1: hand-weighted blend, K=30/budget 4000 | 0.8944 | 0.70588 |
| + LightGBM matcher (52 features) | 0.8944 | 0.91164 |
| + wide blocking K=60/budget 8000 | 0.9448 | 0.93023 |
| + matcher retrained on the wide candidate distribution | 0.9448 | **0.93596** |

Final rule: `score >= 0.86` and `score >= best - 0.08`, at most 12 per entity.
3.09 predicted matches per entity, 6.6% predicted empty, micro precision 0.982.

## Block 1 — features (24 -> 52)

Added: LCS ratio, char-3-gram cosine, rare-token IDF-weighted overlap (token
length as the IDF proxy), a 4-way legal-form categorical, postal-code and
street-number equality, city-token overlap, state match, landmark and
missingness flags, abbreviation detection, and **13 per-entity relative
features** (rank, gap to best, ratio to best, gap to runner-up, within-source
rank, within-entity z-score, candidate count).

The relative features required an architectural change: Day 1 ran S2 and S3 as
separate processes, so an entity's candidates were split across two processes
that never saw each other. Retrieval now holds both pools in one pass, and
parallelism moved to query sharding.

Two bugs found by a unit test on the feature module:

* `L.L.C.` normalises to `l l c`, so the most common US legal form was never
  detected. Runs of single letters are now glued before the legal lookup.
  `corp` vs `ltd` correctly registers as a *mismatch*, which is the signal that
  separates "Acme Corp" from "Acme Ltd".
* `build_norms` was single-threaded (200s+ per source); now parallel.

## Block 2 — classifier

LightGBM, 4-fold **grouped by S1 entity** (pairs from one entity share the query
side verbatim; letting them straddle folds leaks). Isotonic calibration fitted on
the out-of-fold predictions.

* narrow training set: 7.45M pairs, 620k positives (8.33%), OOF AP 0.99815
* wide training set: 6.53M pairs, 393k positives (6.02%), OOF AP 0.99806

An OOF AP of 0.998 is high enough to suspect leakage, so it was checked against
the validation split, whose entities are disjoint from `fit` by construction: the
gap between fit and val performance is consistent with the population being
mostly easy negatives (the median candidate scores 0.0000; positives average
0.971, negatives 0.0027). Blocking recall reproduced at exactly 0.8944 on both
runs, confirming the comparison is clean.

Top features by gain, wide model: `s0_rank`, `s0_r_best`, `s0`, `a_contain`,
`s0_rank_src`, `s0_z` — five of the top six are **relative**.

## Block 3 — decision rule: the plan's prediction did not hold

Full sweep over tau x delta x tau2 x tau_singleton x k_max (~10k combinations):

```
BEST F0.5 = 0.91164   tau=0.80 delta=0.12 tau2=0.00 tau_single=0.00 kmax=12
ablation:
  global threshold only   0.91121
  + full rule family      0.91164   (+0.00043)
```

The rule family was budgeted as "where F_0.5 is won" and returned **+0.0004**.
Both tau2 (a higher bar for 2nd+ matches) and the explicit singleton gate
optimised to zero — they contribute nothing. The premise is sound for a weak
scorer, but this model's score distribution is nearly separable, so the *shape*
of the decision stops carrying information. The optimum is flat from tau=0.76 to
0.90, which is reassurance that the threshold is not tuned to validation noise.

What the plan expected from Block 3 was really delivered by Block 1: the
rank-normalised features are the single biggest win, worth ~0.2 F_0.5.

## Blocking became the dominant lever

An oracle decomposition (perfect matcher on the retrieved candidates) reframed
the priorities:

| | oracle F_0.5 | matcher headroom | blocking headroom |
|---|---|---|---|
| K=30, budget 4000 | 0.94595 | 0.035 | 0.054 |
| K=60, budget 8000 | 0.97609 | 0.040 | 0.024 |

Because the matcher scores junk at 0.0027, widening blocking is nearly free on
precision — the model simply rejects the extra candidates. Recall went
0.8944 -> 0.9448 (235.8 candidates/query, reduction ratio 0.99997715); entities
missing at least one match fell from 21.7% to 13.1%. India 3+ improved
0.882 -> 0.925, US 3+ 0.903 -> 0.959.

## Block 5 — the top false-merge patterns

Micro precision 0.9819 (tp=60,894 fp=1,121) over 20,000 validation entities.

| # | pattern | share | what it is |
|---|---|---|---|
| 1 | `sibling_of_a_true_match` | 44.7% | a near-duplicate record that belongs to a *different* S1 entity — same name, same street, different house number (`frye santana inc` at 108 Tower St vs 19 Tower St) |
| 2 | `generic_name` | 24.8% | the S1 name has no rare token, so the address carries the match while the names diverge entirely (`pediatric partners` vs `wexecto`, identical address) |
| 3 | `empty_address_name_only` | 14.9% | candidate has no address; name-only evidence, usually with reordered tokens |
| 4 | `same_address_different_name` | 6.9% | genuinely different businesses in one building (`rachana co` vs `reliant co`, identical Delhi address) |
| 5 | `name_containment` | 0.9% | parent/subsidiary or added qualifier |

Patterns 1 and 4 together (52%) are the same underlying problem: **address
evidence saturates in dense commercial buildings**, where dozens of distinct
businesses share an address string that is identical after normalisation. The
model has no way to tell "same building" from "same business" when the name
signal is weak. Pattern 2 is the mirror image — the name is uninformative, so
the address wins by default.

Neither classic chain/franchise merging nor holding-company overlap shows up in
any volume, which is worth noting because both were anticipated.

## Day 3 priorities

1. **Discriminative address handling.** Half of all false merges come from
   address saturation. A feature for "how many pool records share this exact
   normalised address" would let the model discount address evidence exactly
   where it is uninformative, and it needs no new data.
2. **Blocking recall 0.945 -> higher.** Still 0.024 of headroom, concentrated in
   cross-script Indian names. A phonetic/consonant-skeleton token built only from
   provided data is the no-licence-risk option.
3. **Name rarity as a feature.** `generic_name` is 25% of false merges and the
   df table needed to detect it already exists in the blocking index.

---

# Day 4 log — leaderboard calibration, model selection, bi-encoder

## Leaderboard check

| file | validation F0.5 | leaderboard |
|---|---|---|
| Day-1 hand-weighted blend | 0.706 | 0.694 |
| Day-3 LightGBM, sparse blocking | 0.943 | **0.925** |

The first 0.694 upload turned out to be the Day-1 file, not LightGBM. For the LightGBM
file, per-country validation scores weighted by the test mix (US 38.3%, India 46.8%,
France 15.0%) predicted **0.9253 with France at 0.85**; the portal returned 0.925.
Validation tracks the leaderboard, and France is the weak slice.

## Structure found in the data

* **One-to-many is exact:** 0 of 7,638,365 ground-truth S2/S3 records belong to more than
  one S1 entity. The LightGBM submission nonetheless had 35,809 contested records — at
  least 44,663 provably wrong pairs.
* **France concentrates conflicts:** 52% of contested records; 5.15% of French predicted
  pairs in a conflict vs 0.51% (US) and 1.03% (India).
* No leakage structure: row order, id values and within-file clustering are
  indistinguishable from random.

## Model selection by 5-fold cross-validation

Evaluation set: every train entity in 8 whole states (133,780 entities, 31.6M candidate
rows), chosen so both sides of a conflict are scored — a random sample of fraction f sees
only ~f² of conflicts. A closure check on real test conflicts showed same-state holds for
~65% of US/India conflicts, so conflict effects are understated. Nested threshold tuning.
Sparse-only 60-feature model. Oracle F0.5 on this set 0.9816, blocking recall 0.9560.

| model + variant | F0.5 | vs plain LightGBM | folds won |
|---|---|---|---|
| **LightGBM + keep-best** | **0.94841** | +0.00219 ± 0.00028 | 5/5 |
| LightGBM+XGBoost + keep-best | 0.94841 | +0.00218 ± 0.00052 | 5/5 |
| 3-model blend + keep-best | 0.94827 | +0.00205 | 5/5 |
| LightGBM + margin | 0.94794 | +0.00172 | 5/5 |
| LightGBM + drop-all | 0.94765 | +0.00142 | 5/5 |
| LightGBM | 0.94623 | — | — |
| XGBoost | 0.94547 | −0.00076 | 0/5 |
| CatBoost | 0.94318 | −0.00305 | 0/5 |

Every fold chose τ=0.88, δ=0.08. XGBoost/CatBoost trained on the GPU (4.5 / 8.6 min for
5 folds vs 19.5 min for LightGBM on CPU). CatBoost hit its 1,500-iteration cap in every
fold, so it may be slightly under-trained.

## Bi-encoder

What sparse blocking missed on validation (7,654 pairs, recall 0.9448): 28.9% cross-script
names, 64.9% same-script names too common to rank, 6.1% alias names (0.34% of all pairs —
unreachable). Probe of four configurations against 800k same-country distractors
(recovery of the missed pairs at K=50): e5-small name-only 25.3%, **e5-small name+address
73.6%**, MiniLM name-only 18.9%, MiniLM name+address 33.8%.

Correction to an earlier claim: I projected that a bi-encoder could reach "at most 0.961"
recall. That held for name-only embeddings (measured 0.959); name+address reached 0.986.

Real recall against the full pools (sparse ∪ dense): K=10 0.9821, **K=25 0.9855 (+38
candidates/entity)**, K=50 0.9875, K=100 0.9896. Dense alone at K=10 (0.9506) already
beats all of sparse blocking (0.9448).

| | sparse only | **sparse ∪ dense** |
|---|---|---|
| validation macro F0.5 | 0.94286 | **0.96090** |
| micro precision / recall | 0.9855 / 0.8889 | 0.9843 / 0.9277 |
| US / India | 0.9573 / 0.9212 | 0.9707 / 0.9462 |
| oracle | 0.9761 | 0.9952 |

Top features of the retrained model: `dcos_rank`, `dense_cos`, `dense_rank`. Rule:
τ=0.90, δ=0.08. On test, the bi-encoder raised India from 3.05 to 3.25 matches/entity
and **doubled France's contested records (18,625 -> 39,328)**; keep-best resolved all
56,531 contested records (89,501 claims removed).

## Engineering notes

* Float32 running sum in the retrieval budget made a query's token cutoff depend on the
  other queries in its chunk (1 of 55,708 entities differed on re-run). Now float64.
  Verified: two processes computing the same 999 test entities produced identical output.
* Selecting random rows from a memory-mapped 7.6 GB feature file pulled ~6 GB into the
  process working set and left 0.8 GB free; replaced with sequential reads (4.4 GB).
* The CUDA PyTorch build and Anaconda's Intel OpenMP runtime cannot share a process; the
  embedding steps run in an isolated venv. All caches and temp files are redirected off
  the system drive (`embed_env.sh`).
* `decide.py` used to overwrite the frozen `best_rule.npy` on every run; it now writes
  only with `--write-rule`.
