# crossscript-entity-resolution

Entity resolution across **10.3M noisy business records** in three scripts and three
countries, with no shared identifiers between sources.

Given a deduplicated reference source (Source 1) and two noisy sources (Source 2 and
Source 3), find every record across S2/S3 that refers to the same real-world business
as each S1 entity. An entity may have zero, one, or many matches.

| | |
|---|---|
| **Validation macro F₀.₅** | **0.94286** |
| Blocking recall | 0.9448 |
| Reduction ratio | 0.99997715 |
| Micro precision / recall | 0.9855 / 0.8889 |
| Scale | 2.2M queries × 10.3M candidates (train) · 1.7M × 10.0M (test) |

Measured on a frozen, stratified held-out split of 39,999 Source-1 entities, searched
against the **full** candidate pools — not a subsample.

---

## The interesting part of this problem

**Names are transliterated into entirely different scripts.** The same business appears
as `Shyam Consulting Pvt Ltd`, `श्याम कंसल्टिंग प्रा. लि.`, and `Shyam Pvt Ltd Center`.
Raw string similarity scores these at ~0 across scripts. The data contains Devanagari,
Tamil, Telugu, Kannada, Gujarati and Bengali.

The fix is cheap and does most of the work: `unidecode`, then **collapse runs of
repeated characters**. Indic transliteration systematically doubles consonants and
lengthens vowels, so `limittedd → limited` and `praaivett → praivet`. This lifts typical
cross-script pair similarity from ~0 to 70–95.

**Addresses saturate.** In dense commercial buildings dozens of distinct businesses
normalise to an identical address string, so address agreement there is nearly
worthless evidence — this turned out to cause 52% of all false merges. About 3% of
records have no address at all.

**France appears only at test time**, with no training representation, so nothing in
the pipeline may be country-coupled.

---

## Approach

**1 · Normalise** — `unidecode` → lowercase → strip punctuation → collapse repeated
characters. Legal forms (`Pvt`/`Private`/`प्रा.`, `L.L.C.`/`LLC`) and honorifics are
folded to canonical tokens and separated from the core name.

**2 · Block** — every record becomes a bag of country-prefixed tokens: core name
tokens, name char-4-grams, address tokens, address digit runs. Retrieval is binary-tf
cosine (Σ idf² ÷ field norm), walked rarest-first under a posting budget.

The key design choice: **name and address are retrieved separately and unioned.** A
true pair often agrees on exactly one field — the address is empty so only the name can
match, or the name is in another script so only the address can. Summing both into one
score buries those pairs.

| blocking iteration | recall |
|---|---|
| composite keys | 0.829 |
| IDF-weighted tokens, one combined field | 0.848 |
| per-field cosine, unioned | 0.894 |
| widened (K=60, budget 8000) | **0.945** |

**3 · Match** — LightGBM over **60 pair features**, cross-validated with folds grouped
by S1 entity, then isotonically calibrated.

The strongest features are **relative, not absolute**: a candidate's rank within its
entity's candidate set, its ratio to the best score, its gap to the runner-up. Five of
the top six features by gain are relative. The question that matters is not "is this
pair similar" but "is this the best explanation for this entity" — which is worth ~0.2
F₀.₅ on its own.

**4 · Decide** — accept when `score ≥ 0.88` **and** `score ≥ best − 0.08`, at most 12
per entity. F₀.₅ weights precision twice recall and is macro-averaged per entity, so a
false merge on a singleton costs a full 1.0.

---

## Results

| stage | blocking recall | macro F₀.₅ |
|---|---|---|
| hand-weighted similarity blend | 0.8944 | 0.706 |
| + LightGBM matcher (52 features) | 0.8944 | 0.912 |
| + wide blocking | 0.9448 | 0.930 |
| + retrained on wide candidate distribution | 0.9448 | 0.936 |
| + corpus-statistic & conflict features (60) | 0.9448 | **0.943** |

A perfect matcher over the current candidate set would score **0.976**, so the matcher
realises 96.6% of what blocking makes available.

### What didn't work, and is worth knowing

* **An elaborate per-entity decision rule was worth +0.0004.** A full sweep over global
  threshold × relative margin × a higher bar for 2nd+ matches × an explicit singleton
  gate × cap barely beat a plain global threshold. Both the second-match threshold and
  the singleton gate optimised to *zero*. With a sufficiently separable score
  distribution, the shape of the decision stops carrying information.
* **Singletons are a small lever.** They are 5.59% of entities and 90.2% are already
  correct; perfect singleton handling would add +0.0055.
* **Chains, franchises and holding-company overlap barely appear** among false merges,
  despite being the textbook failure modes. The real cause is address saturation.

### Top false-merge patterns

| # | pattern | share |
|---|---|---|
| 1 | sibling of a true match (same name, same street, different number) | 44.7% |
| 2 | generic name — address carries it, names diverge entirely | 24.8% |
| 3 | empty address, name-only evidence | 14.9% |
| 4 | same address, genuinely different business | 6.9% |

Patterns 1 and 4 are the same underlying problem. Reading these directly produced the
address-saturation, name-rarity and street-number-conflict features, which moved
F₀.₅ from 0.936 to 0.943.

---

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONHASHSEED=0          # blocking tokens are hashed with hash(); the
export ER_ROOT=/path/to/root     # index and queries must agree on the seed

python src/normalize.py train test     # normalise all 7 source files  (~4 min)
python src/split.py                    # frozen split, SEED=20260925
python src/block2.py train && python src/block3.py train
python src/build_training2.py --out work/fitw --kn 60 --ka 60 --bn 8000 --ba 8000
python src/train_matcher.py --data work/fitw
python src/score_split.py --truth work/val_truth.tsv --out work/val_scored.npz
python src/decide.py --scored work/val_scored.npz
```

Full end-to-end instructions, including sharded test inference, are in
[PIPELINE_README.md](PIPELINE_README.md).

`ER_ROOT` must point at the directory containing `dataset/`. Peak RSS is ~7 GB per
process.

## Layout

| path | role |
|---|---|
| `src/common.py` | IO, normaliser, legal-form / address / state dictionaries |
| `src/block2.py`, `src/block3.py` | postings index; per-field cosine retrieval |
| `src/pairs.py`, `src/features2.py`, `src/engine.py` | candidate generation, the 60 features |
| `src/train_matcher.py`, `src/decide.py` | grouped-CV LightGBM; decision-rule sweep |
| `src/run_pipeline2.py` | sharded inference → submission TSVs |
| `src/error_analysis.py` | false-merge pattern classification |
| `src/evaluate.py` | macro F₀.₅ scorer with per-stratum breakdown |
| [RESULTS.md](RESULTS.md) | full experiment log, including what failed |
| [Documentation_template.md](Documentation_template.md) | methodology write-up |

`src/keys.py`, `src/blocking.py`, `src/features.py`, `src/run_pipeline.py` are the
superseded first-iteration blocking and matching code, kept because the write-up
refers to them.

## Data and provenance

The `dataset/` directory is **not committed**. `utils/validate_submission.py` is
organiser-provided and is likewise excluded; both come from the challenge's
`student_resource` package.

**No external data of any kind is used** — no entity-resolution APIs, business
registries, geocoding services, or internet augmentation. Every statistic the pipeline
relies on (token document frequencies, address saturation counts, IDF weights) is
computed from the provided TSVs alone.

**No pretrained model is used.** The matcher is LightGBM (MIT) trained from scratch.
`Unidecode` is GPL-2.0-or-later and is used only for character transliteration during
preprocessing; `anyascii` (ISC) is a drop-in permissive alternative if that matters for
your use, though it changes transliteration output and would require rebuilding the
indexes and retraining.
