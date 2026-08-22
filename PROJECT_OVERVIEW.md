# JusDef: project overview

**Author**: Awais Abdul Khaliq (PhD candidate, Università degli Studi di Milano, Dipartimento di Informatica "Giovanni degli Antoni")
**Supervisors**: Prof. Stefano Montanelli (tutor), Prof. Alfio Ferrara (co-tutor)
**Repository**: github.com/oziofficial5/jusdefv2
**Last updated**: 22 August 2026

This document is the entry point for a reader who wants the project's story in ten minutes. It is kept aligned with the thesis; where the thesis reports a result with a caveat, that caveat is reproduced here.

---

## TL;DR

We built a defeasibility-aware graph neural network for legal text classification, with explicit operators for affirmation, negation, exception and override. We expected it to beat a standard relational baseline on legal text where defeasibility matters. It did not.

Rather than tune until the number improved, we built the measurement apparatus needed to find out why, and then to ask a harder question: when a structural prior does appear to help, is the structure doing the work, or is it just extra parameters?

The answer for legal defeasibility is mostly the latter. A three-arm permutation control attributes roughly **three-quarters** of our best result to added model capacity rather than to operator meaning. The remaining effect is real but small, and does not clear conventional significance. Fine-tuning the encoder removes the benefit entirely.

**The durable contribution is the permutation control, not the architecture.** It is not specific to law or to JusDef, it applies to any claimed structural prior over typed-edge graphs, and it has been validated across eighteen node- and entity-classification benchmarks where it correctly separates signals that are genuinely used from those that are not.

---

## The claim, stated precisely

Defeasibility-aware aggregation of the JusDef-SP class functions as a **substitute for encoder adaptation**. It is useful in frozen-backbone, low-resource and efficiency-constrained settings, and is subsumed once the encoder can be fine-tuned. On LEDGAR contract clauses a narrow operating regime is located at 10–20% non-AFF operator density, but the density measure is arithmetically confounded with paragraph length, so **the regime is a located boundary rather than an explained one**, and which of the two accounts holds is left open.

The methodology required to establish that, the permutation control and the stratified protocols, together with the density measurement, the annotated corpus and the detector, are contributions independent of any verdict on the architecture.

---

## Ten contributions

| # | Contribution | Type | Thesis chapter |
|---|---|---|---|
| **C1** | **Diagnostic and permutation-control methodology, separating genuine structural mechanism from generic added capacity. Validated as a general diagnostic across eighteen benchmarks. Primary contribution.** | Methodological | 6, 7 |
| C2 | `Y_exc` and density-stratified evaluation protocols | Methodological | 3, 7 |
| C3 | Cross-corpus measurement of *explicitly marked* operator density | Methodological / empirical | 3 |
| C4 | 3,000-sentence operator-annotated corpus, peer-validated at κ = 0.77 | Resource | 3 |
| C5 | Neural operator detector, deployed cross-corpus without retraining | Resource | 3 |
| C6 | JusDef-SP, a signal-preserving architecture grounded in three named prior failure modes | Architectural | 6 |
| C7 | Reduction theorems relating JusDef-SP to R-GCN and hard DMP | Theoretical | 6 |
| C8 | Empirical identification of an operating regime, reported as a located boundary | Empirical | 7 |
| C9 | JusDef-RR, density-routed architecture with two-stage training | Empirical | 8 |
| C10 | Defeasibility-aware aggregation as a substitute for encoder adaptation | Empirical | 7, 8 |

Any two of the three pillars (a method, resources, a characterisation) are intended to stand alone.

---

## What the project contains

### 1. An operator-annotated corpus (C4)

3,000 EUR-Lex sentences labelled with one of four operators (AFF, NEG, EXC, OVR) by an AI-assisted bulk-labelling pipeline, validated in two stages.

**Provenance.** Candidate labels were produced with Claude Opus 5 (Anthropic) in June 2026 under a fixed annotation guideline, then validated by human annotation. This is disclosed in the thesis (Chapter 3) and recorded here because the corpus is released for reuse.

**Validation.** A 100-sentence self-review at Cohen's κ = 0.79, and a 300-sentence blind annotation by an independent peer annotator at **κ = 0.77** (substantial agreement under Landis–Koch). The self-review measures the labelling function's consistency with itself and is therefore an **upper bound**, not independent evidence. The peer figure is the one that carries weight and the one that bounds the detector.

**Known residual ambiguity.** The majority of peer-annotator disagreements concentrate in a single linguistic pattern, "Unless X, Y" constructions, documented as a limitation of the outermost-wins decision rule.

Released under `data/annotations/`, with the annotation guidelines and both IAA files.

### 2. A neural operator detector (C5)

A fine-tuned LegalBERT classifier, applied to three corpora with no per-corpus retraining.

Validation macro-F1 is 0.977, but the validation set also served as the early-stopping and model-selection set, so **this is a selected maximum rather than a clean held-out estimate**. What the detector reproduces is the AI labelling function; that function agrees with expert annotation at κ = 0.77, so end-to-end reliability against expert judgement is bounded by 0.77, not 0.977.

### 3. Cross-corpus operator density (C3)

The first systematic, validated, cross-corpus measurement of *explicitly marked* defeasible-operator density on legal text.

| Corpus | Non-AFF density |
|---|---|
| EUR-Lex, EU regulations (LexGLUE) | **0.71%** |
| ECtHR case-law facts (AUEB-NLP) | **0.52%** |
| LEDGAR, US contract clauses (LexGLUE) | **20.03%** |

The qualifier "explicitly marked" is load-bearing. The measurement counts sentences whose surface form carries a lexical defeat marker. It says nothing about defeasibility operating structurally through *lex specialis* or *lex posterior* without a marker.

The gap between regulations and contract clauses is **28×, more than an order of magnitude**, driven primarily by EXC markers. ECtHR's 0.52% is the corpus figure; the rationale subset is 0.58%.

**Unit caveat.** EUR-Lex density is measured per mention-edge; LEDGAR and ECtHR are measured per sentence. The measurements are therefore not perfectly like-for-like, and the thesis states this in Chapter 3.

This result explains the project's first-half failure: the architecture was being evaluated on corpora that are over 99% non-defeasible, where by our own reduction theorem it degenerates to an R-GCN carrying extra untrained parameters.

### 4. JusDef-SP, a signal-preserving architecture (C6, C7)

The corrected v2 architecture underperforms an R-GCN baseline by **nine macro-F1 points** on EUR-Lex. Five candidate explanations were formulated and tested by single-variable intervention:

| Hypothesis | Intervention | Verdict |
|---|---|---|
| H1 Threshold protocol bug | validation-frozen threshold | gap persists, falsified |
| H2 Capacity bottleneck | R-GCN at double width | wider is worse, falsified |
| H3 Stage-2 curriculum failure | extended patience, log inspection | curriculum fires, F1 flat, falsified |
| H4 Operator-density bottleneck | neural detector swap | F1 unchanged, **set aside, not refuted** |
| H5 Auxiliary losses harmful | disable both | F1 unchanged, falsified |

H4 was set aside rather than falsified because every density obtainable on EUR-Lex lies below the range in which it predicts an effect. The variable returns on LEDGAR as the axis along which the operating regime is found.

JusDef-SP (`src/model/v3_layer.py`) responds to the resulting diagnosis with three modifications, each addressing one named failure mode from the wider GNN literature:

- a single shared message transform, for asymmetric training of operator-indexed `W_ω` at low density;
- operator-conditioned soft attention, for straight-through-estimator bias (Liu et al., ICML 2022);
- per-operator signed coefficients with an L2 drift regulariser, for sign cancellation (Zhu et al., 2024).

Three reduction results establish that with coefficients `c = (1,1,1,1)` and uniform attention JusDef-SP reduces **exactly** to R-GCN, that it recovers hard DMP as attention sharpens to one-hot, and that its hypothesis class strictly contains hard DMP's. A baseline-matching configuration therefore sits inside the hypothesis class and training does not reach it, which locates the problem in optimisation rather than expressivity.

**JusDef-SP did not close the gap on EUR-Lex.**

### 5. The operating regime (C8)

On LEDGAR, JusDef-SP underperforms a mean-aggregation baseline on the full test set (0.6881 against 0.7085 macro-F1, three seeds). Density-stratified evaluation locates a narrow band where it wins:

| Density bin | N | Mean baseline | JusDef-SP | Δ |
|---|---|---|---|---|
| All paragraphs | 10,000 | 0.7085 | 0.6881 | −0.0204 |
| **10–20% non-AFF** | **156** | 0.6193 | 0.6722 | **+0.0624** |
| ≥20% non-AFF | 3,353 | 0.7066 | 0.6871 | −0.0194 |

The regime delta is **+0.0624 on five seeds** (sample SD 0.0280, standard error 0.0125, all five seeds positive) and **+0.0753 on ten seeds**. Quote the seed panel whenever quoting the number. No architectural benefit is detectable at any other density, or on either low-density corpus.

**This is a located boundary, not an explanation of one.** See the confound below.

### 6. The permutation control (C1, C10)

The core methodological contribution. A gain over a baseline can come from the structure being genuinely used, or from the extra parameters that came with it. Three arms separate them:

| Arm | Operator information supplied | Regime gain, ten seeds |
|---|---|---|
| **True** | real operator labels | **+0.0753** |
| **Shuffled** | labels randomly permuted | **+0.0376** |
| **Zeroed** | all set to AFF, no information at all | **+0.0548** |

With **no operator information whatsoever**, the architecture recovers +0.0548 of the +0.0753 gain. **Roughly three-quarters of the headline result is added capacity, not defeasibility.**

Decomposing the remainder:

- real operators beat **randomised** ones by **+0.038**, `p ≈ 0.02` (significant);
- real operators beat **no** operators by only **+0.021**, `p ≈ 0.08` (marginally non-significant).

Operator identity carries some signal, but the total contribution of genuine operator information does not clear conventional significance.

**The control generalises.** Applied unchanged across eighteen node- and entity-classification benchmarks, it separates signals genuinely used (homophilous citation-graph edges, AIFB and BGS relation types) from those that are not (heterophilous edges, MUTAG relation types), placing legal operators at the low end of that spectrum. It is a general diagnostic for any typed-edge graph model, and this generality is part of the contribution.

Implementation: `scripts/perm_control_gnn.py`, `scripts/perm_control_rgcn.py`, `scripts/analyse_thirdarm.py`.

### 7. The density / paragraph-length confound

Per-paragraph density is `d_p = k / n`, with `k` non-AFF sentences and `n` sentences. Both are integers and `n` is small for contract clauses, so each density bin admits only certain `(n, k)` pairs:

| Bin | Constraint | Minimum n |
|---|---|---|
| 0–5% | n > 20k | 21 |
| 5–10% | 10k < n ≤ 20k | 11 |
| **10–20%** | **5k < n ≤ 10k** | **6** |
| 20–30% | 10k/3 < n ≤ 5k | 4 |
| 30–50% | 2k < n ≤ 10k/3 | 3 |
| ≥50% | n ≤ 2k | 1 |

Entering the 10–20% band requires **at least six sentences**. LEDGAR's median paragraph is **two**, and 44.2% are single-sentence, so only **523 of 10,000** paragraphs are eligible on length alone. The 156 that populate the band average **7.74 sentences against 1.96** for the rest.

Three consequences, all stated in the thesis:

1. The thin middle of the density distribution is an artefact of the measure, not a fact about contract drafting.
2. The operating regime is a long-paragraph subset by construction.
3. **The regime claim is not identified.** A signed soft-attention pooler beating a mean pooler on paragraphs long enough for pooling to matter is an equally complete account of the observations, and is consistent with the zeroed arm recovering +0.0548.

The resolving experiment, sentence-count stratification at fixed density over ten seeds, is named in the thesis conclusions and has not been run.

Implementation: `scripts/analyse_ledgar_length_confound.py`.

### 8. JusDef-RR, the routed variant (C9)

`src/model/v4_router.py`. Routes each paragraph to either the JusDef-SP update or the mean baseline according to measured density, trained in two stages to resolve the distribution shift of end-to-end routed training.

JusDef-RR improves on JusDef-SP both in aggregate and on the regime, and **draws level** with the mean baseline in aggregate. Two LEDGAR mean baselines are in circulation, 0.7085 (three seeds) and 0.6955 (five seeds); JusDef-RR at 0.7023 sits above one and below the other, so the aggregate comparison is a **statistical tie and is never reported as a win**.

The negative result travels with it: single-stage end-to-end training of the same architecture fails on the regime. Routing mechanism and training protocol must be designed together.

The routing criterion inherits the confound. It selects on density, density is confounded with length, so it also selects on length. What survives is that selective application removes the aggregate penalty, making C9 a contribution about conditional architecture rather than about defeat semantics.

### 9. Scope-fixing experiments (C10)

**Fine-tuning removes the benefit.** Once LegalBERT is allowed to adapt, there is no aggregate gain and only a small non-significant regime gain.

**A bag-of-words baseline matches a fine-tuned transformer.** TF-IDF with a linear SVM reaches **0.828**; a properly tuned LegalBERT reaches **0.829** (ten seeds). This bounds how much any architectural comparison on this task can establish.

---

## What this project does not claim

- **No claim of state-of-the-art accuracy.** Graph-only architectures, R-GCN included, underperform transformer baselines by roughly 25–30 macro-F1 points on EUR-Lex. That is a property of the architectural class.
- **No claim that defeasibility-aware GNNs help in general.** They do not, on any of the three corpora tested, outside one narrow band on one corpus.
- **No claim that the operating regime is explained.** It is located. Whether it reflects defeasibility or paragraph length is unresolved, and the thesis says so.
- **No claim that the regime gain is mostly mechanism.** Roughly three-quarters is added capacity by our own control.
- **No claim of component-level architectural novelty.** Each JusDef-SP component has prior art in adjacent communities. The contribution is the synthesis targeted at three diagnosed failure modes.
- **No claim about unmarked defeasibility.** All density figures concern *explicitly marked* operators only.

---

## Practical takeaways

1. **Measure operator density before committing to an architecture.** The released detector runs off-the-shelf on any English legal corpus. Below roughly 10% density, defeasibility-aware GNNs of this class are unlikely to help, and a simpler baseline is more parameter-efficient.

2. **Run the permutation control on any structural prior before believing it.** Shuffling and then zeroing the structure costs two extra training runs and tells you whether you have a mechanism or just parameters. On this project it changed the conclusion.

3. **Report stratified metrics alongside aggregates**, and check whether your stratification variable is confounded with something simpler. Ours was.

4. **For absolute accuracy on legal multi-label tasks, fine-tune a transformer.** This project's contribution is to architectural characterisation, not to the accuracy leaderboard.

---

## Code organisation

```
jusdefv2/
├── PROJECT_OVERVIEW.md          ← this document
├── AMPERE_RUNBOOK.md            ← cluster reproducibility runbook
├── notes/
│   ├── methodology.md           ← diagnostic chronology
│   ├── baselines.md
│   └── thesis_outline.md
├── data/
│   ├── annotations/             ← released corpus, guidelines, IAA files
│   │   ├── operator_labels_3000.jsonl
│   │   ├── operator_guidelines.md
│   │   └── iaa/                 ← self-review 100 + peer-annotated 300
│   ├── eurovoc/                 ← raw EuroVoc RDF (gitignored, 58 MB, re-downloadable)
│   └── processed/               ← graphs and embeddings (large artefacts gitignored)
├── src/
│   ├── model/
│   │   ├── jusdef.py            ← assembled v2 model
│   │   ├── dmp_layer.py         ← hard-DMP layer (v1, v2)
│   │   ├── v3_layer.py          ← JusDef-SP, signal-preserving
│   │   ├── v4_router.py         ← JusDef-RR, density-routed
│   │   ├── jusdef_ledgar.py     ← LEDGAR paragraph classifier
│   │   ├── authority_scorer.py
│   │   └── baselines.py         ← R-GCN baseline
│   ├── kg/kg_builder.py
│   ├── preprocess/
│   ├── train/
│   └── eval/                    ← includes bootstrap.py (one-sided, see below)
├── scripts/                     ← pipeline entry points and analyses
└── tests/
    ├── test_architecture_fixes.py
    └── test_ledgar_pipeline.py
```

**Naming.** `v3` and `v4` are code identifiers only. In the thesis and in writing they are **JusDef-SP** (signal-preserving) and **JusDef-RR** (regime-routed).

**Branches.** `thesis-main` is the default branch and now contains the complete LEDGAR line of work. `jusdef-ledgar` is at the same commit. `jusdef-v3` is a historical branch retained for provenance.

**Bootstrap sidedness.** `src/eval/bootstrap.py` computes a one-sided p-value in the observed direction. This is documented in the thesis Chapter 3, including that it is permissive outside the pre-registered regime hypothesis.

---

## Reproducibility

```bash
# Full EUR-Lex pipeline (5–8 days on an A100)
bash scripts/run_all_ampere.sh

# Full LEDGAR pilot (5–7 hours on an A100)
bash scripts/run_pilot_ledgar.sh

# Density-stratified analysis (no GPU, ~30 seconds)
python scripts/analyse_ledgar_density_subset.py

# The paragraph-length confound control (no GPU)
python scripts/analyse_ledgar_length_confound.py

# The three-arm permutation control
python scripts/analyse_thirdarm.py
```

See `AMPERE_RUNBOOK.md` for the detailed runbook. Per-seed JSON result files under `outputs/logs/` back the reported numbers.

---

## Publications

| # | Paper | Venue | Status |
|---|---|---|---|
| 1 | Khaliq & Montanelli, "Language Models for Legal NLP: A Literature Review" | CAiSE 2025 Workshops, Springer LNBIP 556, 326–337 | Published |
| 2 | Khaliq, Riva & Montanelli, "Evaluating Knowledge-Based Approaches for Legal Text Analysis: A Benchmark Study" | Computer Law & Security Review 61:106279 | Published 2026 |
| 3 | Khaliq, Montanelli, Dalianis & Naqvi, "JusDef: Defeasible Message Passing for Exception-Aware Legal Document Classification" | ANNPR 2026 | Accepted |
| 4 | "When Does Defeasibility-Aware Neural Aggregation Help Legal Text Classification? Separating Mechanism from Added Capacity with a Permutation Control" | Artificial Intelligence and Law | Under review |

**On paper 3.** The ANNPR workshop paper reports a positive result for the hard-defeat architecture over a frozen encoder. That evaluation tuned its decision threshold on test labels. Corrected, the result does not survive, and the thesis reports the correction in full.

**On paper 4.** Submitted before the paragraph-length confound was identified, so it states the operating regime without the identification caveat recorded above. **The thesis is the current statement**, and the caveat will be added at revision.

---

## Status, 22 August 2026

- All ten contributions substantiated, with the caveats recorded above
- Multi-seed evaluations complete: three, five and ten-seed panels on LEDGAR; three seeds on EUR-Lex
- Cross-corpus density measurement complete across three corpora
- Permutation control complete on JusDef and on eighteen external benchmarks
- Paragraph-length confound identified and propagated through the thesis
- Thesis complete, approximately 66,400 words, 10 chapters and 3 appendices
- Outstanding: sentence-count stratification at fixed density, the experiment that would resolve the confound

---

*Entry point for the repository. The annotation provenance is in `data/annotations/`, the diagnostic chronology in `notes/methodology.md`, and the per-seed JSON logs under `outputs/logs/` reproduce the reported numbers.*
