# JusDef — Project Overview

**Author**: Awais Abdul Khaliq (PhD, Università degli Studi di Milano)
**Supervisors**: Prof. Stefano Montanelli, Prof. Alfio Ferrara
**Last updated**: 17 June 2026 (post-thesis-first-draft)
**Repository**: github.com/oziofficial5/jusdefv2

**Branches**:
- `thesis-main`: stable v2 + diagnostic + corpus + neural-detector pipeline (EUR-Lex)
- `jusdef-v3`: v3 signal-preserving architecture (EUR-Lex three-seed evaluation)
- `jusdef-ledgar`: v3 applied to LEDGAR contract clauses (where the architecture wins on a narrow operating regime)

This document is the entry point for a quick reader who wants the project's core story in 10 minutes. The full thesis (10 chapters + appendices, ~200 pages of LaTeX) is at `papers/chapter*.tex`. The supervisor outline (5 pages) is at `papers/thesis_outline_for_supervisor.tex`.

---

## TL;DR — what we did and what we found

We built a graph neural network framework (JusDef) for multi-label legal document classification with explicit operators for affirmation, negation, exception, and override. We expected the architecture to outperform a standard relational baseline on legal text where defeasibility matters. We were surprised when it did not. We then constructed the methodological infrastructure to understand why: a 3000-sentence expert-validated operator-annotated corpus, a neural operator detector, a cross-corpus density measurement on three legal datasets, and a systematic diagnostic methodology that ruled out five hypothesised causes of the underperformance. The investigation produced eight contributions and a single calibrated empirical finding: defeasibility-aware GNN architectures of the v3 class provide measurable benefit on a narrow operating regime (10–20% non-AFF operator density on LEDGAR contract clauses), with no measurable benefit outside this window on any of the three corpora tested.

---

## The thesis claim in one sentence

Defeasibility-aware graph neural network architectures have an empirically-identifiable operating regime characterised by sentence-level non-AFF operator density between 10% and 20%; below and above this regime no architectural benefit is detectable; the diagnostic methodology, the cross-corpus density measurement, and the expert-validated operator-annotated corpus required to identify this regime constitute methodological and resource contributions to legal NLP independent of the specific JusDef framework.

---

## Eight contributions at a glance

| # | Contribution | Type | Chapter |
|---|---|---|---|
| C1 | Diagnostic methodology (hypothesis-falsification chain over five candidates) | Methodological | 6 |
| C2 | Y_exc and density-stratified evaluation protocols | Methodological | 3, 8 |
| C3 | Cross-corpus density measurement (EUR-Lex 0.71%, ECtHR 0.58%, LEDGAR 20.03%) | Methodological / empirical | 5 |
| C4 | 3000-sentence operator-annotated corpus with κ = 0.77 independent-annotator IAA | Resource | 5 |
| C5 | Neural operator detector (validation macro-F1 = 0.977) | Resource | 5 |
| C6 | v3 architecture grounded in three named prior failure modes | Architectural | 7 |
| C7 | Three reduction theorems (Propositions 2, 3, 4) | Theoretical | 7 |
| C8 | Empirically-identified operating regime on LEDGAR (10–20% bin, +0.053 F1 multi-seed) | Empirical | 8 |

---

## What the project actually contains

### 1. An expert-validated operator-annotated corpus

3000 EUR-Lex sentences labelled with one of four operators (AFF, NEG, EXC, OVR) by an AI-assisted bulk-labelling pipeline, validated by two stages of inter-annotator agreement: a 100-sentence self-review (Cohen's κ = 0.79) and a 300-sentence independent-annotator blind annotation (Cohen's κ = 0.77, substantial agreement under Landis–Koch). The disagreement structure concentrates 58% of annotator disagreements in a single linguistic pattern ("Unless X, Y" constructions), which is documented as a known limitation of the outermost-wins decision rule. The corpus, the 252-line annotation guidelines, and the IAA validation files are released as `data/annotations/` in the repository.

### 2. A trained neural operator detector

A fine-tuned LegalBERT classifier with held-out validation macro-F1 of 0.977. The detector reproduces the AI labelling function reliably; its end-to-end reliability against expert annotation is bounded by the independent-annotator κ = 0.77. The detector is applied to three legal corpora to produce per-sentence operator labels at scale, without per-corpus retraining. The checkpoint is released as `outputs/checkpoints/operator_detector_neural.pt`.

### 3. The first systematic cross-corpus density measurement on legal text

Using the same detector applied to three datasets:

| Corpus | Non-AFF density |
|---|---|
| EUR-Lex (EU regulations, LexGLUE) | **0.71%** |
| ECtHR case-law facts (AUEB-NLP) | **0.58%** |
| LEDGAR (US contract clauses, LexGLUE) | **20.03%** |

The density spans two orders of magnitude. EU regulations and ECtHR facts both have sub-1% explicit defeasible-operator density, falsifying the hypothesis that case law is intrinsically denser in defeasibility markers than regulations. US contract clauses have approximately 28× higher density than regulations, driven primarily by EXC markers (14.35%, "subject to", "except as provided", "unless").

### 4. A signal-preserving v3 architecture

The v3 architecture (`src/model/v3_layer.py`) replaces the hard-defeat DMP layer of v1/v2 with three modifications, each grounded in a specific named prior failure mode from the broader graph-neural-network literature:

- Single shared message transform (addresses the asymmetric-training failure mode whereby operator-indexed W matrices are undertrained at low non-AFF density);
- Operator-conditioned soft attention (addresses STE gradient bias documented by Liu et al. "Gapped STE" ICML 2022);
- Per-operator signed coefficients with L2 drift regulariser (addresses signed-GNN sign cancellation documented by Zhu et al. "Sign is Not a Remedy" 2024).

Three theoretical propositions establish that v3 reduces to R-GCN under a specific parameter setting (safety guarantee), recovers hard-DMP as a sharpening limit, and strictly generalises both R-GCN and hard-DMP hypothesis classes.

### 5. A systematic diagnostic methodology

The corrected v2 architecture underperforms R-GCN by 9 macro-F1 points on EUR-Lex across three seeds. We formulated five candidate explanations and tested each through a controlled intervention. All five were falsified:

| Hypothesis | Intervention | Outcome |
|---|---|---|
| H1: Threshold protocol bug | Correct val-frozen threshold | Gap persists. Falsified. |
| H2: Capacity bottleneck | R-GCN at h=768 (2× capacity) | h=768 worse than h=512. Falsified. |
| H3: Stage-2 curriculum failure | Extended patience, log inspection | Curriculum activates; F1 flat. Falsified. |
| H4: Operator-density bottleneck | Neural detector → 8.5× density change | F1 unchanged. Falsified. |
| H5: Auxiliary curriculum harmful | Disable L_onto and L_defeat | F1 unchanged. Falsified. |

The falsification chain motivated a mechanistic synthesis grounded in three named prior failure modes from the broader literature (STE bias, sign cancellation, encoder negation blindness), which in turn motivated the v3 architecture.

### 6. The operating-regime identification (the principal positive finding)

On LEDGAR contract clauses, the v3 architecture underperforms a mean-aggregation baseline on the full test set (0.6881 vs. 0.7085 macro-F1, 3-seed mean). But density-stratified evaluation tells a different story:

| Density bin | N | Mean baseline | v3 | Δ |
|---|---|---|---|---|
| All paragraphs | 10,000 | 0.7085 | 0.6881 | −0.0204 |
| **10–20% non-AFF** | **156** | **0.6193** | **0.6722** | **+0.0528 (all 3 seeds positive)** |
| ≥20% non-AFF | 3,353 | 0.7066 | 0.6871 | −0.0194 |

The 10–20% bin corresponds linguistically to "main rule with one or two exceptions" contract clauses. The seed-by-seed deltas are +0.0682, +0.0199, +0.0704. The effect size is 2.3× the seed-to-seed standard deviation. This is the principal positive empirical finding of the thesis.

---

## What the project does NOT claim

- No claim of state-of-the-art macro-F1 on EUR-Lex. Graph-only architectures (including R-GCN) underperform transformer baselines by approximately 25–30 macro-F1 points on this benchmark. This is a documented property of the architectural class, not a critique of JusDef specifically.
- No claim of uniform empirical benefit of defeasibility-aware GNNs. The positive finding is restricted to a narrow operating regime (10–20% non-AFF density on LEDGAR, 1.56% of the test set). Outside this window, no architectural benefit is detectable on any of the three corpora tested.
- No claim of architectural novelty in any individual component of v3. Each component has prior art in adjacent communities (signed-GCN, soft attention, drift regularisation). The architectural contribution is the principled synthesis targeted at three specifically-diagnosed prior failure modes.

---

## Practical takeaways for legal-NLP practitioners

1. **Measure operator density before architectural commitment.** The trained neural detector can be applied off-the-shelf to any English legal corpus to estimate non-AFF density. If the corpus has density below approximately 10%, defeasibility-aware GNN architectures of the v3 class are unlikely to provide benefit, and a simpler baseline (R-GCN, mean aggregation, or a transformer classifier) is more parameter-efficient.

2. **Report stratified metrics alongside aggregate metrics.** Architectural benefits restricted to narrow operating regimes are invisible in aggregate macro-F1 but identifiable in stratified evaluation. The Y_exc protocol on EUR-Lex and the density-stratified evaluation on LEDGAR are released as reusable methodologies.

3. **For overall classification accuracy on legal multi-label tasks, use fine-tuned transformers.** This thesis's contribution is to the architectural-characterisation question, not to the absolute-accuracy comparison.

---

## Code organisation

```
jusdefv2/
├── PROJECT_OVERVIEW.md          ← this document
├── AMPERE_RUNBOOK.md            ← reproducibility runbook for the cluster
├── notes/
│   └── methodology.md           ← detailed diagnostic chronology
├── data/
│   ├── annotations/             ← released corpus + guidelines + IAA files
│   ├── eurovoc/                 ← raw EuroVoc RDF (gitignored, 58 MB)
│   ├── processed/               ← keyword-operator graphs and embeddings
│   └── processed_neural/        ← neural-operator graphs (Stage 7.5 output)
├── src/
│   ├── model/
│   │   ├── jusdef.py            ← assembled v2 model
│   │   ├── dmp_layer.py         ← hard-DMP layer (v1, v2)
│   │   ├── v3_layer.py          ← V3Layer (signal-preserving)
│   │   ├── jusdef_ledgar.py     ← LEDGAR paragraph classifier
│   │   ├── authority_scorer.py
│   │   └── baselines.py         ← R-GCN baseline
│   ├── kg/kg_builder.py
│   ├── preprocess/
│   ├── train/
│   └── eval/
├── scripts/                     ← end-to-end pipeline entry points
├── tests/test_architecture_fixes.py   ← unit tests (9 passing + 2 xfail)
└── outputs/
    ├── checkpoints/             ← trained model checkpoints
    ├── logs/                    ← per-seed JSON results
    └── sentinels/               ← stage-completion markers
```

---

## Reproducibility — one-line summary

```bash
# Full EUR-Lex pipeline (5–8 days on A100)
bash scripts/run_all_ampere.sh

# Full LEDGAR pilot (5–7 hours on A100)
bash scripts/run_pilot_ledgar.sh

# Density-stratified analysis (no GPU needed, ~30 seconds)
python scripts/analyse_ledgar_density_subset.py
```

All numbers in the thesis can be reproduced from the released artefacts. See `AMPERE_RUNBOOK.md` for the detailed runbook.

---

## Publications

| # | Paper | Venue | Status |
|---|---|---|---|
| 1 | Khaliq & Montanelli, "Language Models for Legal NLP: A Literature Review" | CAiSE 2025 Workshops, Springer LNBIP vol. 556 | Published |
| 2 | Khaliq, Riva & Montanelli, "Evaluating Knowledge-Based Approaches for Legal Text Analysis: A Benchmark Study" | Computer Law & Security Review, vol. 61 | Published 2026 |
| 3 | Khaliq, "JusDef: Defeasible Hypergraph Reasoning for Multi-Label Legal Document Classification" | ANNPR 2026 | Submitted, under review |
| 4 | (Planned) v3 architecture + operating-regime identification | TACL or AAAI 2027 | In preparation |

The eight thesis contributions (C1–C8) are thesis-only and do not appear in the three published / submitted papers above. They are the new material introduced in the thesis track.

---

## Timeline

| Date | Milestone |
|---|---|
| 2024–2025 | PhD years 1–2: survey + ASKE benchmark + JusDef v1 (ANNPR submission) |
| 2026-04 to 2026-05 | v2 architectural corrections (F1, F2, F3a); diagnostic methodology |
| 2026-05 to 2026-06 | 3000-sentence corpus; neural detector; cross-corpus density measurement |
| 2026-06 | v3 architecture; LEDGAR operating regime identification |
| **2026-06-17** | **Full thesis first draft complete (10 chapters + appendices)** |
| 2026-06-22 OR 07-02 | Internal committee presentation |
| 2026-07 to 2026-08 | Supervisor revision rounds |
| 2026-09-15 | Target submission |

---

## Status at this document (17 June 2026)

- ✅ All eight contributions empirically and methodologically substantiated
- ✅ Three-seed multi-seed evaluations completed on EUR-Lex and LEDGAR
- ✅ Cross-corpus density measurement across three corpora completed
- ✅ Full thesis first draft complete (10 chapters, 7 appendices, 85+ bibliography entries, ~200 pages)
- ✅ Supervisor outline (5 pages) prepared for review
- 🟡 Cross-reference audit and figure preparation pending
- ⏳ Supervisor revision rounds upcoming
- ⏳ Final submission September 2026

---

*This document is the project's entry point. The full thesis is in `papers/chapter*.tex`. The supervisor outline is in `papers/thesis_outline_for_supervisor.tex`. The diagnostic chronology is in `notes/methodology.md`. The annotation provenance is in `data/annotations/`. The released checkpoints and JSON result files reproduce every empirical claim.*
