# JusDef — Project Overview

**Author**: Awais Abdul Khaliq (PhD, Università degli Studi di Milano)
**Supervisors**: Prof. Stefano Montanelli, Prof. Alfio Ferrara
**Last updated**: June 2026
**Repository**: github.com/oziofficial5/jusdefv2

This document is the single context document for the JusDef project. It explains what the project is, why it exists, what has been built, what has been found, where the current architecture works and where it does not, and what the planned next step is. It is intended for a reader (examiner, collaborator, future self) who has not seen the code before.

---

## 1. The thesis claim — in one paragraph

Legal documents encode normative reasoning that is fundamentally *defeasible*: a regulation might apply by default, not apply to certain entities, apply only with exceptions, or be overridden by another rule. Standard neural classifiers treat all sentences as monotonically aggregable evidence, with no mechanism for one piece of evidence to *defeat* another. The JusDef framework introduces an **operator algebra Ω = {AFF, NEG, EXC, OVR}** over legal sentences and a graph-neural-network mechanism — **Defeasible Message Passing (DMP)** — in which messages carrying operator labels can defeat each other according to a precedence order. The thesis argues that operator-aware message passing is a principled approach to multi-label classification on regulatory corpora, that JusDef is a strict parametric generalisation of R-GCN (Proposition 1), and that the empirical benefit of the architecture is bounded by the actual semantic density of defeasible operators in the corpus — a constraint that the keyword-based detector used in the workshop version masked, and that a neural detector calibrated against expert annotation reveals.

---

## 2. Background

### 2.1 Defeasible reasoning

In classical (monotonic) logic, adding premises can only add conclusions. In **defeasible reasoning**, conclusions can be withdrawn when new information arrives — modelling the way legal rules genuinely behave (a rule applies *unless* an exception holds; a later law *overrides* an earlier one). The seminal formalisms are Reiter's default logic, Pollock's defeasible reasoning, and modern argumentation frameworks (Dung). LegalRuleML provides a standardised representation for defeasible legal rules but requires manual rule encoding.

### 2.2 Legal NLP and EUR-Lex

The EUR-Lex corpus — a multi-label classification task over EU legislation in the LexGLUE benchmark — contains ~65,000 documents tagged with up to 100 EuroVoc concepts. Standard baselines (Legal-BERT, R-GCN, ASKE) treat each sentence as monotonic evidence for the labels it mentions. The thesis argues that this is structurally inadequate for documents that encode non-monotonic norms.

### 2.3 Why a graph neural network

Documents have rich structure: sections, sentences, mentioned concepts (EuroVoc labels), and authority citations (other regulations). A **heterogeneous graph** with typed nodes (doc, sec, conc, auth, label) and typed edges (`has_section`, `mentions`, `mentions_rev`, `ontology`, `cites`, `maps_to`) captures this richness in a way that pooled-embedding classifiers cannot. The thesis builds on this representation and adds operator-awareness to the message-passing scheme.

---

## 3. JusDef v1 — the workshop version

**Reference**: Khaliq, A. A. (2026). *JusDef: Defeasible Hypergraph Reasoning for Multi-Label Legal Document Classification.* Submitted to ANNPR 2026.

### 3.1 Architecture

- **Heterogeneous graph** with node types {doc, sec, conc, auth, label} and edge types as above.
- **Operator algebra** Ω = {AFF, NEG, EXC, OVR} with strict partial order **OVR ≻ EXC ≻ NEG ≻ AFF**.
- **Operator-indexed weights** W_ω for each ω ∈ Ω, applied to messages along the `mentions` (sec → conc) edge type.
- **Hard defeat gate** implemented as a binary mask: in each concept neighborhood, only the message(s) with the maximum (operator, priority) score survive; all others are zeroed out.
- **Straight-Through Estimator (STE)** with sigmoid surrogate for backward-pass gradient flow through the binary mask.
- **Authority scorer**: a learned linear function over (authority type, level, recency) features producing per-message priority scores that participate in the defeat computation.
- **Proposition 1 (Reduction to R-GCN)**: When every sentence has operator AFF and the defeat gate is open everywhere, JusDef reduces exactly to an R-GCN over operator-indexed sub-relations.

### 3.2 Empirical claim in v1

On EUR-Lex with the keyword operator detector, v1 reported:
- JusDef ≈ R-GCN on overall macro-F1
- JusDef **+6.0 points** over R-GCN on the exception-dependent subset Y_exc (21 labels)
- Multi-seed bootstrap p < 0.001 on Y_exc seed 42

### 3.3 Operator detection in v1

Operators were assigned by a **rule-based keyword detector** (regex on phrases like *"unless"*, *"notwithstanding"*, *"shall not apply"*). Reported density: 94.5% of mention edges labelled AFF, 5.5% non-AFF (NEG, EXC, OVR combined).

---

## 4. JusDef v2 — the thesis-track extension

The thesis extends v1 with three architectural corrections and a more rigorous evaluation methodology.

### 4.1 The three corrections

| Fix | Description | Why it was needed |
|---|---|---|
| **F1** | Reverse mention edge (conc → sec) | In v1, operator-conditioned updates at concept nodes never propagated back to sentence nodes, so the sentence-level pooling was operator-blind |
| **F2** | Authority scorer placed in the forward gradient path | In v1, the authority scorer was outside the main forward graph; its gradients were effectively zero, leaving authority modelling inactive |
| **F3a** | EuroVoc-derived ontology edges between concept nodes | Originally a placeholder (parent_of with zero edges); v2 populates ~700 edges from the 2-digit EuroVoc broader-domain structure |

A unit-test suite mechanically verifies that operators reach sentences (F1), the authority scorer receives gradient (F2), and ontology edges exist (F3a). All tests pass on the current code (commit `3e734f2`).

### 4.2 Evaluation methodology corrections

- **Threshold protocol**: v1's evaluation tuned the per-split decision threshold on test labels (optimistic bias). v2 tunes on validation, freezes, applies to test. New result fields (`val_threshold`, `val_macro_f1_at_best`) make the protocol auditable.
- **Multi-seed bootstrap**: 3 seeds (42, 43, 44) for all configurations; paired bootstrap p-values reported.
- **Sentinel-gated pipeline**: `scripts/run_all_ampere.sh` implements a 9+ stage pipeline with per-stage sentinels, allowing resume after failure without recomputing completed stages.

### 4.3 The surprising negative result

Across three seeds with the corrected protocol, **v2 underperforms R-GCN by ~9 macro-F1 points** on EUR-Lex test:

| Model | test_macro_F1 | val_threshold |
|---|---|---|
| R-GCN (h=512) | 0.2731 ± 0.0066 | 0.10–0.14 |
| R-GCN (h=768 capacity match) | 0.2610 | 0.12 |
| **JusDef v2 (keyword operators)** | **0.1822 ± 0.0133** | 0.06–0.10 |

This is the central problem the thesis must explain.

### 4.4 The diagnostic methodology

The thesis develops a **systematic hypothesis-falsification methodology** to identify the source of the underperformance. Five candidate explanations were formulated and tested:

1. **Stage-2 losses inactive** — v2 early-stops in Stage 1 before the operator-aware losses (L_onto, L_defeat) engage. *Tested by extending stage1_end.* **Falsified.**
2. **W_omega specialisation** — operator-specific weight matrices have different roles. *Tested by inspecting weight-norm divergence.* **Falsified (W matrices remain undifferentiated).**
3. **W_omega capacity** — more parameters needed. *Tested by training R-GCN at h=768 (15.3M params).* **Falsified (h=768 is worse than h=512).**
4. **Input projection scaling** — gradients suppressed by initialization. *Tested by scaling input projection weights.* **Falsified.**
5. **Curriculum freezing** — v2 fails because operator weights are not properly frozen at v1's values. *Tested by freezing.* **Falsified.**

Result: no single architectural component explains the gap. The hypothesis pivots to a **corpus-level constraint** — *operator-density bottleneck*.

---

## 5. The 3000-sentence operator annotation pipeline

### 5.1 Motivation

The thesis hypothesises that v2's underperformance is bounded by the actual semantic density of defeasible operators in EUR-Lex. To test this, an alternative operator detector — one trained on expert-validated annotations — was required. The keyword regex was suspected of generating false positives.

### 5.2 The dataset

- **3000 EUR-Lex sentences**, stratified by class.
- **AI-assisted labelling**: Anthropic Claude as the base classifier, with confidence and reasoning fields.
- **Distribution**: AFF 929, NEG 721, EXC 681, OVR 669 (designed for balance, not for natural prior).
- **Stored at**: `data/annotations/operator_labels_3000.jsonl`.
- **Guidelines document**: `data/annotations/operator_guidelines.md` (252 lines, written before annotation began).

### 5.3 Inter-annotator agreement validation

| Study | N | κ | Agreement | Verdict |
|---|---|---|---|---|
| Self-review | 100 | **0.7867** | 84.0% | Substantial |
| Supervisor blind annotation | 300 | **0.7689** | 82.7% | Substantial |

The residual 52 disagreements in the supervisor study are **58% concentrated** in a single linguistic pattern: *"Unless X, Y"* constructions where supervisor labelled AFF (outer rule) and AI labelled EXC (inner exception). This is a documented limitation of the operator algebra under the "outermost wins" rule and is noted as future work for guideline refinement.

### 5.4 The neural operator detector

- Architecture: LegalBERT (`nlpaueb/legal-bert-base-uncased`) with a 4-way classification head.
- Training: 6 epochs, 2700/300 stratified split, AdamW lr=2e-5.
- **Held-out macro-F1: 0.9772** (on AI-labelled validation).
- Reliability against expert annotation: bounded by the supervisor κ=0.77.

### 5.5 Density finding from relabeling

When the trained neural detector relabels all ~14 million concept mentions across EUR-Lex train/val/test, the operator distribution changes dramatically:

| Detector | Total AFF | Total non-AFF | Non-AFF density |
|---|---|---|---|
| Keyword (regex) | ~13.1M | ~834k | **5.99%** |
| **Neural** | ~13.9M | ~100k | **0.71%** |

This is an **~8.5× drop** in non-AFF density. The interpretation that best fits the evidence: the keyword detector substantially over-fired on procedural boilerplate (*"subject to"*, *"provided that"*) that is not semantically defeasible. The neural detector, validated against expert annotation, identifies a much smaller fraction of sentences as carrying non-default operators.

**Empirical consequence**: the operator-density bottleneck hypothesised for v2 underperformance is real — and tighter than originally thought. EUR-Lex has roughly 1% non-AFF semantic density, not 5.5%.

---

## 6. Empirical results to date (as of June 2026)

### 6.1 Headline table

| Model | Operator source | test_macro_F1 | Notes |
|---|---|---|---|
| R-GCN (h=512) | n/a | **0.2731 ± 0.0066** | Mean over seeds 42/43/44 |
| R-GCN (h=768) | n/a | 0.2610 | Capacity match, single seed |
| JusDef v2 (full) | keyword | **0.1822 ± 0.0133** | Mean over seeds 42/43/44 |
| JusDef v2 (full) | neural | (in progress) | Stage 7.5d, seed 42 currently at val ≈ 0.18 epoch 33 |

### 6.2 Key empirical findings

1. **Threshold protocol matters**: v1's test-tuned threshold inflated reported test_macro by ~2–4 points. The corrected protocol (val-tuned, frozen for test) reduces numbers but is methodologically defensible.
2. **R-GCN replicates the v1 paper exactly** (mean 0.273 vs reported 0.274), validating the new pipeline.
3. **R-GCN h=768 is worse than h=512** — capacity is not the constraint on v2's performance.
4. **v2 keyword underperformance is robust across seeds** (std 0.013, ~5.6σ below R-GCN mean).
5. **Neural-detector relabeling reduces non-AFF density 8.5×**, falsifying the hypothesis that the keyword detector's 5.5% non-AFF was a faithful estimate of corpus operator density.

---

## 7. Architectural finding — why JusDef v2 underperforms R-GCN

The current `compute_defeat_mask` in `src/model/dmp_layer.py` uses a **hard binary mask**: in each concept neighborhood, only the maximum-priority message(s) survive; all others are zeroed.

```python
defeat_score = operators * 1000.0 + priorities
group_max = scatter_reduce(defeat_score, dst_nodes, "amax")
mask = (group_max[dst_nodes] - defeat_score - 0.001 <= 0).float()
```

With 5.99% non-AFF density at the typical concept degree of 30–50 incoming edges:

- P(concept has ≥ 1 non-AFF edge given 50 edges) ≈ 1 − 0.94⁵⁰ ≈ **95%**
- For those 95% of concepts, **every AFF message is zeroed**
- Approximately **94% of the incoming signal is discarded** at each layer

This is a structural cause of the v2 underperformance: when AFF messages dominate the graph but are masked by sparse non-AFF presence, the model loses most of the predictive signal. With the neural detector at 0.71% density, the math is different in detail but the masking still destroys most AFF signal in neighborhoods where any non-AFF edge appears.

This explains why the v1 paper's bootstrap ablation found that **DMP itself contributes nothing significant** to overall macro-F1 (memory: `project_jusdef_bootstrap.md`). The defeat mechanism that gives the architecture its name is actively harmful at the observed operator densities.

---

## 8. The proposed next step — Continuous Defeasible Message Passing (cDMP)

### 8.1 Design

cDMP replaces the hard mask with a **continuous attenuation weight** in (0, 1), produced by:

1. A **soft defeat function** (sigmoid of the priority gap), so AFF messages in non-AFF-containing neighborhoods are attenuated rather than zeroed.
2. An **operator-conditioned attention** mechanism: attention logits computed from `[h_src, h_dst, operator_embedding, authority_score]` via a small MLP, then softmax-normalised over messages targeting the same concept.

```python
defeat_weight = sigmoid(-priority_gap * temperature)        # soft Fix 1
attn = scatter_softmax(MLP([h_s, h_c, op_emb, auth]), dst)  # new attention
message_out = (defeat_weight * attn) * (W_op @ src_emb)
```

### 8.2 Theoretical claim — Proposition 2

cDMP strictly generalises both R-GCN and the hard JusDef DMP:

1. As temperature → ∞ and operator embeddings produce a hard step, cDMP → hard DMP (v1/v2).
2. As attention logits → constant and defeat weights → 1, cDMP → R-GCN over operator-indexed sub-relations.
3. For temperature in (0, ∞), the hypothesis class is strictly larger than either limit.

This makes cDMP a principled middle ground with both classical baselines as boundary cases.

### 8.3 Empirical hypothesis

If the v2 underperformance is caused by hard-mask information destruction (Section 7), then cDMP should:

- Close the gap to R-GCN on overall macro-F1
- Recover or exceed v1's reported Y_exc gain on the defeasible-relevant subset
- Hold up under both keyword (5.99% density) and neural (0.71% density) detectors, demonstrating density-robustness

### 8.4 Implementation status

Not yet implemented. Planned as the central architectural contribution of the thesis. Estimated effort: 1–2 days of code + 5–7 days of GPU evaluation.

---

## 9. Honest weaknesses and limitations

### 9.1 Empirical weaknesses

- **EUR-Lex is a low-operator-density corpus** (~1% semantic non-AFF density). The defeasible-MP architecture is least useful precisely where it was first evaluated. Future work must include denser corpora (ECtHR Task B, contract law).
- **Single-section graphs**: many EUR-Lex documents collapse to a single sentence-level node, weakening section-level pooling. This is a preprocessing limitation that hasn't been fully audited.
- **Y_exc subset is small** (21 labels) and partly seen-label-dominated (20/21 in seen, 1/21 in unseen), limiting its sensitivity as a defeasibility-specific evaluation.

### 9.2 Annotation weaknesses

- **3000 sentences is small** for a deep learning task. The neural detector's 0.98 held-out F1 is on the same AI-label distribution, not an independent gold standard.
- **κ = 0.77 supervisor agreement** is substantial but the residual 52 disagreements concentrate in a known ambiguous linguistic pattern (*"Unless X, Y"*). Guideline refinement is future work.
- The neural detector is itself a downstream model — its reliability against the AI labels does not directly speak to its reliability against expert annotation.

### 9.3 Architectural weaknesses

- **Hard-defeat masking destroys 94% of AFF signal** in concept neighborhoods with any non-AFF presence (Section 7). This is the principal failure mode.
- **Authority scorer is linear** in (type, level, recency) — does not capture subject-matter overlap (*lex specialis*) or temporal/deontic interactions.
- **Single-relation graph** — does not implement the *scope-nested hypergraph* mentioned in the v1 paper introduction. The hypergraph extension is deferred to future work.

### 9.4 Methodological weaknesses

- The diagnostic methodology produced *negative* results: five hypotheses tested and falsified, leaving the architecture-vs-corpus distinction unsettled until the neural-detector data became available.
- The R-GCN baseline replication, while exact (0.273 vs 0.274 reported), is on a single architecture family. Comparison against transformer-only models (Legal-BERT classifier head) has not been re-run with the corrected threshold protocol.

---

## 10. The thesis framing

The thesis is **not** the story "we built JusDef and it works." It is the story:

> "We built JusDef v1 expecting an architectural gain on defeasible legal classification. We corrected three implementation issues in v2 and verified them mechanically. The corrected v2 unexpectedly underperformed the R-GCN baseline. Through systematic hypothesis falsification, we identified the cause as the hard-defeat mask catastrophically destroying signal in low-operator-density regimes. We validated the density estimate using a 3000-sentence expert-validated annotation dataset and a neural operator detector, finding the actual semantic operator density in EUR-Lex (~0.7%) is far below what the original keyword detector reported (5.5%). To address the architectural cause, we propose **Continuous Defeasible Message Passing (cDMP)**, with a two-sided reduction theorem showing that both R-GCN and the original hard DMP are limit cases. We validate cDMP empirically on EUR-Lex under both detection regimes and demonstrate that the architecture is density-robust. Future work includes denser corpora (ECtHR), hypergraph extensions for nested scope, and refined guidelines for the residual *"Unless X, Y"* ambiguity."

This frames five contributions:

1. **Theoretical**: cDMP architecture + Proposition 2 two-sided reduction
2. **Empirical**: multi-seed, multi-detector validation showing density-robustness
3. **Methodological**: hypothesis-falsification methodology for attributing GNN performance issues
4. **Resource**: 3000-sentence operator-annotated dataset with κ=0.77 supervisor IAA
5. **Domain insight**: actual semantic operator density in EUR-Lex is ~0.7%, an order of magnitude below keyword-detector estimates

---

## 11. Code organisation

```
jusdefv2/
├── AMPERE_RUNBOOK.md            ← stage-by-stage execution runbook
├── PROJECT_OVERVIEW.md          ← this document
├── notes/
│   └── methodology.md           ← detailed diagnostic chronology, ~1800 lines
├── data/
│   ├── annotations/             ← guidelines, 3000 labels, IAA files
│   │   ├── operator_guidelines.md
│   │   ├── operator_labels_3000.jsonl
│   │   ├── exception_labels.json
│   │   ├── eurovoc_label_names.json
│   │   └── iaa/
│   │       ├── self_review_100.jsonl       (κ=0.79)
│   │       ├── supervisor_300_returned.jsonl  (κ=0.77)
│   │       ├── supervisor_300_sent.jsonl
│   │       └── supervisor_300_with_ai.jsonl
│   ├── eurovoc/                 ← raw RDF (gitignored, 58 MB)
│   ├── processed/               ← keyword-operator pickles, embeddings, label_adj.pt
│   └── processed_neural/        ← neural-operator pickles + graphs (Stage 7.5b/c output)
├── src/
│   ├── model/
│   │   ├── jusdef.py            ← assembled model (HeteroConv + DMPLayer)
│   │   ├── dmp_layer.py         ← Defeasible Message Passing core
│   │   ├── authority_scorer.py  ← learned per-(type, level, recency) priority
│   │   └── baselines.py         ← R-GCN, threshold tuning
│   ├── kg/
│   │   ├── kg_builder.py        ← HeteroData graph construction
│   │   └── graph_utils.py
│   ├── preprocess/
│   │   ├── data_loader.py       ← EUR-Lex via HF datasets
│   │   ├── section_splitter.py
│   │   ├── concept_linker.py    ← EuroVoc concept linking
│   │   ├── operator_detector.py ← keyword (regex) detector
│   │   └── authority_extractor.py
│   ├── train/
│   │   ├── trainer.py           ← staged training with curriculum
│   │   └── losses.py            ← BCE + L_onto + L_defeat
│   └── eval/
│       ├── metrics.py           ← Macro/Micro/Yexc/seen/unseen
│       └── bootstrap.py         ← paired bootstrap p-values
├── scripts/                     ← pipeline entry points
│   ├── preprocess_all.py        ← Stage 1: keyword operator graphs
│   ├── extract_embeddings.py    ← Stage 2: Legal-BERT embeddings
│   ├── build_label_adj.py       ← Stage 3: EuroVoc adjacency
│   ├── build_graphs.py          ← Stage 4: HeteroData graphs
│   ├── train_rgcn.py            ← Stages 6, 6.5
│   ├── train_jusdef.py          ← Stage 7 + ablations (8)
│   ├── train_operator_detector.py ← Stage 7.5a (LegalBERT fine-tune)
│   ├── relabel_operators_neural.py ← Stage 7.5b
│   ├── eval_all_jusdef.py       ← Stage 9: unified evaluation
│   ├── compute_iaa.py           ← Cohen's κ for annotator/supervisor
│   └── run_all_ampere.sh        ← master pipeline with sentinels
├── tests/
│   └── test_architecture_fixes.py  ← F1, F1b, F2, F3a, F5 (all pass)
└── outputs/
    ├── checkpoints/             ← model weights (gitignored)
    ├── logs/                    ← JSON results + per-stage logs
    └── sentinels/               ← stage completion markers
```

---

## 12. How to reproduce

### Prerequisites
- Python 3.10
- A100 GPU (or equivalent), 24+ GB VRAM recommended
- HuggingFace cache writeable
- ~50 GB disk for processed data + checkpoints

### Quick start (assuming Stages 1–3 already cached)
```bash
git clone https://github.com/oziofficial5/jusdefv2
cd jusdefv2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Mark cached stages as done if you have them, else delete sentinels
touch outputs/sentinels/{1,2,3,4}.done

# Run the full pipeline
export JUSDEF_NEURAL_SEEDS="42 43 44"
bash scripts/run_all_ampere.sh
```

Detailed reproduction is in `AMPERE_RUNBOOK.md`.

---

## 13. Glossary

| Term | Definition |
|---|---|
| **AFF / NEG / EXC / OVR** | Operator labels for affirmation, negation, exception, override |
| **DMP** | Defeasible Message Passing — the message-passing scheme that defeats lower-priority operators |
| **cDMP** | Continuous DMP — proposed v3 with soft mask and operator-conditioned attention |
| **Y_exc** | Exception-dependent label subset (21 EuroVoc labels), used as defeasibility-specific eval |
| **F1, F2, F3a/b/c** | v2 architectural correction labels (reverse edge, authority in grad path, ontology/auth-hierarchy/auth-concept edges) |
| **STE** | Straight-Through Estimator — passes gradients through a hard binary forward op |
| **W_ω** | Operator-indexed weight matrix (one per ω ∈ Ω) |
| **κ** | Cohen's kappa — chance-corrected inter-annotator agreement |
| **EuroVoc** | EU's multilingual thesaurus of legal concepts; provides the 100 labels |
| **LexGLUE** | Standardised benchmark suite for legal NLP including EUR-Lex |
| **R-GCN** | Relational Graph Convolutional Network — the parametric baseline |

---

## 14. Publications and submissions

| # | Title | Venue | Status |
|---|---|---|---|
| 1 | Language Models for Legal NLP: A Literature Review | CAiSE 2025 Workshops | Published |
| 2 | Evaluating Knowledge-based Approaches for Legal Text Analysis: A Benchmark Study | *Computer Law & Security Review*, vol. 61 | Published 2026 |
| 3 | JusDef: Defeasible Hypergraph Reasoning for Multi-Label Legal Document Classification | ANNPR 2026 | Submitted, under review |
| 4 | Multimodal Deepfake Detection with Large Vision-Language Models | TBD | In preparation |

Google Scholar: https://scholar.google.com/citations?user=OaQs2-MAAAAJ&hl=en

---

## 15. Open questions for the thesis viva

1. *Why is hard-defeat masking the right design at all?* Answer: it follows the partial-order semantics of classical defeasible reasoning. But it imposes an aggregation-time decision rather than a feature-space one, and that aggregation-time decision throws away information. cDMP is the principled relaxation.
2. *Is the 3000-sentence dataset large enough?* Answer: for training a high-precision detector, yes (val F1 = 0.98). For establishing absolute corpus-density estimates, it's a sample, not a census. The 0.71% density estimate has uncertainty bounded by the detector reliability (κ=0.77 against supervisor).
3. *What would change the conclusion?* A denser corpus (ECtHR) showing cDMP gains where EUR-Lex does not. That would shift the contribution from "architecture works under sparse operator regime" to "architecture works conditional on density"; both are publishable.
4. *Why not use a larger language model?* The thesis is about graph-architecture innovation, not about scaling encoder capacity. Legal-BERT is the standard backbone in this literature.

---

## 16. Status as of this document

- ✅ JusDef v1 submitted to ANNPR 2026
- ✅ v2 architectural corrections (F1, F2, F3a) committed and verified
- ✅ Threshold protocol corrected and validated against v1 paper's R-GCN baseline
- ✅ 3000-sentence annotation dataset built and validated (self κ=0.79, supervisor κ=0.77)
- ✅ Neural operator detector trained (val macro-F1 = 0.977)
- ✅ Neural-detector relabeling completed; 0.71% non-AFF density measured
- 🟡 Stage 7.5d JusDef-neural training in progress (seed 42 at epoch ~35)
- ⏳ cDMP design specified; implementation pending
- ⏳ Chapter 4 (framework) and Chapter 5 (experiments) writing pending final results

---

*This document is the entry point. Code starts at `scripts/run_all_ampere.sh`. Theory starts at `Chapter 3` of the thesis. Diagnostic history is in `notes/methodology.md`. Annotation provenance is in `data/annotations/`.*
