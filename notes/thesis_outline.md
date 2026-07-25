# PhD Thesis Outline (final, post-density-measurement)

**Working title**: *Defeasibility-Aware Graph Neural Networks for Legal Multi-Label Classification: A Diagnostic Framework, Operator-Annotated Corpus, and Signal-Preserving Architecture*

**Author**: Awais Abdul Khaliq
**Supervisors**: Prof. Stefano Montanelli, Prof. Alfio Ferrara
**Target submission**: 15 September 2026
**Outline status**: final — incorporating 14 June 2026 ECtHR density measurement

---

## Thesis statement (one sentence)

*The empirical benefit of defeasibility-aware graph neural networks on legal multi-label classification is bounded by three architectural failure modes documented in the broader graph-learning literature (Straight-Through Estimator bias on hard gates, sign cancellation across depth in signed graph neural networks, and pretrained-encoder negation blindness) interacting with a previously-uncharacterised property of natural legal corpora: explicit defeasible-operator density is intrinsically low (< 1% non-AFF) on both EU regulations (EUR-Lex) and European human-rights case law (ECtHR); we contribute a diagnostic methodology grounded in named prior failures, an operator-annotated corpus with expert inter-annotator validation (Cohen's κ = 0.77), the first two-corpus measurement of explicit operator density in legal NLP, the Y\_exc defeasibility-stratified evaluation protocol, and a signal-preserving architecture that recovers R-GCN performance as a limit case while preserving defeasible semantics.*

---

## Contributions (mapped to chapters)

| # | Contribution | Type | Primary chapter |
|---|---|---|---|
| **C1** | Diagnostic methodology for attributing failure in defeasibility-aware GNNs (hypothesis-falsification chain on capacity, threshold, density) | Methodological | Ch. 6 |
| **C2** | Y\_exc defeasibility-stratified evaluation protocol (no published precedent for multi-label legal classification) | Methodological | Ch. 3, Ch. 8 |
| **C3** | **Two-corpus explicit operator density measurement** (EUR-Lex 0.71%, ECtHR 0.52-0.58%) — first systematic characterisation in legal NLP | Empirical | Ch. 6 |
| **C4** | 3000-sentence operator-annotated corpus with expert IAA (Cohen's κ = 0.77 independent annotator) | Resource | Ch. 5 |
| **C5** | Neural operator detector with documented density calibration (5.99% keyword → 0.71% neural on EUR-Lex; 0.58% on ECtHR) | Resource | Ch. 5 |
| **C6** | v3 architecture (signal-preserving, signed-coefficient + soft attention) grounded in three NAMED prior failure modes | Architectural | Ch. 7 |
| **C7** | Theoretical claims: v3 strictly generalises R-GCN (Prop 2) and recovers hard DMP as a limit case (Prop 3) | Theoretical | Ch. 7 |

C1, C2, C3, C4 are the **primary contributions**. C5 is the **secondary** infrastructure. C6 and C7 are the **tertiary** architectural payoff.

---

## Why this framing (final, post-density-measurement)

Three pieces of evidence drove the final framing:

1. The June 2026 literature review showed there is **no clean architectural novelty** at the GNN+defeasible+legal intersection. Every individual component of any v3 architecture has prior art in adjacent communities (FAGCN/Signed-GCN for signed aggregation, CompGCN for subtractive composition, HGT/HAN for typed attention, Cocarascu and Galassi for argumentation GNNs).

2. The 14 June 2026 density measurement on AUEB-NLP/ecthr_cases produced 0.52% (all paragraphs) and 0.58% (rationale paragraphs) non-AFF density — *below* EUR-Lex's 0.71%. The hypothesis "case law is denser in defeasibility markers than regulations" is **falsified** by this dataset. Two corpus measurements now both show < 1% explicit operator density.

3. The Stage 7.5 multi-seed JusDef-neural result (0.18 ± 0.01 across 3 seeds) confirmed that operator-density alone does not change downstream performance. Combined with R-GCN's robust 0.273 baseline, the architectural gap is not a density artefact.

The defensible thesis is therefore methodology + dataset + targeted architectural fix grounded in cited prior failures, with empirical claims on Y\_exc rather than overall macro-F1. The two-corpus density measurement is now a primary contribution (C3) because it generalises beyond one dataset.

EUR-Lex transformer baselines (Chalkidis et al. ACL 2022) hit macro-F1 ≈ 0.57 with LegalBERT. Graph-only baselines including R-GCN hit ~0.27. **The thesis does not claim macro-F1 SOTA** — that is not what graph-only models do on this benchmark.

---

## Research questions (formal)

**RQ1** (methodological): *Does operator density alone explain the empirical underperformance of hard-defeat message passing on EUR-Lex relative to R-GCN baselines?*

**RQ2** (resource): *Can a neural operator detector trained on expert-validated annotations produce reliable operator labels at scale, and how does the resulting operator density compare to keyword-based detection?*

**RQ3** (empirical, two-corpus): *Is explicit defeasible-operator density a corpus-intrinsic property of natural legal text, and does it generalise from EU regulations to human-rights case law?*

**RQ4** (architectural): *Can a defeasibility-aware GNN architecture grounded in three named prior failure modes (STE bias, signed-GNN sign cancellation, encoder negation blindness) recover R-GCN-level overall performance while delivering measurable gains on a defeasibility-stratified evaluation subset?*

---

## Chapter-by-chapter structure

### Chapter 1 — Introduction (~10 pages)

- 1.1 Motivation: legal text encodes non-monotonic norms
- 1.2 The problem: defeasibility-aware GNNs lack benchmarks, datasets, validated architectures, and corpus-density characterisation
- 1.3 Thesis statement and research questions (RQ1–RQ4)
- 1.4 Contributions (C1–C7)
- 1.5 Publications arising from the thesis
- 1.6 Thesis structure
- 1.7 Reading guide

**Anchor sentence**: "We do not claim macro-F1 SOTA on EUR-Lex. We claim a methodological framework, an annotated resource, a two-corpus empirical density characterisation, and a targeted architectural improvement on a defined defeasibility-stratified subset."

---

### Chapter 2 — Background and Related Work (~20 pages)

Sections 2.1 through 2.11 as listed in the previous outline draft. **No changes from previous version.** Key citation clusters preserved:

- 2.1 Defeasible reasoning (Reiter, Pollock, Dung)
- 2.2 Neuro-symbolic defeasible reasoning (Garcez & Lamb, DeepProbLog, Riveret line)
- 2.3 Legal NLP and EUR-Lex baselines (Chalkidis et al.)
- 2.4 GNNs for legal text (D2GCLF, CaseGNN, LA-MGFM)
- 2.5 Heterogeneous typed-edge GNNs (R-GCN, CompGCN, HGT, HAN)
- 2.6 Signed and polarity-aware GNNs ("Sign is Not a Remedy" 2024)
- 2.7 Hard masking vs soft attention (Gapped STE ICML 2022)
- 2.8 Negation in pretrained encoders (Semantic Adapter 2025, ScoNe)
- 2.9 Argumentation GNNs (Cocarascu, Galassi, Lippi, Multi-view HGNN 2024)
- 2.10 Long-tail multi-label evaluation (methodological analog for Y\_exc)
- 2.11 Synthesis: the precise gap at the GNN + defeasibility + legal intersection

---

### Chapter 3 — Problem Formulation (~15 pages)

This chapter is partially drafted at `chapter3_problem_formulation.tex` and will be updated.

- 3.1 Task definition: multi-label legal classification on EUR-Lex
- 3.2 Operator algebra Ω = {AFF, NEG, EXC, OVR}
- 3.3 The heterogeneous legal graph G
- 3.4 Defeasible message passing — abstract definition (used by v1, v2, v3)
- 3.5 The R-GCN baseline and Proposition 1 (reduction)
- 3.6 **Y\_exc — the defeasibility-stratified evaluation subset** (C2)
- 3.7 Evaluation metrics and multi-seed protocol
- 3.8 The four formal research questions

---

### Chapter 4 — The JusDef Framework: v1 + v2 corrections (~15 pages)

- 4.1 JusDef v1 architecture (ANNPR contribution)
- 4.2 The authority scorer
- 4.3 Proposition 1: JusDef-v1 reduces to R-GCN in the no-defeat regime
- 4.4 The v2 architectural corrections (F1, F2, F3a) and unit-test verification
- 4.5 The corrected evaluation protocol
- 4.6 Software architecture and reproducibility (AMPERE_RUNBOOK.md)
- 4.7 What v1 and v2 alone cannot answer (motivates Chapters 5 and 6)

---

### Chapter 5 — An Operator-Annotated Corpus for Legal Defeasibility (~15 pages)

This is C4 + C5 (primary resource contributions).

- 5.1 Motivation: why no existing dataset suffices
- 5.2 The operator algebra in practice
- 5.3 Annotation guidelines (252-line document)
- 5.4 AI-assisted annotation pipeline
- 5.5 Inter-annotator agreement validation
  - Self-review (100 sentences, κ = 0.79)
  - Independent-annotator blind annotation (300 sentences, κ = 0.77)
  - Disagreement structure analysis (*"Unless X, Y"* concentration)
- 5.6 The neural operator detector (val macro-F1 = 0.977)
- 5.7 Dataset release
- 5.8 Discussion: AI-assistance in legal annotation pipelines

**Note**: This chapter is publication-quality material in its own right (target: LREC-COLING).

---

### Chapter 6 — Diagnostic Methodology and Two-Corpus Density Findings (~22 pages)

This is C1 (primary methodological contribution) and C3 (the two-corpus density measurement).

- 6.1 The hypothesis-falsification methodology
- 6.2 Setting up the diagnostic: R-GCN baseline (0.273 ± 0.007), JusDef v2 (0.182 ± 0.013)
- 6.3 **Hypothesis H1**: Threshold protocol bug → corrected, still underperforms → **falsified as sole cause**
- 6.4 **Hypothesis H2**: Capacity bottleneck → R-GCN h=768 is worse → **falsified**
- 6.5 **Hypothesis H3**: Stage-2 auxiliary losses inactive → extended runs don't help → **falsified**
- 6.6 **Hypothesis H4**: Operator-density bottleneck
  - 6.6.1 Trained neural detector measures EUR-Lex density at 0.71% (vs keyword's 5.99%)
  - 6.6.2 Multi-seed JusDef-neural test_macro = 0.18 ± 0.01 (identical to keyword)
  - 6.6.3 **Falsified for EUR-Lex** — density change does not change performance
- 6.7 **The two-corpus density measurement (C3, new addition)**
  - 6.7.1 Motivation: does the density-invariance generalise across legal corpora?
  - 6.7.2 ECtHR cases dataset (AUEB-NLP, 1000 test cases)
  - 6.7.3 Neural detector applied to ECtHR sentences (same model as EUR-Lex)
  - 6.7.4 ECtHR all paragraphs: 0.52% non-AFF
  - 6.7.5 ECtHR rationale paragraphs (silver_rationales): 0.58% non-AFF
  - 6.7.6 **Finding: both corpora show < 1% non-AFF density**. The density-floor is corpus-intrinsic.
  - 6.7.7 Limitations: AUEB-NLP/ecthr_cases contains Facts text; Law sections are HUDOC-only. The measurement therefore characterises Facts-section density; future work needs HUDOC scraping for Law-section measurement.
- 6.8 **Mechanistic diagnosis** — three architectural failure modes
  - 6.8.1 Failure mode 1: hard STE gate bias (Jang ICLR 2017; Gapped STE ICML 2022)
  - 6.8.2 Failure mode 2: asymmetric W-matrix training (operator-specific weights with 100× imbalance in training examples)
  - 6.8.3 Failure mode 3: pretrained-encoder negation blindness (ScoNe ACL 2020, Truong ACL Findings 2023, Semantic Adapter 2025)
- 6.9 Synthesis: each failure mode is cited from broader GNN/encoder literature — this is what enables the principled v3 design (Chapter 7)

---

### Chapter 7 — A Signal-Preserving Architecture (v3) (~15 pages)

This is C6 + C7. v3 is presented as a principled synthesis of fixes for three named prior failure modes, not as a novel architecture in its own right.

- 7.1 Design principles
  - Address each failure mode from §6.8 with a targeted architectural choice
  - Maintain reducibility to R-GCN as a safety guarantee (Proposition 2)
  - Stay close to standard heterogeneous GNN templates (HAN, HGT)
- 7.2 The v3 architecture (`src/model/v3_layer.py` on branch `jusdef-v3`)
  - 7.2.1 **Single shared W transform** (replaces 4 separate W_ω)
    - Every message contributes to W training
    - Addresses Failure mode 2 (W-undertraining)
  - 7.2.2 **Soft sigmoid attention** (replaces hard STE binary mask)
    - All messages contribute with weights in (0, 1)
    - Addresses Failure mode 1 (STE bias)
    - Cite: Jang et al. ICLR 2017; Gapped STE ICML 2022
  - 7.2.3 **Per-operator signed coefficient with drift regulariser**
    - Coefficients initialised AFF=+1, NEG=−1, EXC=−0.5, OVR=+1
    - L2 regulariser pulls coefficients toward init to prevent collapse
    - Addresses signed-GNN sign cancellation
    - Cite: "Sign is Not a Remedy" 2024; "Signed Graph Approach to Oversmoothing" 2025
  - 7.2.4 **Operator-conditioned attention MLP**
    - Attention logits computed from `[shared_msg, op_emb, auth_emb, dst_emb]`
    - Cite: HGT (Hu et al. WWW 2020); HAN (Wang et al. WWW 2019)
- 7.3 Theoretical claims (proofs in Appendix C)
  - **Proposition 2** (Reduction to R-GCN): with `op_coef = (1,1,1,1)` and uniform attention, v3 reduces to R-GCN over the `mentions` relation. *Safety guarantee: v3 cannot perform worse than R-GCN with appropriate parameter setting.*
  - **Proposition 3** (Hard-DMP limit): as attention sharpens to one-hot and op_coef → (1,0,0,0), v3 recovers the surviving-message structure of hard DMP without W-undertraining noise.
  - **Proposition 4** (Strict generalisation): v3's hypothesis class strictly contains both R-GCN's and hard DMP's.
- 7.4 Implementation: `src/model/v3_layer.py`, integration via `dmp_variant="v3"` in `JusDef.__init__`, training via `scripts/train_jusdef.py --dmp_variant v3`. Unit tests verify Propositions 2 and 4 empirically (passing).
- 7.5 Hyperparameters and training protocol

---

### Chapter 8 — Experimental Evaluation (~20 pages)

- 8.1 Experimental setup (hardware, software, seeds, bootstrap)
- 8.2 Main results table
  - R-GCN h=512 mean (3 seeds): 0.2731 ± 0.0066
  - R-GCN h=768 (1 seed capacity match): 0.2610
  - JusDef v2 hard-DMP (3 seeds keyword): 0.1822 ± 0.0133
  - JusDef v2 hard-DMP (3 seeds neural): ~0.18 (3 seeds in)
  - **JusDef v3 (3 seeds keyword)**: TBD
  - **JusDef v3 (3 seeds neural)**: TBD
  - Transformer baselines from LexGLUE (context, not SOTA claim)
- 8.3 Y\_exc evaluation (C2)
- 8.4 Two-corpus density sensitivity (C3 validation)
- 8.5 Ablation studies
  - v3 without sign-cancellation regulariser
  - v3 without operator-conditioned attention
  - v3 with separate W_op (revert Failure-Mode-2 fix)
  - v3 with hard mask (revert Failure-Mode-1 fix)
  - v3 with op_coef = (1,1,1,1) — empirical R-GCN reduction check (Proposition 2)
  - Authority scorer ablation
  - F3a ontology edges ablation
- 8.6 Significance testing (paired bootstrap, 10000 resamples)
- 8.7 Qualitative analysis
- 8.8 Compute and reproducibility

---

### Chapter 9 — Discussion (~10 pages)

- 9.1 What the architecture buys (Proposition 2 floor + Y\_exc gain)
- 9.2 Where defeasibility-aware GNNs win and lose
- 9.3 The corpus-density-architecture interaction (the two-corpus finding)
- 9.4 Encoder negation blindness as a binding constraint
- 9.5 Limitations
- 9.6 Threats to validity

---

### Chapter 10 — Conclusion and Future Work (~7 pages)

- 10.1 Summary of contributions (C1–C7)
- 10.2 Answering the four research questions (RQ1–RQ4)
- 10.3 Practical implications for legal NLP practitioners
- 10.4 Future work
  - HUDOC full-judgment density measurement (need Law-section text)
  - Hypergraph extension for nested scope
  - Multilingual extension (MultiEURLEX)
  - Stronger encoders (DeBERTa-v3, Legal-DeBERTa)
  - LLM-based operator detection
  - Argumentation-mining pipeline integration
- 10.5 Closing thoughts

---

### Appendices

- A — Operator annotation guidelines (full 252-line document)
- B — Inter-annotator agreement methodology and per-class statistics
- C — Full proofs of Propositions 1, 2, 3, 4
- D — Reproducibility: full Ampere pipeline scripts, sentinel structure, GPU-hour table
- E — Per-label and per-document failure analysis
- F — Neural operator detector training logs
- G — ECtHR density measurement protocol and per-case breakdown
- H — Bibliography

---

## Pages budget

| Chapter | Pages |
|---|---|
| 1 Introduction | 10 |
| 2 Background & Related Work | 20 |
| 3 Problem Formulation | 15 |
| 4 JusDef Framework (v1+v2) | 15 |
| 5 Operator-Annotated Corpus | 15 |
| 6 Diagnostic + Two-Corpus Density | 22 |
| 7 v3 Architecture | 15 |
| 8 Experimental Evaluation | 20 |
| 9 Discussion | 10 |
| 10 Conclusion & Future Work | 7 |
| Appendices | 30 |
| **Total** | **179 + 30 appendix = ~209 pages** |

---

## Writing schedule (June 14 → September 15, 93 days)

### Week 1 (June 14–20): branch + v3 + outline
- Outline finalised (this document) ✓
- v3 architecture implemented on branch `jusdef-v3` ✓
- All architecture tests pass (9 passed, 2 xfailed) ✓
- Submit DOCX PhD activity report by June 16
- Pilot v3 on islab (single seed, ~10h GPU)

### Week 2 (June 21–27): v3 full evaluation
- v3 3 seeds keyword + 3 seeds neural (~60h GPU)
- **Internal committee presentation June 22 OR July 2**
- Chapter 5 (corpus) draft

### Week 3 (June 28–July 4): ablations + writing
- v3 ablations (~30h GPU)
- Chapter 6 (diagnostic) draft including two-corpus density section
- Chapter 7 (v3) draft

### Week 4 (July 5–11): writing core
- Chapter 8 (experiments) draft
- Y\_exc analysis
- Chapter 2 (related work) finalised

### Week 5 (July 12–18): integration
- All chapters first-draft complete
- Send to supervisors

### Week 6–7 (July 19–Aug 1): supervisor review + revisions

### Week 8–10 (Aug 2–22): polish + appendices + figures

### Week 11–12 (Aug 23–Sep 7): submission prep + viva prep

### Sep 13–15: submission

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| v3 underperforms R-GCN even on Y\_exc | Proposition 2 guarantees v3 ≥ R-GCN by construction. If empirical v3 underperforms, it indicates a regulariser/hyperparameter issue, not an architectural impossibility. Pilot identifies this in 10 hours. |
| Internal committee June 22 isn't ready | Present diagnostic + two-corpus density + dataset + early v3 pilot. The methodology + dataset + density are complete material. |
| Supervisor wants different framing | Outline is editable. The literature review and density measurement back every framing decision. |
| GPU budget runs out | Plan uses ~130 GPU-hours of the ~1-month booking; substantial buffer. |
| HUDOC full-judgment density measurement is needed mid-thesis | Explicitly noted as future work (§10.4). Thesis stands without it. |
| ANNPR paper review returns major revisions | The thesis cites the workshop version as v1; major revisions to ANNPR do not change the thesis structure. |

---

## What this outline commits us to (final)

1. **One architecture**: v3 as specified in Chapter 7, with the four targeted modifications and Propositions 2/3/4. No more architectural pivots.
2. **One evaluation protocol**: Y\_exc-stratified, on EUR-Lex with both keyword and neural detectors. No corpus pivots without supervisor agreement.
3. **Two empirical density data points**: EUR-Lex 0.71% and ECtHR 0.58%. The two-corpus finding (C3) is now part of the thesis.
4. **No SOTA claim**: empirical target is recovery of R-GCN baseline (overall macro-F1) and gain on Y\_exc, not transformer-comparable macro-F1.
5. **Three primary contributions** (methodological, dataset, density); two secondary (architectural, theoretical).

The thesis is publishable even if v3 only matches R-GCN, because the methodological and dataset contributions are independently sufficient.

---

*This outline supersedes all earlier "v3 candidate architecture" notes. Future architectural decisions cite this outline. The v3 implementation is at `src/model/v3_layer.py` on branch `jusdef-v3`.*
