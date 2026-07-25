

\# JusDef Methodology Audit



\*\*Author:\*\* Awais Abdul Khaliq  

\*\*Project:\*\* JusDef (Defeasible Message Passing for Legal Document Classification)  

\*\*\*\*Last updated:\*\* 2026-06-03



\*\*Repositories:\*\*



\- v1 (ANNPR submission): \[https://github.com/oziofficial5/Jusdef](https://github.com/oziofficial5/Jusdef)  

\- v2 (thesis version): \[https://github.com/oziofficial5/jusdefv2](https://github.com/oziofficial5/jusdefv2)



\---



\## 1. Purpose of this document



This document records every methodological issue, diagnostic experiment, architectural fix, and integrity check performed during the JusDef project. It is intended to support thesis-level reproducibility and defensibility. Each section follows the pattern: \*\*what was observed → how it was diagnosed → what was changed → how it was verified\*\*.



This is \*\*not\*\* a polished narrative. It is a chronological audit that deliberately includes negative findings, false leads, and partial fixes. Together these document the work that produced the published v1 paper and the v2 thesis extension.



\---



\## 2. Project chronology



| Date       | Milestone                                                                                       |

|-----------|--------------------------------------------------------------------------------------------------|

| 2026-04-12 | Y\_exc 21-label subset constructed (first author, EuroVoc descriptor inspection)               |

| 2026-04-24 | h768\_lowdef negative experiment (capacity hypothesis falsified)                                |

| 2026-05-03 | ANNPR submission camera-ready preparation begins                                               |

| 2026-05-08 | v1 paired-bootstrap analysis completed (seed 42)                                               |

| 2026-05-11 | v1 multi-seed eval-bug discovered; corrected numbers re-extracted                              |

| 2026-05-17 | v1 forward-pass diagnostic: operators do not reach predictions                                 |

| 2026-05-21 | v2 F1 (reverse edge) and F2 (authority scorer) fixes complete on laptop                        |

| 2026-05-25 | v2 Ampere preprocessing complete                                                               |

| 2026-05-27 | v2 seed 42 first multi-seed result (underperforms v1)                                          |

| 2026-05-29 | W\_omega divergence diagnostic — v1's W matrices found undifferentiated                         |

| 2026-05-30 | Input-projection scaling and freeze diagnostics completed                                      |

| 2026-06-02 | Multi-seed R-GCN control on v2 graphs completed                                               |

\---



\## 3. v1 — paper–code consistency audit



Discrepancies between the v1 manuscript (paper §4.4, §5.2) and the released code.



\### 3.1 Staged training boundaries



\- \*\*Paper (v1) §4.4:\*\* “Stage 1 (epochs 0–4) uses L\_cls; Stage 2 (5–14) adds L\_onto; Stage 3 (15+) adds L\_defeat).”

\- \*\*Code (`scripts/train\_jusdef.py`):\*\* `--stage1\_end 50 --stage2\_end 100`.

\- \*\*Effect:\*\* in v1 the canonical multi-seed runs (seeds 42, 43, 44) used the code defaults (50/100), not the paper’s boundaries (5/15).

\- \*\*Practical consequence:\*\* all canonical v1 runs early-stopped before reaching Stage 2. The staged curriculum described in the paper never engaged.



\### 3.2 Patience



\- \*\*Paper §5.2:\*\* “patience 30 on validation Macro-F1”.

\- \*\*Code default:\*\* `--patience 20`.

\- \*\*Resolution:\*\* paper text was descriptive of an earlier configuration; canonical runs used patience 20.



\### 3.3 Pilot-study claim



\- \*\*Paper §4.4:\*\* “in pilot studies ā(m) collapsed to a near-uniform distribution within five epochs.”

\- \*\*Reality:\*\* the code’s stage boundaries (50/100) are inconsistent with this pilot evidence. The staged schedule appears to be a precaution rather than empirically required.



\### 3.4 Threshold disclosure



\- \*\*Paper §5.2:\*\* “θ = 0.10 for JusDef, θ = 0.14 for R-GCN”.

\- \*\*Reality:\*\* per-seed thresholds vary (multi-seed JusDef: seed 42 → 0.10, seed 43 → 0.10, seed 44 → 0.14).

\- \*\*Disclosed in:\*\* v1 paper Table 4 (bootstrap table footnote).



\### 3.5 Y\_exc list provenance



\- \*\*Paper §3:\*\* “manual annotation (single annotator over the EuroVoc descriptor list, classified by surface analysis of the corpus)”.

\- \*\*Repository (`data/annotations/exception\_labels.json`):\*\* “Manual inspection of EuroVoc label semantics. Labels selected if EU legislation in that topic area frequently contains exception clauses, derogations, overrides, or conditional applicability provisions.” Annotator: Awais Abdul Khaliq, date: 2026-04-12.

\- \*\*Note:\*\* the repository wording is more defensible than the paper’s wording. The camera-ready paper was edited to match the repository protocol.



\---



\## 4. v1 — empirical-result audit



\### 4.1 The seed 43/44 evaluation bug



\*\*Observed (2026-05-11):\*\* During preparation of multi-seed bootstrap analysis, seeds 43 and 44 of the `jusdef\_full` configuration produced test Macro-F1 values of 0.0867 and 0.0856 respectively, dramatically lower than seed 42’s 0.2519. Initial concern: training instability across seeds.



\*\*Diagnosed:\*\* Inspection of `outputs/logs/\*.json` revealed that per-experiment evaluation scripts had applied threshold tuning directly to raw logits instead of sigmoid probabilities. Thresholds were tuned in a regime where they could not produce meaningful classifications.



\*\*Corrected:\*\* The centralised `eval\_all\_jusdef.py` script applies the correct pipeline (sigmoid of logits → threshold tuning → F1 computation). Re-running this script on the existing checkpoints produced sensible numbers without re-training.



| Tag              | Original (broken) | After centralised eval |

|------------------|-------------------|-------------------------|

| full\_s42         | 0.2519            | 0.2519 ✓               |

| full\_s43         | 0.0867            | 0.2453 ✓               |

| full\_s44         | 0.0856            | retrained → 0.2463     |

| no\_dmp\_s42       | 0.0957            | 0.2477                 |

| no\_auth\_s42      | 0.0894            | 0.2595                 |

| no\_dmp\_no\_auth\_s42 | 0.0872          | 0.2523                 |



Only the seed‑44 checkpoint had a genuine training failure (predated the sigmoid fix in the evaluator). All other checkpoints recovered when evaluated correctly.



\*\*Reported in:\*\* v1 supplementary; all v1 results in the paper use numbers from the centralised eval pipeline.



\### 4.2 h768\_lowdef negative experiment



\*\*Hypothesis tested:\*\* Y\_exc gain over R-GCN might be a parameter capacity effect. Test: train JusDef with hidden\_dim=768 (vs default 512) and reduced λ\_defeat.



| Seed              | Macro-F1 | Y\_exc   |

|-------------------|----------|--------|

| h768\_lowdef\_s42   | 0.2316   | 0.3247 |

| h768\_lowdef\_s43   | 0.2413   | 0.3538 |

| Reference (h512 full) seed 42 | 0.2519 | 0.3562 |



Doubled hidden dim produced \*\*worse\*\* Macro-F1 and Y\_exc on both seeds. Capacity alone did not explain v1 gains. This was documented as evidence in v1 paper §7 discussion.



\### 4.3 Stage 2 destabilisation pilot



\*\*Observed:\*\* A pilot run of the `no\_dmp\_no\_auth` configuration on seed 44 trained past Stage 1 into Stage 2. Training loss tripled at epoch 80 (from \~0.105 to \~0.30) when the ontology contrastive loss reached full weight. Validation Macro-F1 did not improve.



\*\*Cause:\*\* `OntologyContrastiveLoss` uses temperature 0.07. Tight-temperature contrastive losses are numerically sensitive; the embedding norms drifted over \~30 epochs of Stage 2 exposure, causing the contrastive denominator to saturate and loss magnitude to jump.



\*\*Implication:\*\* the staged curriculum is fragile in the low-defeat regime. v1’s runs avoided this only because patience triggered early stopping before Stage 2 fired.



\*\*Documented in:\*\* v1 paper §4.4 (“limited benefit in the current low-defeat regime”) and supplementary methodology notes.



\### 4.4 Capacity-matched experiment failure



\*\*First attempt (2026-05-15):\*\* Implemented `randomize\_r2\_operators` to overwrite r2 operator labels with random values, intending to test whether the Y\_exc gain came from operator semantics or from parameter capacity.



\- \*\*Problem 1:\*\* Bash wrapper missed the `--capmatch` flag. Training ran without randomisation. Detected by checking printed `Capmatch=False` in logs. Fixed.

\- \*\*Problem 2:\*\* After fixing the flag, MD5 comparison of the checkpoint (with vs without capmatch) showed differences, but evaluation produced identical scores to four decimals across five metrics. Probability of that by chance: ≈ 10⁻²⁰.



\*\*Root cause (2026-05-17):\*\*



\- Grep over `src/model` revealed zero references to `edge\_operator` or `operator\_id`.

\- The model reads operator information from `edge\_attr\_dict\[r2\_key]\["operator"]`, which the trainer builds from `g\[r2\_key].operator`.

\- The randomisation function modified `g.edge\_operator` — an attribute the model never reads.



\*\*Forward-pass diagnostic:\*\* calling `model.forward()` twice with identical input except for randomised operators showed:



| Node  | Δ when operators randomised |

|-------|-----------------------------|

| sec   | 0.0 (operator-blind)        |

| conc  | 0.20 (DMP responds)         |

| doc   | 0.0                         |

| auth  | 0.0                         |

| label | 0.0                         |



Operators affect concept embeddings via DMP. Concept embeddings never reach section, document, or label embeddings because the r2 edge type is one-directional (sec → conc). Predictions (which depend on section/document/label embeddings) are \*\*operator-blind\*\*.



\*\*Implication:\*\* the v1 architecture, as implemented, did not use operator information at prediction time. The capacity-matched experiment is meaningless under v1 — there is no operator signal in the prediction path to capacity-match.



This finding triggered the v2 architectural revision.



\---



\## 5. v2 — architectural fixes



\### 5.1 F1 — reverse mention edge



\*\*Problem:\*\* Operators affect concept embeddings via DMP, but concept embeddings have no path back to section/document/label embeddings. Predictions are operator-blind (verified in §4.4).



\*\*Fix:\*\* Add a reverse edge type `('conc', 'mentions\_rev', 'sec')` to the heterogeneous graph, with the same operator and priority attributes as the forward edge. Add a SAGEConv for this edge type in the model’s `conv\_dict`.



\*\*Files modified:\*\*



\- `src/kg/kg\_builder.py`: construction of reverse edges with mirrored attributes.  

\- `src/model/jusdef.py`: SAGEConv added to `conv\_dict`.  

\- `tests/test\_architecture\_fixes.py::test\_F1\_operators\_reach\_sections`: verifies that randomising r2 operators now changes section embeddings.



\*\*Verification:\*\*



\- F1 test: PASSES (sec\_diff > 1e−4 after the fix; was 0.0 before).  

\- F1b test: PASSES (operators now reach prediction scores end-to-end).  

\- Empirical: forward-pass diagnostic shows sec Δ ≈ 4.6e−3 when operators are randomised (was 0.0 pre-fix).



\### 5.2 Subordinate bug — `conv\_edge\_types` tuple/string mismatch



\*\*Problem:\*\* PyG’s `HeteroConv` internally stores edge type keys as strings (`"conc\_\_mentions\_rev\_\_sec"`), not tuples. A naive `k in hetero\_conv.convs.keys()` check using tuples silently returned False, so some edge types never participated in message passing.



\*\*Effect:\*\* The model code ran without crashing; reverse edges existed in the graph but were never used in `HeteroConv`. F1 test failed silently with sec\_diff = 0.0.



\*\*Fix:\*\* Maintain a separate `self.hetero\_conv\_edge\_types\[layer\_idx]` list of tuples in `\_\_init\_\_`, populated from `conv\_dict.keys()`, and use this list for filtering in `forward`.



\*\*Verification:\*\* F1 test passes after the fix.



\### 5.3 F2 — authority scorer in gradient path



\*\*Problem (v1):\*\* `AuthorityScorer` was instantiated when `use\_authority=True`, but the forward pass contained an identical-arms branch:



```python

if self.use\_authority and "auth" in h:

&#x20;   r2\_pri = r2\_pri\_raw

else:

&#x20;   r2\_pri = r2\_pri\_raw

```



The scorer was never called. Its parameters received zero gradient. The `use\_authority` flag changed parameter count but not the computation.



\*\*Implication for v1:\*\* The −Authority ablation effect in v1 Table 3 (Δ = +0.0076 at seed 42, p = 0.006) cannot come from the scorer; its weights were untrained. The effect must be from initialisation/parameter-count differences only.



\*\*Fix:\*\* Wire the scorer into the forward pass with real authority features (auth\_type, auth\_level, auth\_recency) extracted per r2 edge from the citing section’s dominant authority.



\*\*Files modified:\*\*



\- `src/kg/kg\_builder.py`: per-edge authority features computed and attached to forward and reverse r2 edges.  

\- `src/preprocess/authority\_extractor.py`: year extraction for recency.  

\- `src/train/trainer.py`: `edge\_attr\_dict` now passes authority features.  

\- `src/model/jusdef.py`: dead branch removed; `self.authority\_scorer` used to transform authority features into priorities.



\*\*Verification:\*\*



\- F2 test (`test\_F2\_authority\_scorer\_in\_gradient\_path`): PASSES (parameters receive non-zero gradient after backward pass).  

\- Empirical: across-graph distributions of `auth\_type`, `auth\_level`, `auth\_recency` show real variance (auth\_type 0–3 with skew; level std ≈ 0.21; recency std ≈ 0.086).



\### 5.4 Subordinate bug — in-place `scatter\_reduce\_` broke autograd



\*\*Problem:\*\* `compute\_defeat\_mask` in `dmp\_layer.py` used in-place `scatter\_reduce\_` on a tensor created with `torch.full`:



```python

group\_max = torch.full((num\_groups,), -1e9, device=...)

group\_max.scatter\_reduce\_(0, dst\_nodes, defeat\_score, reduce="amax")

priority\_gap = group\_max\[dst\_nodes] - defeat\_score - 0.001

```



PyTorch autograd does not fully track gradient through this in-place mutation. The `priority\_gap` stayed `requires\_grad=True` via the `- defeat\_score` term, but gradient through the max path was effectively dead.



\*\*Effect:\*\* Authority scorer parameters received almost no useful gradient via the defeat mask. F2 test remained failing even with the scorer wired in.



\*\*Fix:\*\* Replace with non-in-place `scatter\_reduce`:



```python

base = torch.full((num\_groups,), -1e9, device=...)

group\_max = base.scatter\_reduce(

&#x20;   0, dst\_nodes, defeat\_score, reduce="amax", include\_self=True

)

```



\*\*Verification:\*\* F2 test passes after the fix.



\### 5.5 F3a — EuroVoc ontology edges



\*\*Status:\*\* Implemented in `kg\_builder.py` but currently inactive. Graphs are constructed with edge type `('conc', 'ontology', 'conc')` but with zero edges because `label\_adj.pt` (EuroVoc adjacency) is not yet built.



\*\*Test:\*\* `test\_F3a\_concept\_ontology\_edge\_present` PASSES (verifies edge type exists). It does not check edge count.



\*\*Pending work (post-thesis):\*\* build `label\_adj.pt` from EuroVoc SKOS RDF.



\### 5.6 F3b, F3c — authority hierarchy and authority–concept edges



\*\*Status:\*\* Deferred. Marked as `pytest.mark.xfail` in the test suite.



\### 5.7 Outstanding issue — `auth\_recency` constant within graph



\*\*Observed:\*\* Within a single document (one section in current preprocessing), all r2 edges inherit the same dominant authority features. Per-document authority features are constant.



\*\*Across documents:\*\* `auth\_type`, `auth\_level`, `auth\_recency` all vary (verified across 500 test graphs).



\*\*Implication:\*\* the authority scorer receives document-level but not within-document variation. This is by design given the one-section-per-document assumption, not a bug.



\---



\## 6. Verification methodology



\### 6.1 Test suite



`tests/test\_architecture\_fixes.py` contains seven unit tests:



| Test | Verifies                                                       |

|------|-----------------------------------------------------------------|

| F1   | Randomising operators changes section embeddings               |

| F1b  | Operators reach prediction scores (end-to-end)                 |

| F2   | Authority scorer parameters receive non-zero gradient          |

| F3a  | EuroVoc ontology edge type present in graph                    |

| F3b  | Authority hierarchy edge present (xfail)                       |

| F3c  | Authority–concept edge present (xfail)                         |

| F5   | DMP defeat mask = all-ones when all operators are AFF          |



All non-xfail tests pass on v2 multi-seed graphs. The test suite must be green before any training submission.



\### 6.2 Forward-pass diagnostic



A script loads one test graph and calls `model.forward()` twice — once with original operators, once with operators randomised — and reports max element-wise differences per node type.



Pre-F1 fix:



\- sec: max |Δ| = 0.0  

\- conc: max |Δ| = 0.20  



Post-F1 fix:



\- sec: max |Δ| ≈ 4.6e−3  

\- conc: max |Δ| ≈ 0.29  

\- doc: max |Δ| = 0.0 (no incoming edges)  

\- auth: max |Δ| = 0.0  

\- label: max |Δ| = 0.0  



\### 6.3 Centralised evaluation pipeline



`scripts/eval\_all\_jusdef.py` is the single source of truth for all reported metrics. It:



\- Loads checkpoints from `outputs/checkpoints/`.  

\- Computes sigmoid probabilities from logits.  

\- Tunes threshold on validation Macro-F1.  

\- Computes Macro-F1, Micro-F1, F1(Y\_s), F1(Y\_u), F1(Y\_exc).  

\- Saves results to `outputs/logs/{tag}\_s{seed}.json`.



All thesis numbers come from this pipeline. Per-experiment evaluation scripts (source of the 2026-05-11 bug) are not used.



\---



| 7.0     | 2026-06-03 | Added §7.11–§7.17, updated chronology, resolved internal consistency notes, and closed the R-GCN control / freeze-test gaps |



\## 7. v1 → v2 empirical comparison



\### 7.1 Single-seed v2 results (seed 42, hidden\_dim 512)



| Metric    | v1         | v2 (default curriculum) | v2 (Stage‑1‑only) |

|-----------|------------|-------------------------|-------------------|

| Macro-F1  | 0.2519     | 0.1971 (−0.055)         | 0.1749 (−0.077)   |

| Micro-F1  | 0.3879     | 0.2806 (−0.107)         | 0.2357 (−0.152)   |

| F1(Y\_s)   | 0.3085     | 0.2398 (−0.069)         | 0.2049 (−0.104)   |

| F1(Y\_u)   | 0.0255     | 0.0264 (≈ noise)        | 0.0549            |

| F1(Y\_exc) | 0.3562     | 0.2761 (−0.080)         | 0.2372 (−0.119)   |

| Threshold | 0.10       | 0.08                    | 0.08              |



Result: v2 underperforms v1 on all primary metrics. The architectural fixes that are mechanically correct (F1 and F2 both green) produce empirically worse results.



\### 7.2 Stage curriculum hypothesis



\*\*Hypothesis A (tested):\*\* v2’s underperformance is due to Stage 2 ontology loss disrupting training, similar to the v1 pilot in §4.3.



\*\*Test:\*\* run seed 42 with `--stage1\_end 200` so Stage 2 never activates.



\*\*Result:\*\* Stage‑1‑only is worse than default curriculum (Macro 0.175 vs 0.197; Y\_exc 0.237 vs 0.276). Hypothesis A falsified: Stage 2 is not the source of v2’s underperformance.



\### 7.3 W\_omega weight divergence diagnostic (2026-05-29)



\*\*Hypothesis B (tested):\*\* v1’s Y\_exc gain came from operator-indexed W\_omega matrices specialising by operator; v2’s architecture disrupted this specialisation.



\*\*Test:\*\* Compare W\_omega matrices between v1 and v2 checkpoints (seed 42).



\*\*v1 W\_omega divergence (max |W\_i − W\_j|):\*\*



\- Layer 0:  

&#x20; - AFF vs NEG: 0.0882  

&#x20; - AFF vs EXC: 0.0881  

&#x20; - AFF vs OVR: 0.0883  

&#x20; - NEG vs EXC: 0.0882  

&#x20; - NEG vs OVR: 0.0883  

&#x20; - EXC vs OVR: 0.0881  

\- Layer 1: same range (≈ 0.088 uniform)



\*\*v2 W\_omega divergence:\*\*



\- Layer 0:  

&#x20; - AFF vs NEG: 0.3928  

&#x20; - AFF vs EXC: 0.6327  

&#x20; - AFF vs OVR: 0.6348  

&#x20; - NEG vs EXC: 0.4186  

&#x20; - NEG vs OVR: 0.3878  

&#x20; - EXC vs OVR: 0.3363  

\- Layer 1: ≈ 0.088 (same as v1)



\*\*W\_omega norms (layer 0):\*\*



\- v1: AFF=13.07, NEG=13.06, EXC=13.07, OVR=13.04  

\- v2: AFF=2.40, NEG=1.83, EXC=1.89, OVR=1.82  



\*\*Findings:\*\*



1\. v1’s W\_omega matrices are virtually undifferentiated across operators. Element-wise differences are uniformly ≈0.088 (≈0.7% relative to ||W|| ≈ 13). Operator indexing learned nothing operator-specific.  

2\. v1’s W\_omega matrices have ≈6× larger norm than v2’s across all operators.  

3\. v2’s W\_omega matrices at Layer 0 do specialise by operator (divergences 0.34–0.63). This is the behaviour the paper claimed operator-indexed parameterisation should produce.  

4\. v2, with correctly-specialising W\_omega, produces \*\*lower\*\* Y\_exc than v1 with undifferentiated high-norm W\_omega.



Hypothesis B is falsified. v1’s gain did not come from operator specialisation in W\_omega.



\### 7.4 Capacity-magnitude scaling test (2026-05-29)



\*\*Hypothesis C:\*\* v1's higher W\_omega weight norms (≈13) provided more expressive capacity than v2's smaller norms (≈2), contributing to v1's Y\_exc gain.



\*\*Test:\*\* Clone v2's checkpoint. Multiply every W\_omega matrix (layer 0 and layer 1) by 6.0 (the approximate ratio of v1 to v2 norms). Re-evaluate on the test set without retraining.



\*\*Result:\*\*



| Metric    | v2 unscaled | v2 with W\_omega × 6 | Δ       |

|-----------|-------------|---------------------|---------|

| Macro-F1  | 0.1971      | 0.1970              | −0.0001 |

| Micro-F1  | 0.2806      | 0.2805              | −0.0001 |

| F1(Y\_s)   | 0.2398      | 0.2397              | −0.0001 |

| F1(Y\_u)   | 0.0264      | 0.0264              | 0.0000  |

| F1(Y\_exc) | 0.2761      | 0.2759              | −0.0002 |



\*\*Hypothesis C falsified.\*\* Scaling W\_omega by 6× changes predictions by less than 0.001 on every metric. The W\_omega matrices are functionally irrelevant for predictions in the v2 forward pass.



\*\*Implication:\*\* v1's empirical advantage cannot come from the operator-indexed weight matrices, regardless of their magnitude or specialisation. The advantage must lie in some other component of v1's trained parameters or training dynamics.



\*\*Next diagnostic:\*\* full module-by-module weight-norm comparison between v1 and v2 to identify which components actually control predictions and differ most in magnitude.



\### 7.5 Falsified hypotheses summary



After §§7.1–7.4, the following candidate explanations for v1's Y\_exc advantage are falsified:



\- \*\*A:\*\* Stage 2 destabilisation (Stage‑1‑only v2 is worse, not better).  

\- \*\*B:\*\* Operator-specific W\_omega specialisation in v1 (v1's W matrices are undifferentiated).  

\- \*\*C:\*\* W\_omega capacity or weight magnitude (scaling them in v2 has no effect on predictions).



This supports a stronger methodological framing:



> We document that the JusDef framework's empirical advantage on exception-dependent labels arose from preserved LegalBERT initialisation signal in its input projection layers, \*\*not\*\* from the operator-aware architecture the paper credited. The operator-indexed weight matrices were largely undifferentiated and partially untrained. Our corrected architecture (v2) added gradient paths via reverse mention edges and an active authority scorer; these changes inadvertently caused the model to discard the LegalBERT signal at the input projection layers (norms collapsed from ≈13 to ≈0.02), eliminating the Y\_exc advantage. This finding has implications for how we should evaluate GNN-based legal NLP: input-projection magnitude is an unrecognised inductive-bias driver.



\### 7.6 Module-level weight norm comparison (2026-05-30)



\*\*Diagnostic:\*\* Compare ||W|| of every module weight between v1 seed 42 and v2 seed 42 checkpoints. Sort by absolute difference.



\*\*Top findings (illustrative):\*\*



| Module                         | v1 \\|\\|W\\|\\| | v2 \\|\\|W\\|\\| | Δ       | Interpretation                                                |

|--------------------------------|-------------:|-------------:|--------:|---------------------------------------------------------------|

| out\_proj.weight                |       23.817 |        9.711 | 14.107  | v1 outputs at ≈2.5× v2 magnitude                             |

| input\_proj.conc.weight         |       13.057 |        0.021 | 13.037  | v2 collapsed to ≈0 — LegalBERT concept signal effectively killed |

| input\_proj.sec.weight          |       11.569 |        0.281 | 11.288  | v2 severely attenuated section input                          |

| dmp\_layers.0.W\_op.{0–3}        |     ≈13 each |      ≈2 each | ≈11     | v2 trains smaller, specialised matrices                      |

| authority\_scorer.\*             |   ≈8.2, 0.58 |   0.000, 0.0 | ≈8.2    | v2 decayed scorer weights to zero                             |

| dmp\_layers.1.W\_op.{0–3}        |     ≈13 each |     ≈13 each | ≈0.000  | untrained in both; layer‑1 DMP parameters effectively static |



\*\*Findings:\*\*



1\. The largest norm differences lie in \*\*input and output projections\*\*, not in DMP.  

2\. v2’s `input\_proj.conc.weight` is essentially zero (||W|| ≈ 0.021). The LegalBERT semantic signal is destroyed at concept input.  

3\. v2’s authority scorer decayed to ||W|| ≈ 0.0. With F2 wiring it into the gradient path and weight decay 0.01, the optimiser learned to push the scorer to zero.  

4\. DMP layer‑1 `W\_op` matrices are byte-identical between v1 and v2; they receive zero gradient in both and remain at initialisation.



\*\*Revised explanation for v1's Y\_exc advantage:\*\*  



v1's Y\_exc gain came from \*\*preserved LegalBERT initialisation signal\*\* in `input\_proj.conc` and `input\_proj.sec` (norms ≈12–13). v2’s F1+F2 architectural changes added gradient paths and parameters that optimised the input projections toward zero, destroying the LegalBERT signal that was driving v1's exception-label performance.



The operator-indexed processing (W\_omega matrices) is \*\*not\*\* the source of v1's empirical advantage — those matrices are largely undifferentiated (§7.3) and irrelevant to predictions (§7.4).



The architectural advantage in v1 is therefore not “operator-aware reasoning” but \*\*preserved input embedding magnitudes\*\*. This is a hidden inductive-bias mechanism that the paper’s stated contribution does not capture.



\### 7.7 Hypothesis tree — current state



| Hypothesis                                                       | Tested in | Result      |

|------------------------------------------------------------------|-----------|-------------|

| A: Stage 2 destabilisation                                       | §7.2      | FALSIFIED   |

| B: W\_omega specialisation in v1                                  | §7.3      | FALSIFIED   |

| C: W\_omega capacity / magnitude                                  | §7.4      | FALSIFIED   |

| \*\*D: Input projection magnitude preserves LegalBERT signal\*\*     | §7.6      | \*\*SUPPORTED — leading\*\* |



Hypothesis D is the \*\*simplest explanation consistent with all observed data\*\*. It is also directly testable: scale up v2's `input\_proj.conc.weight` and `input\_proj.sec.weight` to v1-like magnitudes and re-evaluate. If Y\_exc recovers, that confirms input‑projection magnitude as the dominant mechanism.

&#x20; ### 7.8 Input projection magnitude scaling (2026-05-30)



&#x20; \*\*Hypothesis D:\*\* v1's higher input projection magnitudes

&#x20; (input\_proj.conc ||W|| = 13.057 vs v2's 0.021) preserved LegalBERT

&#x20; semantic signal at the input layer and drove the Y\_exc gain.



&#x20; \*\*Test:\*\* Clone v2's checkpoint. Scale input\_proj.conc.weight by

&#x20; 622×, input\_proj.sec.weight by 41×, and out\_proj.weight by 2.45×

&#x20; to match v1's magnitudes. Re-evaluate without retraining.



&#x20; \*\*Result:\*\*



&#x20; | Metric | v1 | v2 unscaled | v2 input-scaled |

&#x20; |---|---|---|---|

&#x20; | Macro-F1 | 0.2519 | 0.1971 | 0.1415 |

&#x20; | Micro-F1 | 0.3879 | 0.2806 | 0.1387 |

&#x20; | F1(Y\_s) | 0.3085 | 0.2398 | 0.1679 |

&#x20; | F1(Y\_u) | 0.0255 | 0.0264 | 0.0360 |

&#x20; | F1(Y\_exc) | 0.3562 | 0.2761 | 0.1790 |

&#x20; | Threshold | 0.10 | 0.08 | 0.02 |



&#x20; \*\*Hypothesis D falsified.\*\* Scaling input projections to v1 magnitudes

&#x20; worsened all primary metrics. v2's weights are co-adapted to its

&#x20; architecture; transplanting v1 magnitudes breaks the model.



&#x20; ### 7.9 Meta-finding — no single-component explanation survives



&#x20; Four hypotheses have now been systematically tested and falsified

&#x20; through controlled interventions (curriculum modification or post-hoc

&#x20; weight scaling):



&#x20; | Hypothesis | Tested by | Outcome |

&#x20; |---|---|---|

&#x20; | A. Stage 2 destabilisation | Stage-1-only retraining | falsified (worse) |

&#x20; | B. W\_omega specialisation in v1 | Divergence comparison | falsified (v1 W's undifferentiated) |

&#x20; | C. W\_omega capacity/magnitude | 6× scaling | falsified (no change) |

&#x20; | D. Input projection magnitude | 41–622× scaling | falsified (much worse) |



&#x20; \*\*Interpretation:\*\* v1's empirical advantage cannot be localised to any

&#x20; single architectural component. The weights are co-adapted under v1's

&#x20; specific gradient flow patterns. v2's F1 and F2 architectural fixes

&#x20; create different gradient dynamics that converge to a qualitatively

&#x20; different (and on overall metrics, inferior) solution. The components

&#x20; of a heterogeneous GNN are not independent contributors whose effects

&#x20; can be isolated through ablation or transplantation.



&#x20; \*\*Implication for GNN-based legal NLP methodology:\*\* Standard ablation

&#x20; studies (remove component, measure metric change) may misattribute

&#x20; empirical effects because components are jointly optimised. Forward-pass

&#x20; verification and post-hoc weight-transplantation experiments — both

&#x20; demonstrated here — should be standard practice when claiming

&#x20; architectural contributions.



&#x20; ### 7.10 Reframed thesis contribution



&#x20; In light of §7.1–§7.9, the thesis contribution shifts from

&#x20; "JusDef provides operator-aware reasoning for legal NLP" to:



&#x20; 1. \*\*Methodology contribution:\*\* Forward-pass correctness in

&#x20;    heterogeneous GNNs must be empirically verified. Standard ablation

&#x20;    methodology is insufficient because architectural components are

&#x20;    co-adapted.



&#x20; 2. \*\*Architectural contribution:\*\* F1 (reverse edge), F2 (authority

&#x20;    scorer in gradient path), and F3a (ontology edges) are necessary

&#x20;    corrections to v1's architecture that make operator information

&#x20;    actually reach predictions.



&#x20; 3. \*\*Empirical contribution:\*\* With v2's corrected architecture, the

&#x20;    model under-performs v1 on overall metrics by 5–10 F1 points. This

&#x20;    tells us that v1's empirical advantage was not from operator-aware

&#x20;    reasoning, but from an emergent property of v1's training dynamics

&#x20;    that no single-component intervention can transplant.



&#x20; 4. \*\*Future-work direction:\*\* Increasing operator density via a

&#x20;    neural detector (planned annotation work, in progress) may shift

&#x20;    v2's optimisation landscape enough to deliver real operator-aware

&#x20;    gains. v1's effective signal was a hidden inductive bias, not a

&#x20;    genuine semantic mechanism; v2 with sufficient defeat signal could

&#x20;    produce the gain the paper claimed.

\### 7.11 Note on internal consistency after §7.8



Section §7.7 presented Hypothesis D as the leading explanation before the direct scaling test was run. After §7.8, that hypothesis is no longer supported. The chronology is retained for audit transparency, but later sections supersede the provisional interpretation in §7.7.



\### 7.12 Audit check — what remained unresolved before the final updates



At the end of §7.10, two issues still remained open in the document state captured on 2026-05-29:



\- whether the v1→v2 underperformance was specific to JusDef or reflected a broader degradation in the v2 graph construction;

\- whether the tentative input-projection explanation survived a proper freeze-based test rather than only post-hoc scaling.



Those two gaps are closed in §7.13–§7.16 below.



\### 7.13 R-GCN baseline on v2 graphs (multi-seed control)



To isolate whether v2's underperformance comes from JusDef-specific components or from the graph structural changes (reverse mention edge, ontology placeholder, per-edge authority features), R-GCN was evaluated on the v2 graphs across three seeds.



| Seed | Macro-F1 | Micro-F1 | Threshold |

|---|---|---|---|

| 42 | 0.2728 | 0.4441 | 0.14 |

| 43 | 0.2671 | 0.4189 | 0.10 |

| 44 | 0.2797 | 0.4242 | 0.10 |

| \*\*Mean\*\* | \*\*0.2732 ± 0.0063\*\* | \*\*0.4291 ± 0.0136\*\* | — |



Comparison to v1 R-GCN (paper Table 2):



| Metric | v1 R-GCN | v2 R-GCN (this work) | Δ |

|---|---|---|---|

| Macro-F1 | 0.2741 ± 0.0047 | 0.2732 ± 0.0063 | −0.0009 |

| Micro-F1 | 0.4276 ± 0.0137 | 0.4291 ± 0.0136 | +0.0015 |



R-GCN performs essentially identically on v1 and v2 graphs. The differences are well below seed variance.



\*\*Conclusion:\*\* v2's graph structural changes do not break the R-GCN baseline. The v2 underperformance is therefore localised to JusDef-specific components: the Defeasible Message Passing layer, the authority scorer, and the operator routing through reverse edges.



\### 7.14 Freeze at collapsed value (misconfigured freeze)



A first freeze experiment attempted to preserve v2's collapsed `input\_proj.conc` value during training. This run verified that the parameter could be frozen, but it did not test the substantive hypothesis of interest, because it froze the already-collapsed v2 value rather than transplanting the v1 magnitude.



Result: no recovery of Macro-F1 or Y\_exc. The experiment is retained as a negative control on the mechanics of freezing, but it does not constitute a valid test of whether v1's higher input-projection norm explains its empirical advantage.



\### 7.15 Corrected freeze test (Hypothesis D, full version)



Following the misconfigured freeze in §7.14, a corrected run initialised `input\_proj.conc` at v1's weight (norm 13.057) and froze it during v2 training (seed 42, default curriculum, hidden\_dim 512).



Verification:



\- Console: `FROZE input\_proj.conc at v1 values (weight norm = 13.057)`

\- Post-training checkpoint norm: 13.057 (unchanged — freeze worked)



Final test metrics:



| Metric | v1 ref | v2 default s42 | v2 freeze-at-v1 s42 |

|---|---|---|---|

| Macro-F1 | 0.2519 | 0.1971 | 0.1717 |

| Micro-F1 | 0.3879 | 0.2806 | 0.2590 |

| F1(Y\_s) | 0.3085 | 0.2398 | 0.1989 |

| F1(Y\_u) | 0.0255 | 0.0264 | 0.0626 |

| F1(Y\_exc) | 0.3562 | 0.2761 | 0.2092 |



\*\*Hypothesis D falsified, with surprises.\*\*



1\. The corrected freeze made Macro and Y\_exc worse than default v2, not better.

2\. F1(Y\_u) more than doubled, suggesting that Y\_u and Y\_exc may prefer different input-projection regimes in the corrected architecture.

3\. The model trained, but reached a lower best validation Macro-F1 than the default v2 seed-42 run.



These observations rule out Hypothesis D in its strongest form: forcing v1-magnitude inputs fails to recover v1 performance and actively damages the metric it was meant to explain.



\### 7.16 Final diagnostic summary — five hypotheses falsified



Across §7.1–§7.15, the following hypotheses for v1's empirical advantage have been systematically tested and falsified:



| # | Hypothesis | Test | Result |

|---|---|---|---|

| A | Stage 2 destabilisation | Stage-1-only retrain (§7.2) | Falsified |

| B | W\_omega operator specialisation | Divergence diagnostic (§7.3) | Falsified |

| C | W\_omega magnitude/capacity | 6× scaling (§7.4) | Falsified |

| D' | Freeze at collapsed value | Misconfigured freeze (§7.14) | Falsified |

| D | Freeze at v1 value | Corrected freeze (§7.15) | Falsified |



A separate control experiment (§7.13) shows that the v2 graph changes do not affect R-GCN; v2's underperformance is therefore localised to JusDef-specific components. Together, these results support a methodological conclusion: v1's empirical advantage on exception-dependent labels is not localisable to any single architectural component or weight magnitude. It emerges from the co-adapted optimisation of all parameters under v1's specific gradient flow patterns, while the architecturally corrected v2 converges to a qualitatively different — and on all primary metrics, inferior — equilibrium.



\### 7.17 Implications and future work



The systematic falsification of single-component hypotheses leads to two implications.



First, \*\*methodology\*\*: ablation studies in GNN-based legal NLP should include forward-pass propagation verification, weight-magnitude inspection across all modules, and post-hoc parameter interventions (scaling, freezing) when interpreting architectural contributions. Standard ablations alone are insufficient because components are co-adapted.



Second, \*\*future work\*\*: the operator-density bottleneck remains the binding constraint. With operators rarely active, the gradient signal to operator-aware components is insufficient to distinguish them from operator-blind alternatives. The neural operator detector is the most theoretically justified remaining intervention.

\---



\## 8. Reproducibility



\### 8.1 Environments



\- Laptop (development): Windows 11, Python 3.10, conda env `jusdefv2`. `requirements.txt` pinned exactly.  

\- Ampere cluster: Ubuntu 20.04, Python 3.9, venv. `requirements\_ampere.txt` pins the core deep-learning stack; auxiliary versions resolved by pip.



\### 8.2 Hardware



\- Laptop: CPU only (development, tests).  

\- Ampere: NVIDIA A100 80GB PCIe, driver 550.163.01, CUDA 12.4.



\### 8.3 Reproducing v1 numbers



```bash

git clone https://github.com/oziofficial5/Jusdef

cd Jusdef

conda env create -f environment.yml

conda activate jusdef

bash scripts/run\_all.sh         # \~12 GPU-hours

python scripts/eval\_all\_jusdef.py

```



All numbers in v1 Tables 2–4 come from JSONs produced by `eval\_all\_jusdef.py`. Per-experiment scripts (that contained the sigmoid bug) are not used.



\### 8.4 Reproducing v2 numbers



```bash

git clone https://github.com/oziofficial5/jusdefv2

cd jusdefv2

python -m venv .venv \&\& source .venv/bin/activate

pip install -r requirements\_ampere.txt

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html



pytest tests/ -v   # F1, F1b, F2, F5 must be green; F3a green if label\_adj built

python scripts/build\_graphs.py

python scripts/run\_multiseed.sh

python scripts/eval\_all\_jusdef.py

```



\### 8.5 Deterministic settings



\- `torch.manual\_seed`, `numpy.random.seed`, and `random.seed` all set at training start.  

\- `torch.use\_deterministic\_algorithms` not enforced; some PyG ops (HeteroConv aggregation) are non-deterministic. Residual variance is absorbed by multi-seed runs.



\---



\## 9. Outstanding methodological items



\### 9.1 Inter-annotator agreement on Y\_exc



\- Y\_exc list constructed by a single annotator (first author) on 2026-04-12.  

\- Pending: the independent annotator annotates the 21 EuroVoc labels. Compute Cohen’s κ; target ≥ 0.7. Refine guidelines or Y\_exc definition if disagreement is substantial.



\### 9.2 Inter-annotator agreement on operator annotations



\- Operator annotation task for the neural detector (3000 sentences) underway.  

\- Pending: the independent annotator annotates a random 300-sentence subset. Compute Cohen’s κ; refine guidelines if κ < 0.7.



\### 9.3 Authority recency extraction



\- v2 uses regex year extraction from authority text with fallback 0.5 (sentinel for unknown).  

\- Verified: across-graph recency std ≈ 0.086, range \[0.50, 0.85]; real signal present.  

\- Limitation: year extraction is heuristic, not based on structured metadata. Ambiguous texts may yield wrong years.



\### 9.4 Capacity-matched comparison (v2 architecture)



\- Original v1 capacity-matched experiment invalidated by forward-pass blindness (§4.4).  

\- With v2’s corrected architecture, weight-scaling and freeze diagnostics have now been completed (§7.8, §7.14, §7.15).  

\- Any future capacity-matched experiment should therefore target operator density or routing, not weight magnitude alone.



\### 9.5 Multi-seed v2 confirmation



\- The original pending concern was whether seed 42 was an outlier. That concern is now partly resolved on the control side: R-GCN is stable across three seeds on v2 graphs (§7.13).  

\- Full three-seed JusDef v2 confirmation remains pending only if a final thesis table requires it, but the graph-construction explanation is no longer tenable.

\---



\## 10. Implications for the field



\### 10.1 Forward-pass propagation must be empirically verified



Many GNN-based legal NLP architectures use heterogeneous graphs with directional edge types. The v1→v2 experience shows that one-directional edges can produce \*\*silent failures\*\*: novel components train without crashing, pass unit tests of local correctness, yet never influence predictions because forward and gradient paths are blocked.



A simple two-call forward-pass test (perturb input, measure output change per node type) catches this in seconds and should be standard in heterogeneous-GNN methodology.



\### 10.2 Ablations conflate routing with semantics



The v1 ablation table (−DMP, −Authority, −DMP−Auth) implicitly assumed that removing a component removes its semantic contribution. v1’s forward-pass blindness invalidates this assumption: components contributed only via parameter-count or initialisation differences, not via their intended roles.



In v2, F1 and F2 ensure that components are architecturally active; ablations are now interpretable as originally intended.



\### 10.3 Weight magnitude as hidden inductive bias



The v1 vs v2 weight-norm comparison (||W|| ≈ 13 vs ≈ 2) shows that different architectures can converge to substantially different weight magnitudes. When this happens, “same architecture, different config” comparisons may still embed weight-magnitude differences that drive performance.



Methodologically: weight norms should be reported alongside metrics. Weight magnitude can act as an inductive bias as consequential as any architectural component.



\---



\## 11. Document version history



| Version | Date       | Changes                                              |

|---------|------------|------------------------------------------------------|

| 1.0     | 2026-05-11 | Initial draft: §3, §4 (v1 audit)                    |

| 2.0     | 2026-05-17 | Added §4.4 (forward-pass blindness discovery)       |

| 3.0     | 2026-05-21 | Added §5 (v2 architectural fixes)                   |

| 4.0     | 2026-05-25 | Added §6 (verification methodology)                 |

| 5.0     | 2026-05-27 | Added §7.1 (v1 vs v2 single-seed comparison)        |

| 6.0     | 2026-05-29 | Added §7.3 (W\_omega diagnostic) and §10 (implications) |



\---



End of methodology audit.```



However, `edge\_index\_dict` uses \*\*tuple\*\* keys like `("conc", "mentions\_rev", "sec")`, whereas `hetero\_conv.convs.keys()` can be \*\*strings\*\* internally (e.g. `"conc\_\_mentions\_rev\_\_sec"`) depending on how PyG wraps the modules. The membership test `k in conv\_edge\_types` therefore always returned `False`, silently dropping all edges from `candidate\_edges`. The reverse edge existed in the graph schema, but message passing over it never actually ran. \[pytorch-geometric.readthedocs](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html)



The fix was to \*\*store the tuple keys explicitly\*\* at construction time, instead of rediscovering them through PyG’s internal representation. In `JusDef.\_\_init\_\_`, I now build each layer’s `conv\_dict` as before, then cache its keys:



```python

self.hetero\_convs = nn.ModuleList()

self.hetero\_conv\_edge\_types = \[]

for \_ in range(num\_layers):

&#x20;   conv\_dict = {

&#x20;       ("doc", "has\_section", "sec"): SAGEConv(hidden\_dim, hidden\_dim),

&#x20;       ("label", "maps\_to", "conc"): SAGEConv(hidden\_dim, hidden\_dim),

&#x20;       ("label", "parent\_of", "label"): SAGEConv(hidden\_dim, hidden\_dim),

&#x20;       ("sec", "cites", "auth"): SAGEConv(hidden\_dim, hidden\_dim),

&#x20;       ("conc", "mentions\_rev", "sec"): SAGEConv(hidden\_dim, hidden\_dim),

&#x20;   }

&#x20;   if not use\_dmp:

&#x20;       conv\_dict\[("sec", "mentions", "conc")] = SAGEConv(hidden\_dim, hidden\_dim)



&#x20;   self.hetero\_conv\_edge\_types.append(set(conv\_dict.keys()))

&#x20;   self.hetero\_convs.append(HeteroConv(conv\_dict, aggr="sum"))

```



In `forward`, the filter uses this cached set:



```python

conv\_edge\_types = self.hetero\_conv\_edge\_types\[layer\_idx]

candidate\_edges = {

&#x20;   k: v

&#x20;   for k, v in edge\_index\_dict.items()

&#x20;   if k != r2\_key and v.size(1) > 0 and k in conv\_edge\_types

}

```



Now the membership test is tuple-vs-tuple and HeteroConv actually runs on all intended relations, including the reverse mention edge. The unit test `test\_F1\_operators\_reach\_sections` then passes: randomising r2 operators changes section embeddings (max |Δ| > 0), demonstrating that operator-aware concept updates are propagating back to sections as designed. \[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/149322011/9c3edef9-5bad-4bf8-b6fd-05f65e4c32c7/kg\_builder.py)



\### F2: Authority scorer on the gradient path



The second failure mode (F2) was more insidious: the authority scorer module existed and was even being instantiated, but its parameters received \*\*zero gradient\*\* during backprop. From a training perspective, this is equivalent to having no authority model at all.



F2 required several coordinated changes:



1\. \*\*Authority features on r2 edges (test graphs)\*\*  

&#x20;  The model expects each r2 edge to carry not only the operator and base priority, but also three authority features: `auth\_type`, `auth\_level`, and `auth\_recency`. On the laptop, these did not yet exist in the cached graphs. I introduced a patch script:



&#x20;  ```python

&#x20;  PATHS = \[

&#x20;      "data/processed/graphs/test\_graphs.pt",

&#x20;      # later extended to train/validation

&#x20;  ]

&#x20;  R2 = ("sec", "mentions", "conc")

&#x20;  R2\_REV = ("conc", "mentions\_rev", "sec")



&#x20;  for path in PATHS:

&#x20;      graphs = torch.load(path, map\_location="cpu")

&#x20;      rng = np.random.RandomState(42)



&#x20;      for g in graphs:

&#x20;          for et in (R2, R2\_REV):

&#x20;              if et not in g.edge\_types:

&#x20;                  continue

&#x20;              store = g\[et]

&#x20;              n = store.edge\_index.size(1)

&#x20;              if n == 0:

&#x20;                  store.auth\_type = torch.zeros(0, dtype=torch.long)

&#x20;                  store.auth\_level = torch.zeros(0, dtype=torch.float)

&#x20;                  store.auth\_recency = torch.zeros(0, dtype=torch.float)

&#x20;                  continue

&#x20;              store.auth\_type = torch.tensor(rng.randint(0, 6, size=n), dtype=torch.long)

&#x20;              store.auth\_level = torch.tensor(rng.uniform(0, 1, size=n), dtype=torch.float)

&#x20;              store.auth\_recency = torch.tensor(rng.uniform(0, 1, size=n), dtype=torch.float)

&#x20;  ```



&#x20;  This is explicitly a \*\*synthetic\*\* patch for test-time smoke; real features will be built in `kg\_builder.py` on Ampere. After patching, each r2 edge had the three required tensors, with shapes matching the r2 edge count. \[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/149322011/b9f19c60-fef0-488b-86bb-9bb652e26c3a/train\_jusdef.py)



2\. \*\*Plumbing authority features into the forward pass (tests + model)\*\*  

&#x20;  The test helper `\_forward` in `tests/test\_architecture\_fixes.py` initially only passed `operator` and `priority`. I updated it to opportunistically include the authority fields if present:



&#x20;  ```python

&#x20;  if r2\_key in g.edge\_types and g\[r2\_key].edge\_index.size(1) > 0:

&#x20;      r2 = g\[r2\_key]

&#x20;      attrs = {"operator": r2.operator, "priority": r2.priority}

&#x20;      for k in ("auth\_type", "auth\_level", "auth\_recency"):

&#x20;          if hasattr(r2, k):

&#x20;              attrs\[k] = getattr(r2, k)

&#x20;      edge\_attr\_dict = {r2\_key: attrs}

&#x20;  ```



&#x20;  In `JusDef.forward`, I replaced the dead branch that always set `r2\_pri = r2\_pri\_raw` with a real authority path:



&#x20;  ```python

&#x20;  r2\_ops = edge\_attr\_dict\[r2\_key]\["operator"]

&#x20;  r2\_pri\_raw = edge\_attr\_dict\[r2\_key]\["priority"]



&#x20;  if (

&#x20;      self.use\_authority

&#x20;      and r2\_key in edge\_attr\_dict

&#x20;      and "auth\_type" in edge\_attr\_dict\[r2\_key]

&#x20;      and "auth\_level" in edge\_attr\_dict\[r2\_key]

&#x20;      and "auth\_recency" in edge\_attr\_dict\[r2\_key]

&#x20;  ):

&#x20;      r2\_pri = self.authority\_scorer(

&#x20;          edge\_attr\_dict\[r2\_key]\["auth\_type"],

&#x20;          edge\_attr\_dict\[r2\_key]\["auth\_level"],

&#x20;          edge\_attr\_dict\[r2\_key]\["auth\_recency"],

&#x20;      )

&#x20;  else:

&#x20;      r2\_pri = r2\_pri\_raw



&#x20;  dmp\_out = self.dmp\_layers\[layer\_idx](

&#x20;      src\_embs, dst\_ids, r2\_ops, r2\_pri, concept\_ids, num\_conc

&#x20;  )

&#x20;  ```



&#x20;  This ensures the DMP layer sees either learned priorities (when authority features are present) or the original priorities (for backwards compatibility).



3\. \*\*Fixing autograd in the defeat mask (scatter\_reduce\_)\*\*  

&#x20;  The most subtle bug was inside `compute\_defeat\_mask` in `dmp\_layer.py`. The original implementation computed per-concept maxima using an \*\*in-place\*\* scatter:



&#x20;  ```python

&#x20;  defeat\_score = operators.float() \* 1000.0 + priorities

&#x20;  num\_groups = dst\_nodes.max().item() + 1

&#x20;  group\_max = torch.full((num\_groups,), -1e9, device=operators.device)

&#x20;  group\_max.scatter\_reduce\_(0, dst\_nodes, defeat\_score, reduce="amax")

&#x20;  priority\_gap = group\_max\[dst\_nodes] - defeat\_score - 0.001

&#x20;  ```



&#x20;  Here, `group\_max` is created without `requires\_grad`, then mutated in place by `scatter\_reduce\_`. PyTorch’s autograd does not retroactively tie that mutated tensor back to `defeat\_score`: `defeat\_score.requires\_grad` is `True`, but `group\_max.requires\_grad` stays `False`, and the gradient signal from the max operation is effectively lost. As a result, `priority\_gap` only depended on `-defeat\_score`, and for the max element in each group the two terms partially cancelled. The Straight-Through Estimator (STE) then had almost no meaningful gradient to propagate back to `priorities` and the authority scorer. \[stackoverflow](https://stackoverflow.com/questions/60949682/how-to-break-pytorch-autograd-with-in-place-ops)



&#x20;  The fix was to switch to the \*\*non-in-place\*\* scatter\_reduce API, which returns a new tensor that correctly participates in the graph:



&#x20;  ```python

&#x20;  defeat\_score = operators.float() \* 1000.0 + priorities

&#x20;  num\_groups = dst\_nodes.max().item() + 1

&#x20;  base = torch.full((num\_groups,), -1e9, device=operators.device)

&#x20;  group\_max = base.scatter\_reduce(

&#x20;      0, dst\_nodes, defeat\_score, reduce="amax", include\_self=True

&#x20;  )

&#x20;  priority\_gap = group\_max\[dst\_nodes] - defeat\_score - 0.001

&#x20;  ```



&#x20;  After this change, both `group\_max` and `priority\_gap` correctly report `requires\_grad=True`, and autograd can connect the defeat mask back to the authority priorities. \[glaringlee.github](https://glaringlee.github.io/autograd.html)



4\. \*\*Softening the defeat mask usage to preserve gradients\*\*  

&#x20;  Finally, the way the defeat mask was used inside the attention block also mattered for gradient flow. The initial pattern combined STE with hard boolean thresholding:



&#x20;  ```python

&#x20;  active\_msg = msg \* defeat\_mask.unsqueeze(-1)

&#x20;  attn\_raw = self.attn(active\_msg).squeeze(-1)

&#x20;  attn\_raw = attn\_raw.masked\_fill(defeat\_mask < 0.5, -1e9)

&#x20;  attn\_exp = torch.exp(attn\_stable)

&#x20;  attn\_exp = attn\_exp \* (defeat\_mask > 0.5).float()

&#x20;  ```



&#x20;  The STE already provides a surrogate gradient through `defeat\_mask`, but the downstream boolean masks (`< 0.5`, `> 0.5`) turn it back into a hard gate, effectively killing the continuous signal. To keep the model faithful to the defeasible semantics while retaining a usable gradient, I switched to a \*\*soft-mask\*\* design: \[discuss.pytorch](https://discuss.pytorch.org/t/how-does-applying-a-mask-to-the-output-affect-the-gradients/126520)



&#x20;  ```python

&#x20;  defeat\_mask = compute\_defeat\_mask(operators, priorities, concept\_ids, self.temperature)



&#x20;  # Soft-mask messages

&#x20;  active\_msg = msg \* defeat\_mask.unsqueeze(-1)



&#x20;  # Compute raw attention from msg (not already-masked active\_msg)

&#x20;  attn\_raw = self.attn(msg).squeeze(-1)



&#x20;  # Group-wise normalization

&#x20;  attn\_base = torch.full((num\_dst,), -1e9, device=device)

&#x20;  attn\_max = attn\_base.scatter\_reduce(

&#x20;      0, dst\_node\_ids, attn\_raw, reduce="amax", include\_self=True

&#x20;  )

&#x20;  attn\_stable = attn\_raw - attn\_max\[dst\_node\_ids]



&#x20;  # Exponentiate and softly weight by defeat\_mask

&#x20;  attn\_exp = torch.exp(attn\_stable) \* defeat\_mask

&#x20;  attn\_sum = torch.zeros(num\_dst, device=device)

&#x20;  attn\_sum.scatter\_add\_(0, dst\_node\_ids, attn\_exp)

&#x20;  attn\_sum = attn\_sum.clamp(min=1e-8)



&#x20;  attn\_weights = attn\_exp / attn\_sum\[dst\_node\_ids]

&#x20;  weighted\_msg = active\_msg \* attn\_weights.unsqueeze(-1)

&#x20;  out = torch.zeros(num\_dst, self.out\_dim, device=device)

&#x20;  out.scatter\_add\_(0, dst\_node\_ids.unsqueeze(-1).expand(-1, self.out\_dim), weighted\_msg)

&#x20;  ```



&#x20;  This keeps the STE mask in the differentiable path: defeated messages get downweighted rather than obliterated by an additional hard mask. Conceptually, this is still a defeasible aggregation, but now the gradient with respect to priorities (and therefore the authority scorer) is rich enough for learning.



With these changes, the test `test\_F2\_authority\_scorer\_in\_gradient\_path` passes: after a backward pass on a simple scalar loss, at least one authority scorer parameter has non-zero gradient. Taken together, F1 and F2 mark the end of “Phase 0”: the architecture now behaves as intended both in forward semantics (operators and authority reaching sections) and in backward semantics (gradients flowing through the defeasible machinery to the learnable parameters.) \[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/149322011/9c3edef9-5bad-4bf8-b6fd-05f65e4c32c7/kg\_builder.py)



\*\*\*





