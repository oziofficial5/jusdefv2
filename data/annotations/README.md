# Operator-annotated legal corpus and validation data

Resources for the four-operator defeasibility algebra
Ω = {AFF, NEG, EXC, OVR} used by the JusDef architectures. AFF asserts default
applicability; NEG denies applicability at rule level; EXC carves a sub-case out
of an otherwise applicable rule; OVR asserts precedence over another provision.

**This is a silver corpus.** The 3,000 labels were produced by a language model
and then *measured* against independent human annotation — not corrected by it.
No adjudication pass was run and no consensus label was substituted for a
disputed one, so every label here is the labeller's own. Use the agreement
figures below as the reliability bound, not the labels' apparent authority.

## Files

| File | Rows | What it is |
|---|---|---|
| `operator_labels_3000.jsonl` | 3,000 | The corpus. EUR-Lex sentences with AI-assigned operator labels. |
| `operator_guidelines.md` | — | The 252-line annotation guidelines, drafted before labelling began. |
| `exception_labels.json` | 21 | The Y_exc subset: EuroVoc labels judged to depend on exception/override structure. |
| `eurovoc_label_names.json` | 100 | Index → EuroVoc concept name for the LexGLUE EUR-Lex label vector. |
| `iaa/self_review_100.jsonl` | 100 | Stage 1. Author's own blind re-annotation. |
| `iaa/Annotation_300_sent.jsonl` | 300 | Stage 2, as issued: text only, no labels. |
| `iaa/Annotation_300_paired.jsonl` | 300 | Stage 2, resolved: AI label beside the independent annotator's. |
| `ledgar_cross_genre/ledgar_160_paired.jsonl` | 160 | Stage 3. Detector vs. blind human annotation on contract text. |
| `ledgar_cross_genre/ledgar_labelling_scheme.md` | — | The contract-adapted scheme used for that pass. |

`id` is a stable 12-hex-character digest of the sentence text, so the same
sentence carries the same id in every file containing it.

## Agreement

| Stage | n | Raw | Cohen's κ | 95% CI |
|---|---|---|---|---|
| Self-review (EUR-Lex) | 100 | 84.0% | 0.787 | [0.70, 0.88] |
| Independent peer (EUR-Lex) | 300 | 82.7% | 0.769 | [0.71, 0.83] |
| Independent peer (LEDGAR, cross-genre) | 160 | 90.0% | 0.867 | [0.81, 0.93] |

All three recompute from the files in this directory.

**The two κ values are not on a common scale.** The EUR-Lex sample is a random
draw; the LEDGAR sample is stratified by detector prediction (40 per class) and
so over-represents the rare non-AFF classes. Chance-corrected agreement is
sensitive to class prevalence. No ordering between 0.769 and 0.867 is claimed,
and none should be inferred. The two passes also used different instruments —
`operator_guidelines.md` for EUR-Lex, `ledgar_labelling_scheme.md` for LEDGAR.

## Known limitations

- **Two annotators, not three.** Enough for a chance-corrected statistic, not
  enough for a multi-rater coefficient or an estimate of annotator variance.
- **One lexical trigger dominates the disagreement.** Every sentence containing
  *unless* in the 300-sentence sample was labelled differently by the two
  annotators — 21 sentences, 40% of all disagreement. Excluding them raises
  agreement on the remainder to κ = 0.851. The trigger is documented rather than
  removed; downstream users can re-label that subset under their own convention.
- **No free-text rationale.** The labelling prompt returns a label and a
  confidence grade, nothing more, and the batch variant explicitly suppresses
  explanations. Audit is possible at the level of *which* labels departed from
  the keyword candidate and how confidently they were assigned, not *why* any
  individual label was chosen.
- **Confidence is binary**, `high` or `low` — 2,598 high (86.6%) and 402 low
  (13.4%). There is no medium grade.
- **`keyword_candidate` is an input, not a prediction.** It carries the class
  proposed by the rule-based detector used for the stratified sampling, passed
  to the labeller and copied through unchanged. It records the sampling stratum,
  not the model's reasoning. Of 3,000 sentences the labeller moved 208 (6.9%)
  off that candidate, and every move was toward AFF: none of the 750
  keyword-AFF sentences was relabelled non-AFF.

## Provenance

- **Sentences**: EUR-Lex training split of LexGLUE (`coastalcph/lex_glue`,
  config `eurlex`); LEDGAR test split for the cross-genre set.
- **Labelling**: Claude Sonnet 5 (`claude-sonnet-5`), 2 June 2026, single
  batched run, one sample per sentence. Decoding temperature was not recorded.
- **Independent annotation**: Waseeqa Ghazanfer (Air University, Islamabad,
  Pakistan; ORCID 0000-0003-3948-6602), blind, without sight of the candidate
  labels. She was not involved in designing the operator scheme, in the
  AI-assisted labelling, or in the analysis.
- **Y_exc subset**: A. A. Khaliq, 12 April 2026, by inspection of EuroVoc label
  semantics — a judgement about how a topic area is drafted, not about the
  corpus, so it was not fitted to any evaluation split. Single annotator, no
  agreement figure: treat it as a protocol proposal rather than a validated
  instrument.

## Licence and reuse

The sentences derive from EUR-Lex, reusable under the European Commission's
reuse decision with source acknowledgement. LEDGAR text is governed by the
LexGLUE terms; the cross-genre file is distributed as labels keyed to sentence
text drawn from that public dataset. ECtHR-derived material is deliberately
**not** released in any sentence-level form — those judgments carry the personal
data of identifiable litigants, and the aggregate density figures reported in
the accompanying work serve every research purpose a sentence-level derivative
would.

If you use these resources, cite the accompanying thesis and acknowledge
EUR-Lex and LexGLUE as sources.
