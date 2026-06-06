# JusDef Operator Annotation Guidelines

**Version 1.0**
**Project:** JusDef — Defeasible Graph Neural Networks for Legal Document Classification
**Annotator(s):** Awais Abdul Khaliq (primary), supervisor (IAA on 300-sentence subset)
**Purpose:** Label EUR-LEX sentences with one of four scope operators to train a neural operator detector, replacing the keyword-based detector that currently leaves 94.5% of mention edges in the default class.

---

## 0. Why this document exists

This is the single most important artifact for the thesis-level defensibility of the operator-detection experiment. An examiner will ask: *"How do we know your AFF/NEG/EXC/OVR decisions reflect a defensible legal interpretation rather than arbitrary judgement?"* The answer is: (a) these written guidelines, fixed **before** annotation begins, and (b) inter-annotator agreement (Cohen's κ ≥ 0.7) measured against the supervisor on a 300-sentence subset.

**The cardinal rule of annotation: write the guidelines first, then annotate.** Do not invent decision rules at the end to justify labels you have already assigned. Every edge case you encounter that is not covered here gets added to Section 4 *before* you decide it, and that decision then applies to all future instances of the same pattern.

---

## 1. The four labels

The operator set is $\Omega = \{\text{AFF}, \text{NEG}, \text{EXC}, \text{OVR}\}$. Each sentence receives exactly one label, corresponding to its **outermost** scope operator.

### AFF — Default applicability  (label = 0)

**Marker:** the sentence states that something applies, holds, or is required in the normal course, without any exception, exclusion, or override.

**Assign AFF when:** the sentence expresses an obligation, permission, definition, or factual statement that does not modify the scope of another rule.

Examples:
- "Member States shall ensure compliance with this Regulation."
- "This Directive establishes a framework for the protection of natural persons."
- "The competent authority shall publish the list referred to in paragraph 1."
- "For the purposes of this Regulation, 'controller' means the natural or legal person which determines the purposes of processing."

### NEG — Negation of scope  (label = 1)

**Marker:** the sentence states that a rule, regulation, or provision **does not apply** to some entity, class, or situation.

**Assign NEG when:** the structural meaning is the exclusion of something from the scope of a rule. The phrase typically contains "shall not apply", "does not apply", "is excluded", "shall not be subject to".

Examples:
- "This Regulation shall not apply to financial instruments."
- "This Directive does not apply to activities falling outside the scope of Union law."
- "Articles 12 to 22 shall not apply to processing for archiving purposes."
- "These provisions shall not be subject to the obligations laid down in Chapter III."

### EXC — Exception (lex specialis)  (label = 2)

**Marker:** the sentence carves out a sub-case where the default behaviour changes — a specific situation in which the rule's normal effect is suspended or altered.

**Assign EXC when:** the outermost structural move is an exception introduced by "except where", "unless", "save where", "save for", "with the exception of", "other than", "by way of derogation".

Examples:
- "Article 5 shall apply except to small enterprises."
- "The data subject shall have the right to erasure unless processing is necessary for compliance with a legal obligation."
- "Member States may, save where otherwise provided, adopt stricter measures."
- "This obligation applies with the exception of micro-enterprises employing fewer than ten persons."

### OVR — Override (lex superior / lex posterior)  (label = 3)

**Marker:** the sentence asserts that one rule prevails over, takes precedence over, or operates without prejudice to another rule.

**Assign OVR when:** the structural move is a precedence relation between provisions, introduced by "notwithstanding", "without prejudice to", "shall prevail", "irrespective of", "by way of derogation from [a specific superior instrument]".

Examples:
- "Notwithstanding Article 5, Article 7 shall prevail."
- "Without prejudice to Article 5, Article 7 applies in cases of cross-border processing."
- "This Regulation shall apply irrespective of whether the processing takes place in the Union."
- "In the event of conflict, the provisions of this Regulation shall prevail over national law."

---

## 2. Decision procedure

For each sentence, follow these steps in order:

1. **Read the sentence once, fully.** Identify the main verb and the structural meaning — what is the sentence *doing* to the applicability of a rule?

2. **Identify the outermost scope operator, if any.** Ask: does the sentence exclude something (NEG), carve out an exception (EXC), assert precedence (OVR), or simply state a rule (AFF)?

3. **If multiple operators are present, label the outermost.** The outermost operator is the one that governs the overall structural meaning of the sentence (see Section 3 on nesting).

4. **Default to AFF when no clear scope-modifying signal is present.** A sentence that merely states an obligation, definition, or fact is AFF — even if it contains modal verbs (may, shall, must) or temporal conditions.

5. **If genuinely unsure, mark confidence = low.** Do not skip unless the sentence is unintelligible or not a legal-norm sentence at all (e.g., a fragment, a heading, a citation list).

---

## 3. The nesting rule (critical)

Legal sentences frequently embed one operator inside another. The annotation rule is: **label the outermost scope operator — the one that carries the dominant structural meaning of the sentence.**

The canonical case:

> "This Regulation shall not apply, except where the Member State opts in."

This sentence contains both a NEG ("shall not apply") and an EXC ("except where"). **Label it NEG**, because the dominant structural move is the exclusion; the exception is nested *inside* and modifies that exclusion. The "shall not apply" is what determines the sentence's primary effect on scope.

Contrast with:

> "Article 5 shall apply except to small enterprises."

Here the outermost move is the exception — Article 5 applies (AFF-like default) but with a carve-out. There is no negation of the whole. **Label it EXC.**

The distinction: in the first example, the rule's default is *non-application* with a re-inclusion; in the second, the rule's default is *application* with a carve-out. The outermost operator is whichever governs the sentence's overall claim.

**The composition algebra in the paper handles what happens to the nested operators downstream — the annotator's only job is to identify the outermost one.**

---

## 4. Edge cases (decided in advance — do not deviate)

| Sentence pattern | Label | Reasoning |
|---|---|---|
| "Member States shall ensure compliance with this Regulation." | AFF | Default applicability, no scope modification |
| "This Regulation shall not apply to financial instruments." | NEG | Whole-class exclusion |
| "This Regulation does not apply, except where the Member State opts in." | NEG | NEG dominates; EXC is nested inside |
| "Article 5 shall apply except to small enterprises." | EXC | Outermost move is the exception |
| "Notwithstanding Article 5, Article 7 shall prevail." | OVR | Override of one provision by another |
| "Without prejudice to Article 5, Article 7 applies in cases X." | OVR | Override expressing precedence |
| "Member States may adopt stricter measures." | AFF | Deontic modality (may) — normal regulatory language, not a scope operator |
| "The controller shall implement appropriate measures." | AFF | Deontic (shall) — obligation, not scope modification |
| "This Regulation shall apply from 1 January 2024." | AFF | Temporal scope — timing, not an exception |
| "See Article 5." / "as referred to in Article 12" | AFF | Cross-reference, not a scope modifier |
| "There is no obligation to notify in such cases." | AFF | Statement of fact (negative phrasing), not a scope negation of a rule |
| "This Article is without prejudice to national criminal law." | OVR | Precedence relation (without prejudice to) |
| "By way of derogation from Article 6, processing shall be lawful where…" | OVR | Derogation *from a specific named provision* = override |
| "By way of derogation, the time limit may be extended." | EXC | Derogation *without naming a superior instrument* = exception/carve-out |

### Explicit non-operators (always AFF)

These are **not** in the JusDef operator vocabulary and the paper's §3 explicitly notes them as part of the ~15% of constructs the four-operator scheme does not cover. **Label them AFF** and let the framework treat them as default applicability:

- **Deontic modalities:** "may", "shall", "must", "should" — these express obligation/permission, not scope modification.
- **Temporal scope:** "shall apply from", "until", "during the transitional period", "for a period of X years".
- **Cross-references:** "see Article X", "referred to in", "pursuant to", "in accordance with".
- **Negative factual statements** that do not negate a rule's scope: "there is no requirement to…", "no fee shall be charged" (this is a substantive rule, not a scope negation).

### The "derogation" disambiguation (most contested case)

"Derogation" is ambiguous and appears frequently in EU law. Resolve it as follows:

- **"By way of derogation from [specific Article/Regulation], …"** → **OVR.** It names a superior instrument that is being overridden.
- **"By way of derogation, …"** (no named instrument, just a carve-out) → **EXC.** It is functioning as an exception to the general rule.

When in doubt on a derogation, check whether a specific provision is named. Named = OVR. Unnamed = EXC.

---

## 5. Examples of disagreement, resolved in advance

These are the patterns most likely to cause annotator disagreement. They are resolved here so that the primary annotator and the supervisor apply the same rule:

1. **"May" vs. "shall"** — both AFF. Deontic strength does not change the operator; neither modifies scope.
2. **Temporal scopes** — AFF. Timing is not an exception even though it limits when a rule applies.
3. **Nested scopes** — outermost operator wins (Section 3).
4. **Negative phrasing without scope negation** — "There is no obligation to…" is AFF (a substantive statement), whereas "This Regulation shall not apply to…" is NEG (a scope negation). The test: is the sentence negating the *applicability of a rule* (NEG) or stating a *substantive fact/rule* (AFF)?
5. **Conditional applicability** — "This applies where X" is AFF (it is a positive scope condition, not an exception). "This applies except where X" is EXC. The presence of "except/unless/save" is the trigger.
6. **Multiple sentences merged by preprocessing** — if a single annotation item contains two clauses with different operators, label by the operator of the main/governing clause. If they are genuinely co-equal, label by the first one and mark confidence = low.

---

## 6. What you produce per sentence

Each annotation is one JSONL line:

```json
{
  "text": "This Regulation shall not apply to financial instruments...",
  "label": 1,
  "label_name": "NEG",
  "confidence": "high",
  "heuristic": "likely_NEG",
  "annotator": "Awais",
  "timestamp": "2026-05-26T15:23:01"
}
```

- `label`: integer 0=AFF, 1=NEG, 2=EXC, 3=OVR
- `confidence`: "high" or "low" — low-confidence items can be excluded from the training set later if needed
- `heuristic`: the pre-classification bucket from the candidate generator (used only for stratification, **never** as a hint — see Section 8)

---

## 7. Quality control during annotation

**After your first 100 annotations, stop and calibrate:**

1. Re-read this guidelines document end to end.
2. Count your labels per class. If any defeat-relevant class (NEG/EXC/OVR) has fewer than ~10 examples, the candidate stratification may need adjustment.
3. If you are skipping more than 20% of sentences, the guidelines need refinement — most likely Section 4 needs another edge case.
4. Identify any recurring pattern you found ambiguous and add it to Section 4 with a fixed decision.

**This calibration step is where most annotation projects fail.** Inconsistent early annotations poison the training set. Spend the 30 minutes.

**Commit every 100 annotations** to git so no work is lost:

```bash
git add data/annotations/operator_labels.jsonl
git commit -m "annotation: 100 more sentences (total NNNN)"
git push
```

---

## 8. Independence and leakage prevention (critical for thesis integrity)

You are annotating **individual sentences in isolation**, not whole documents. To keep the operator-detection dataset independent of the downstream classification task, **do not** consult:

- Training-set EuroVoc label distributions
- Document-level statistics
- Existing model predictions
- Which concepts are rare or frequent

If you let reasoning like *"this concept is rare, so this sentence is probably a NEG"* enter your decisions, you induce a leak between the $\mathcal{Y}_{\mathrm{exc}}$ subset construction and the operator labels. That would make the neural detector's apparent success circular and would not survive examiner scrutiny.

**Treat each sentence purely on its own linguistic and structural merits.** The `heuristic` field is present only so the candidate set was stratified to include enough rare operators; it must never guide your label. In fact, the most valuable annotations are the ones where you *disagree* with the heuristic — those are exactly the cases the keyword detector gets wrong, and they are why the neural detector is worth building.

---

## 9. Inter-annotator agreement protocol

After ~1000 annotations:

1. Select 300 random sentences you have already labelled.
2. Strip your labels from a copy; keep only the sentence text.
3. Send the stripped copy plus **this guidelines document** to the supervisor.
4. The supervisor annotates independently, applying these guidelines.
5. Compute Cohen's κ:

```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(yours, supervisor)
print(f"Cohen's kappa: {kappa:.3f}")
```

**Target: κ ≥ 0.7.** If lower:
- Build a confusion matrix of your labels vs. the supervisor's.
- Identify which label pair is most often confused (likely NEG↔EXC or EXC↔OVR).
- Refine the relevant Section 4 / Section 5 entry.
- Optionally drop the disputed examples from the training set and note this in the methodology appendix.

The κ value and the confusion analysis go directly into the thesis appendix on dataset construction. This is the step that makes the dataset defensible.

---

## 10. Version history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-26 | Initial guidelines, fixed before annotation begins |

*Any change to the decision rules after annotation has started must be logged here, with the date and the number of sentences re-checked for consistency with the new rule.*
