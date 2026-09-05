# LEDGAR operator labelling scheme

The contract-adapted restatement of the operator scheme used for the 160-sentence
cross-genre validation pass. It keeps the operator set and the priority order of
the EUR-Lex guidelines (`../operator_guidelines.md`) but adds contract cues absent
from the regulatory guidelines and resolves the polysemy of "subject to"
explicitly. Released because the agreement figure it produced is not comparable
with the EUR-Lex figure, and the reason is partly this instrument.

Ask the four questions IN ORDER. Stop at the first "yes". The earlier / stronger
operator wins. Priority: OVR > EXC > NEG > AFF.

1. **OVR** - override / takes priority over another rule.
   Cues: notwithstanding, prevails, supersedes, without prejudice to,
   subject to the foregoing.
   e.g. "Notwithstanding any other provision, the indemnity survives termination."

2. **EXC** - exception / carve-out that narrows a rule.
   Cues: except, except as, unless, save for, other than, provided that,
   so long as, to the extent that, subject to (a condition).
   e.g. "The Information is confidential, except for information already public."

3. **NEG** - denies or negates applicability.
   Cues: shall not, will not, may not, must not, no ... shall, does not apply,
   is not entitled, neither ... nor.
   e.g. "No party may assign this Agreement without consent."

4. **AFF** - default: a plain rule, obligation, permission, definition, or statement.
   e.g. "The Borrower shall repay the loan within 30 days."
        "This Agreement is governed by the laws of New York."

## Consistency rules

- No operator cue at all (recitals, definitions) -> AFF. Most sentences will be
  AFF; that is expected.
- "including without limitation", "for the avoidance of doubt" -> not operators -> AFF.
- "subject to" -> EXC if it conditions a rule; OVR if it means "this yields to
  that other section".
- NEG vs EXC: if removing the "except..." clause changes the core meaning -> EXC;
  if the sentence is fundamentally a prohibition -> NEG. Pick one interpretation
  and apply it to every row.

## Protocol

Labelling was performed in a workbook whose detector-prediction column was hidden
until the pass was complete, so the annotator worked blind. Roughly 15-20 seconds
per sentence, top to bottom, without revisiting earlier rows except for genuine
inconsistency.
