"""Why is EUR-Lex 0.71% per mention edge but 4.40% per sentence?

The detector labels SENTENCES. Each mention edge then inherits the label of the
sentence its concept falls in. So the edge population weights every sentence by
how many concept mentions it contains. If non-AFF sentences carry fewer concept
mentions than AFF sentences, the edge-weighted rate is lower than the sentence
rate, and the architecture's conditioning population systematically under-weights
exactly the sentences the operator algebra is about.

This tests that directly, and settles a second problem on the way: Ch.3
Table 3.x reports the EUR-Lex test population as "~80,000" mention edges, which
over 5,000 documents implies 16 edges per document. A ten-document debug sample
gives a median of 120 and a minimum of 67, so the quoted total looks wrong by
one to two orders of magnitude.

Run on Ampere, where the full processed split exists:
    python scripts/check_edge_vs_sentence_density.py
"""
import os
import sys
import pickle
import re
import collections
import statistics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SENT = re.compile(r"(?<=[.;])\s+")
NONAFF = {"NEG", "EXC", "OVR"}


def governing_index(sentences, start):
    """Which segment does this character offset fall in? Mirrors
    src/preprocess/operator_detector.detect_operators_in_section."""
    cc = 0
    for i, s in enumerate(sentences):
        cc += len(s) + 1
        if cc >= start:
            return i
    return max(0, len(sentences) - 1)


def main():
    path = "data/processed/test_processed.pkl"
    if not os.path.isfile(path):
        print("missing %s" % path)
        return 1
    docs = pickle.load(open(path, "rb"))
    print("documents: %d" % len(docs))
    if len(docs) < 100:
        print("\n*** This looks like the --debug build (10 documents). The counts")
        print("*** below are indicative only. Run against the full split.\n")

    per_doc, edge_ops = [], collections.Counter()
    concepts_by_op = collections.defaultdict(list)
    sent_ops = collections.Counter()
    truncated = 0

    for d in docs:
        n = 0
        for s in d.get("sections", []):
            text = s.get("text") or ""
            cs = s.get("concepts") or []
            n += len(cs)
            for c in cs:
                edge_ops[(c.get("operator") or "AFF").upper()] += 1
            if not text.strip():
                continue
            if len(text) <= 501:
                truncated += 1
            sents = SENT.split(text)
            bysent = collections.defaultdict(list)
            for c in cs:
                idx = governing_index(sents, int(c.get("span_start") or 0))
                bysent[idx].append((c.get("operator") or "AFF").upper())
            for idx, ops in bysent.items():
                op = collections.Counter(ops).most_common(1)[0][0]
                sent_ops[op] += 1
                concepts_by_op[op].append(len(ops))
        per_doc.append(n)

    tot_edges = sum(per_doc)
    print("\n=== MENTION EDGES ===")
    print("  total                    : %d" % tot_edges)
    print("  per document  mean %.1f  median %.1f  min %d  max %d"
          % (statistics.mean(per_doc), statistics.median(per_doc), min(per_doc), max(per_doc)))
    na = sum(v for k, v in edge_ops.items() if k in NONAFF)
    print("  operator distribution    : %s" % dict(edge_ops))
    if tot_edges:
        print("  non-AFF per EDGE         : %d / %d = %.4f%%" % (na, tot_edges, 100.0 * na / tot_edges))
    print("  Ch.3 Table 3.x quotes ~80,000 for this population.")
    if len(docs) >= 100:
        print("  VERDICT: %s" % ("consistent" if 60000 <= tot_edges <= 100000
                                 else "NOT consistent -- Table 3.x needs correcting"))

    print("\n=== SENTENCES (as the deployed segmenter cuts them) ===")
    tot_sent = sum(sent_ops.values())
    nas = sum(v for k, v in sent_ops.items() if k in NONAFF)
    print("  sentences carrying >=1 concept : %d" % tot_sent)
    print("  operator distribution          : %s" % dict(sent_ops))
    if tot_sent:
        print("  non-AFF per SENTENCE           : %d / %d = %.4f%%" % (nas, tot_sent, 100.0 * nas / tot_sent))

    print("\n=== THE MECHANISM: concept mentions per sentence, by operator ===")
    for op in ("AFF", "NEG", "EXC", "OVR"):
        v = concepts_by_op.get(op)
        if not v:
            continue
        print("  %-4s  n=%-7d mean %.2f  median %.1f  concepts/sentence"
              % (op, len(v), statistics.mean(v), statistics.median(v)))
    aff = concepts_by_op.get("AFF") or [1]
    non = [x for op in NONAFF for x in (concepts_by_op.get(op) or [])] or [1]
    ratio = statistics.mean(aff) / statistics.mean(non)
    print("\n  AFF sentences carry %.2fx as many concept mentions as non-AFF ones." % ratio)
    print("""
  READING:
    ratio well above 1  -> confirmed. The edge population weights sentences by
        concept count, so it under-samples defeasible sentences. The 0.71% edge
        figure and the 4.40% sentence figure are then both correct and their gap
        is explained, and Ch.3 should say that the architecture conditions on a
        population that structurally under-represents the operators it is built
        for. That is a sharper reason for the EUR-Lex null than low density.
    ratio near 1        -> refuted. The gap has another cause, and the 4.40%
        per-sentence measurement should be re-derived before it is relied on.""")
    if truncated:
        print("\n  NOTE: %d sections have text of exactly 500 characters; this is the" % truncated)
        print("  truncated debug copy and the sentence-side numbers are not meaningful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
