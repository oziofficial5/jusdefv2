"""Y_exc evaluation of the v1 checkpoints produced by run_thesis_gaps.sh PART 2.

Why this exists
---------------
PART 2 measured v1 at 0.1717 overall test macro-F1 against R-GCN's 0.2731, which
dissolves Chapter 6's "anomaly" framing. But the v1 workshop paper's actual
empirical claim was never an overall one: it reported +6.0 macro-F1 over R-GCN on
the exception-dependent subset Y_exc (Ch. 5, sec:v1-empirical). PART 2 therefore
removes the anomaly's premise without testing the workshop claim head-on.

This does test it. Inference only over the released test split, using the
checkpoints PART 2 already wrote -- no retraining.

The v1 checkpoints were trained under --ablate all, which empties two relations
and swaps the label adjacency to identity. Parameter shapes are unchanged, so the
state dict loads into a normally-constructed JusDef; what must be reproduced at
eval time is the *graph* side of the ablation. That is done below, mirroring
scripts/train_jusdef.py lines 90-120 exactly.

Usage:
    python scripts/eval_v1_yexc.py
    python scripts/eval_v1_yexc.py --tags v1_corrected_s42 v1_corrected_s43
"""
import os
import sys
import json
import glob
import argparse
import warnings

warnings.filterwarnings("ignore")
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import f1_score
from src.model.baselines import tune_threshold
from src.model.jusdef import JusDef
from src.train.trainer import forward_one_graph

REV = ("conc", "mentions_rev", "sec")
ONT = ("conc", "ontology", "conc")


def drop_relation(graphs, rel):
    dropped = touched = 0
    for g in graphs:
        if rel in g.edge_types:
            dropped += int(g[rel].edge_index.size(1))
            g[rel].edge_index = torch.zeros((2, 0), dtype=torch.long)
            touched += 1
    return touched, dropped


def logits_for(model, graphs, note=""):
    out, tgt = [], []
    with torch.no_grad():
        for i, g in enumerate(graphs):
            if note and i % 1000 == 0:
                print("    %s %d/%d" % (note, i, len(graphs)), flush=True)
            scores, _, _ = forward_one_graph(model, g, "cpu")
            out.append(scores.cpu().view(-1))
            tgt.append(g.y.cpu())
    L = torch.stack(out).numpy()
    T = torch.stack(tgt).numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(L, -40, 40))), T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None,
                    help="checkpoint tags; default = every v1_corrected* found")
    ap.add_argument("--graph_dir", default="data/processed/graphs")
    ap.add_argument("--out", default="outputs/logs/v1_yexc.json")
    args = ap.parse_args()

    exc = json.load(open("data/annotations/exception_labels.json"))
    exc_idx = exc["exception_override_labels"]
    print("Y_exc: %d labels -> %s" % (len(exc_idx), exc_idx), flush=True)

    if args.tags:
        tags = args.tags
    else:
        tags = sorted(os.path.basename(p)[len("jusdef_"):-len(".pt")]
                      for p in glob.glob("outputs/checkpoints/jusdef_v1_corrected*_s*.pt"))
    if not tags:
        print("\nNo v1 checkpoints found under outputs/checkpoints/.")
        print("PART 2 wrote them as jusdef_v1_corrected<SFX>_s<seed>.pt --")
        print("if they were cleaned up, this evaluation needs PART 2 re-run.")
        return 1
    print("checkpoints: %s\n" % ", ".join(tags), flush=True)

    print("loading graphs from %s ..." % args.graph_dir, flush=True)
    val_g = torch.load("%s/validation_graphs.pt" % args.graph_dir, map_location="cpu")
    test_g = torch.load("%s/test_graphs.pt" % args.graph_dir, map_location="cpu")

    # Reproduce the --ablate all graph edits. Without this the v1 checkpoints
    # would be scored on edges they were never trained to use.
    for name, gs in (("val", val_g), ("test", test_g)):
        for rel, what in ((REV, "reverse_edges"), (ONT, "ontology_edges")):
            t, d = drop_relation(gs, rel)
            print("  %s/%s: emptied on %d graphs (%d edges removed)" % (what, name, t, d),
                  flush=True)

    results = {}
    for tag in tags:
        ckpt = "outputs/checkpoints/jusdef_%s.pt" % tag
        if not os.path.exists(ckpt):
            print("SKIP %s: no checkpoint at %s" % (tag, ckpt), flush=True)
            continue
        print("\n=== %s ===" % tag, flush=True)
        sd = torch.load(ckpt, map_location="cpu")
        adj_keys = [k for k in sd if "adj" in k.lower()]
        if adj_keys:
            print("  note: state dict carries %s -- the identity adjacency used at"
                  " training time travels with the checkpoint." % adj_keys, flush=True)
        else:
            print("  note: no adjacency in the state dict; if the model consumes"
                  " label_adj at forward time, verify it is the identity here.", flush=True)
        model = JusDef(in_dim=768, hidden_dim=512, num_layers=2,
                       use_dmp=True, use_authority=True)
        try:
            model.load_state_dict(sd)
        except Exception as e:
            print("  LOAD ERROR: %s" % e, flush=True)
            continue
        model.eval()

        vp, vt = logits_for(model, val_g)
        thr, _ = tune_threshold(vp, vt)
        print("  val-tuned threshold: %.4f" % thr, flush=True)

        tp, tt = logits_for(model, test_g, note="test")
        pred = (tp >= thr).astype(int)

        macro = float(f1_score(tt, pred, average="macro", zero_division=0))
        yexc = float(f1_score(tt[:, exc_idx], pred[:, exc_idx],
                              average="macro", zero_division=0))
        results[tag] = {"threshold": thr, "test_macro_f1": macro, "test_yexc_macro_f1": yexc}
        print("  overall macro-F1 : %.4f" % macro, flush=True)
        print("  Y_exc  macro-F1 : %.4f" % yexc, flush=True)

    if not results:
        print("\nNothing evaluated.")
        return 1

    os.makedirs("outputs/logs", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)

    ys = [r["test_yexc_macro_f1"] for r in results.values()]
    ms = [r["test_macro_f1"] for r in results.values()]
    mean = lambda v: sum(v) / len(v)
    print("\n" + "=" * 66)
    print("  v1 across %d seeds: overall %.4f | Y_exc %.4f" % (len(ys), mean(ms), mean(ys)))
    print("""
  READING FOR CHAPTER 6 / CHAPTER 5:
  The v1 workshop paper claimed +6.0 macro-F1 over R-GCN on Y_exc. To settle
  that claim, compare the Y_exc figure above against R-GCN's Y_exc on the same
  protocol -- run this script's logic over the R-GCN checkpoints, or read it
  from scripts/eval_rgcn_detailed.py, which already computes Y_exc.

    v1 Y_exc clearly above R-GCN's -> the workshop claim survives the corrected
        protocol; it was always a Y_exc claim, and Ch. 5 sec:v1-empirical should
        say the overall shortfall and the Y_exc gain coexist.
    v1 Y_exc at or below R-GCN's   -> the workshop claim does not reproduce, and
        Ch. 5 should say so plainly. This is the stronger result for the thesis,
        because it makes the negative EUR-Lex finding uniform across metrics.""")
    print("  written to %s" % args.out)
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
