# Ampere Runbook — JusDef v2

**Target compute**: 2 weeks on Milan A100 Ampere.
**Goal**: full reproducible pipeline from raw EUR-Lex → final results, with no resubmission needed.
**Branch**: `thesis-main` @ `3d74ed0` (Pre-Ampere fixes commit).

This runbook walks through the exact sequence of files to run on Ampere, the expected duration of each stage, and what to verify before moving on.

---

## 0. Pre-flight (on the head node, before sbatch)

```bash
# 1. Pull latest
cd ~/jusdefv2
git pull origin thesis-main

# 2. Verify the partition name and GPU type
sinfo                          # see what partitions exist
sacctmgr show qos               # see what QoS limits apply

# 3. Edit run_all_ampere.slurm to set:
#    --partition=<correct-name>
#    --gres=gpu:a100:1 (or whatever the actual GPU type string is)

# 4. Set up environment
module load python/3.10           # cluster-specific
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest                # if not in requirements

# 5. Confirm CUDA visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True NVIDIA A100-...`. If `False`, stop and fix env before submitting.

---

## 1. Submit the master job

```bash
mkdir -p logs outputs/checkpoints outputs/logs outputs/sentinels
sbatch run_all_ampere.slurm
squeue -u $USER                   # check it's queued/running
```

The slurm script executes `scripts/run_all_ampere.sh` which runs the 9 stages below. Each stage writes a sentinel to `outputs/sentinels/<n>.done`. If a stage already has its sentinel, it is skipped — so you can re-`sbatch` safely after fixing any single stage.

---

## 2. Stage-by-stage breakdown

The master script `scripts/run_all_ampere.sh` runs:

| Stage | Script | Purpose | Est. time on A100 | Sentinel |
|---|---|---|---|---|
| 0 | (inline) | Environment check (torch/cuda/pyg/transformers/sklearn) | <1 min | — |
| 1 | `scripts/preprocess_all.py` | Sections + concepts + operators (keyword) + authorities. Reads LexGLUE EUR-Lex, writes `data/processed/{train,validation,test}_processed.pkl` | **6–10 h** | `1.done` |
| 2 | `scripts/extract_embeddings.py` + `scripts/make_label_embs.py` | LegalBERT doc + section embeddings; label embeddings from EuroVoc names | **8–14 h** | `2.done` |
| 3 | `scripts/build_label_adj.py` | EuroVoc 2-digit broader-domain adjacency. Outputs `data/processed/label_adj.pt` | <5 min | `3.done` |
| 4 | `scripts/build_graphs.py` | Assembles HeteroData graphs (one per doc) with all node and edge types | **1–2 h** | `4.done` |
| 5 | `pytest tests/test_architecture_fixes.py` + `scripts/smoke_test_jusdef.py` | Verify F1/F1b/F2/F3a + 5-epoch sanity on 10 graphs | **5–10 min** | (no sentinel — always runs) |
| 6 | `scripts/train_rgcn.py` × 3 seeds | R-GCN baseline, seeds 42 43 44 | **3 × 4–6 h ≈ 12–18 h** | `6.done` |
| 7 | `scripts/train_jusdef.py --tag full` × 3 seeds | JusDef v2 main, seeds 42 43 44 | **3 × 8–12 h ≈ 24–36 h** | `7.done` |
| 8 | `scripts/train_jusdef.py` × 3 ablations × 3 seeds | `no_dmp`, `no_auth`, `no_dmp_no_auth` | **9 × 8–12 h ≈ 72–108 h** | `8.done` |
| 9 | `scripts/eval_all_jusdef.py` | Unified eval: macro, micro, seen, unseen, Yexc on all checkpoints. Threshold tuned on val, frozen for test | **2–3 h** | (overwrites each time) |

**Total estimate**: ~5–8 days on a single A100. Comfortable within your 2-week booking.

---

## 3. If you only have time for one pass

If your 2 weeks shrink, drop ablations first (Stage 8). The thesis story is built on Stages 1-7 + 9. Ablations are nice-to-have for the appendix.

To skip Stage 8: `touch outputs/sentinels/8.done` before submitting.

To run only the eval (after Stages 1-7 complete): `touch outputs/sentinels/{1,2,3,4,6,7,8}.done` then run Stage 9 manually:

```bash
python scripts/eval_all_jusdef.py 2>&1 | tee outputs/logs/eval_only.log
```

---

## 4. Monitoring the job

```bash
# Watch the slurm output stream
tail -F logs/run_all-*.out

# Watch per-stage logs
tail -F outputs/logs/stage1_preprocess.log
tail -F outputs/logs/stage7_full_s42.log

# Check sentinels (which stages are done)
ls outputs/sentinels/

# Check checkpoint progress
ls -la outputs/checkpoints/
```

Stop the job if you see:
- `RuntimeError: CUDA out of memory` (rare on A100 with hidden_dim=512, but possible at hidden=768).
- Repeated `LOAD ERROR` in eval logs (means a checkpoint was saved with mismatched config).
- Stage hangs >2× the expected duration with no progress in the log.

---

## 5. What each stage *must* output before the next stage can run

| After Stage | Required files | Verification command |
|---|---|---|
| 1 | `data/processed/{train,validation,test}_processed.pkl` | `python -c "import pickle; d=pickle.load(open('data/processed/test_processed.pkl','rb')); print(len(d), 'docs;', sum(1 for x in d for s in x['sections'] for c in s['concepts']), 'concepts')"` |
| 2 | `data/processed/embeddings/{train,validation,test}_{doc,section}_embs.pt` and `label_embs.pt` | `python -c "import torch; e=torch.load('data/processed/embeddings/test_doc_embs.pt'); print(e.shape)"` — expect `[N_test, 768]` |
| 3 | `data/processed/label_adj.pt` | `python -c "import torch; a=torch.load('data/processed/label_adj.pt'); print(a.shape, (a!=0).sum().item())"` — expect `[100,100]`, ~700 nonzeros |
| 4 | `data/processed/graphs/{train,validation,test}_graphs.pt` | `python -c "import torch; gs=torch.load('data/processed/graphs/test_graphs.pt'); print(len(gs), 'graphs;', gs[0].edge_types)"` |
| 6 | `outputs/checkpoints/best_rgcn_seed{42,43,44}.pt` + `outputs/logs/baseline_rgcn_seed{42,43,44}.json` | `cat outputs/logs/baseline_rgcn_seed42.json` |
| 7 | `outputs/checkpoints/jusdef_full_s{42,43,44}.pt` + matching JSONs | `cat outputs/logs/jusdef_full_s42.json` |
| 9 | `outputs/logs/jusdef_all_detailed.json` | Contains macro/micro/seen/unseen/exc for every config |

---

## 6. Recovering from a stage failure

The pipeline uses sentinels for resumability. If any stage fails mid-way:

1. **Identify which stage failed** from the slurm log.
2. **Inspect the stage log** at `outputs/logs/stage<N>_*.log`.
3. **Do not delete sentinels for completed stages**. Only delete the sentinel for the failed stage.
4. **Resubmit**: `sbatch run_all_ampere.slurm`. The script will skip completed stages and resume at the failed one.

Example: if Stage 7 (`jusdef_full_s43`) fails on epoch 50:

```bash
# The sentinel 7.done does NOT exist yet because the loop wasn't completed.
# But the s42 checkpoint exists. Manually re-run just s43 and s44:
python scripts/train_jusdef.py --seed 43 --tag full
python scripts/train_jusdef.py --seed 44 --tag full
touch outputs/sentinels/7.done
sbatch run_all_ampere.slurm   # picks up at Stage 8
```

---

## 7. Outputs you need after the run

After Stage 9 completes you should have:

```
outputs/logs/
├── baseline_rgcn_seed42.json     ← R-GCN seed 42 results
├── baseline_rgcn_seed43.json
├── baseline_rgcn_seed44.json
├── jusdef_full_s42.json          ← JusDef full seed 42
├── jusdef_full_s43.json
├── jusdef_full_s44.json
├── jusdef_no_dmp_s42.json        ← ablations
├── jusdef_no_dmp_s43.json
├── jusdef_no_dmp_s44.json
├── jusdef_no_auth_s42.json
├── jusdef_no_auth_s43.json
├── jusdef_no_auth_s44.json
├── jusdef_no_dmp_no_auth_s42.json
├── jusdef_no_dmp_no_auth_s43.json
├── jusdef_no_dmp_no_auth_s44.json
└── jusdef_all_detailed.json      ← unified table with macro/micro/seen/unseen/exc
```

These are the inputs for Chapter 5 (Experiments) and the bootstrap p-value computations.

**Download them all to your laptop** before the Ampere booking ends:

```bash
# from your laptop, in papers/jusdefv2
scp -r milan:~/jusdefv2/outputs/logs ./outputs/
scp -r milan:~/jusdefv2/outputs/checkpoints ./outputs/  # if you want checkpoints for failure-mode analysis
```

---

## 8. What is NOT in this run (and why)

The following are intentionally deferred from this Ampere session:

- **Neural operator detector trained on 3000-sentence annotated dataset**. This is the "decisive next experiment" per Chapter 1. It requires (a) training a sentence classifier, (b) re-running Stage 1 with the neural detector replacing `operator_detector.py`, (c) re-running Stages 2-9. Estimated additional compute: ~3-5 days. Add as Stage 8.5 if you finish ablations early.
- **F3b and F3c** (authority hierarchy and authority-concept edges). Tests are xfailed. Requires graph rebuild with new edge types. Mention in future work.
- **Capacity-matched R-GCN at hidden_dim=768**. If you have leftover budget after Stage 8, run `python scripts/train_rgcn.py --seed 42 --hidden_dim 768 --tag h768` and add to eval_all_jusdef.py configs.

---

## 9. Quick sanity reference

The pre-Ampere commit `3d74ed0` includes these critical contract fixes:

1. **Threshold tuned on val, frozen for test** (trainer.py, train_rgcn.py, eval_all_jusdef.py). You should NOT see `test_threshold` re-optimised per split in any results JSON.
2. **`data/processed/label_adj.pt` is tracked in git** (40 KB). Stage 3 regenerates it, but if the cluster's RDF source is missing, training will fail loudly rather than silently using identity.
3. **Section embedding fallback writes a WARN to stderr** in `kg_builder.py`. Watch for `[kg_builder] WARN` in Stage 4 logs — any count > 0 means some graphs have degenerate section representations.
4. **All architecture-fix tests pass on laptop**. Stage 5 runs them on the cluster as a gate before training; if any fail, fix on the cluster before continuing.

If a result JSON has `val_threshold` field present (added by the new evaluate signature), the test result used the val-tuned threshold. If it's missing, the JSON was produced by an older script and the number is suspect — rerun.

---

## 10. Summary: one command to start

```bash
ssh milan
cd ~/jusdefv2
git pull origin thesis-main
# Edit run_all_ampere.slurm partition name and confirm GPU type
sbatch run_all_ampere.slurm
squeue -u $USER
tail -F logs/run_all-*.out
```

Then come back in ~5-8 days, download `outputs/logs/`, and start writing Chapter 5.
