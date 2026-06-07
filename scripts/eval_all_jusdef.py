"""Re-evaluate all JusDef checkpoints with proper sigmoid + threshold pipeline."""
import os, sys, json, torch, numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import f1_score
from src.model.baselines import tune_threshold
from src.model.jusdef import JusDef
from src.train.trainer import forward_one_graph

print('Loading test data...', flush=True)
exc_data = json.load(open('data/annotations/exception_labels.json'))
exc_idx = exc_data['exception_override_labels']
seen = list(range(80))
unseen = list(range(80, 100))

# Keyword-operator graphs (default split)
test_graphs_kw = torch.load('data/processed/graphs/test_graphs.pt', map_location='cpu')
val_graphs_kw = torch.load('data/processed/graphs/validation_graphs.pt', map_location='cpu')
print(f'Keyword graphs — Test: {len(test_graphs_kw)}  Val: {len(val_graphs_kw)}', flush=True)

# Neural-operator graphs (if Stage 7.5 has run)
neural_graph_dir = 'data/processed_neural/graphs'
test_graphs_neural = None
val_graphs_neural = None
if os.path.exists(f'{neural_graph_dir}/test_graphs.pt'):
    test_graphs_neural = torch.load(f'{neural_graph_dir}/test_graphs.pt', map_location='cpu')
    val_graphs_neural = torch.load(f'{neural_graph_dir}/validation_graphs.pt', map_location='cpu')
    print(f'Neural graphs  — Test: {len(test_graphs_neural)}  Val: {len(val_graphs_neural)}', flush=True)
else:
    print('Neural graphs not found at', neural_graph_dir, '(Stage 7.5 not run yet)', flush=True)

# (tag, hidden_dim, use_dmp, use_authority, graph_source)
# graph_source: "kw" = keyword operators, "neural" = neural detector operators
configs = [
    # Main JusDef on keyword operators
    ('full_s42',            512, True,  True,  'kw'),
    ('full_s43',            512, True,  True,  'kw'),
    ('full_s44',            512, True,  True,  'kw'),
    # Ablations on keyword operators
    ('no_dmp_s42',          512, False, True,  'kw'),
    ('no_dmp_s43',          512, False, True,  'kw'),
    ('no_dmp_s44',          512, False, True,  'kw'),
    ('no_auth_s42',         512, True,  False, 'kw'),
    ('no_auth_s43',         512, True,  False, 'kw'),
    ('no_auth_s44',         512, True,  False, 'kw'),
    ('no_dmp_no_auth_s42',  512, False, False, 'kw'),
    ('no_dmp_no_auth_s43',  512, False, False, 'kw'),
    ('no_dmp_no_auth_s44',  512, False, False, 'kw'),
    # JusDef on neural-detector operators (Stage 7.5)
    ('full_neural_s42',     512, True,  True,  'neural'),
    ('full_neural_s43',     512, True,  True,  'neural'),
    ('full_neural_s44',     512, True,  True,  'neural'),
    # Historical configs (may exist from earlier runs)
    ('full_v2_s42',         512, True,  True,  'kw'),
    ('h768_lowdef_s42',     768, True,  True,  'kw'),
    ('h768_lowdef_s43',     768, True,  True,  'kw'),
]

results = {}
for tag, hd, dmp, auth, src in configs:
    ckpt = f'outputs/checkpoints/jusdef_{tag}.pt'
    if not os.path.exists(ckpt):
        print(f'SKIP {tag}: no checkpoint', flush=True)
        continue

    # Select graphs: keyword or neural
    if src == 'neural':
        if test_graphs_neural is None:
            print(f'SKIP {tag}: neural graphs missing', flush=True)
            continue
        val_graphs = val_graphs_neural
        test_graphs = test_graphs_neural
    else:
        val_graphs = val_graphs_kw
        test_graphs = test_graphs_kw

    print(f'\n{tag} (h={hd}, dmp={dmp}, auth={auth}, graphs={src})', flush=True)
    try:
        model = JusDef(in_dim=768, hidden_dim=hd, num_layers=2,
                       use_dmp=dmp, use_authority=auth)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        model.eval()
    except Exception as e:
        print(f'  LOAD ERROR: {e}', flush=True)
        continue

    # Tune threshold on validation set
    val_logits, val_targets = [], []
    with torch.no_grad():
        for g in val_graphs:
            scores, _, _ = forward_one_graph(model, g, 'cpu')
            val_logits.append(scores.cpu().view(-1))
            val_targets.append(g.y.cpu())
    val_logits = torch.stack(val_logits).numpy()
    val_targets = torch.stack(val_targets).numpy()
    val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -40, 40)))
    best_t, _ = tune_threshold(val_probs, val_targets)
    print(f'  val-tuned threshold: {best_t:.4f}', flush=True)

    # Apply frozen threshold to test
    all_logits, all_targets = [], []
    with torch.no_grad():
        for i, g in enumerate(test_graphs):
            if i % 1000 == 0:
                print(f'  progress: {i}/{len(test_graphs)}', flush=True)
            scores, _, _ = forward_one_graph(model, g, 'cpu')
            all_logits.append(scores.cpu().view(-1))
            all_targets.append(g.y.cpu())

    logits = torch.stack(all_logits).numpy()
    targets = torch.stack(all_targets).numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    preds = (probs >= best_t).astype(int)
    
    r = {
        'macro': float(f1_score(targets, preds, average='macro', zero_division=0)),
        'micro': float(f1_score(targets, preds, average='micro', zero_division=0)),
        'seen': float(f1_score(targets[:,seen], preds[:,seen], average='macro', zero_division=0)),
        'unseen': float(f1_score(targets[:,unseen], preds[:,unseen], average='macro', zero_division=0)),
        'exc': float(f1_score(targets[:,exc_idx], preds[:,exc_idx], average='macro', zero_division=0)),
        'threshold': float(best_t),
    }
    results[tag] = r
    print(f'  RESULT: macro={r["macro"]:.4f} micro={r["micro"]:.4f} seen={r["seen"]:.4f} unseen={r["unseen"]:.4f} exc={r["exc"]:.4f} t={r["threshold"]:.2f}', flush=True)

with open('outputs/logs/jusdef_all_detailed.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n\n=== FINAL SUMMARY ===', flush=True)
print(f'{"tag":<25} {"macro":<8} {"micro":<8} {"seen":<8} {"unseen":<8} {"exc":<8}', flush=True)
print('-' * 70, flush=True)
for tag, r in results.items():
    print(f'{tag:<25} {r["macro"]:<8.4f} {r["micro"]:<8.4f} {r["seen"]:<8.4f} {r["unseen"]:<8.4f} {r["exc"]:<8.4f}', flush=True)

print(f'\nSaved to outputs/logs/jusdef_all_detailed.json', flush=True)
