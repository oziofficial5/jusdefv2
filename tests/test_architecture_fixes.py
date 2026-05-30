"""
Phase 0 architecture tests.

Each test verifies one architectural fix. They should all FAIL on v1's
unchanged jusdef.py and PASS after the corresponding fix is applied.

Run with: pytest tests/test_architecture_fixes.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

R2_KEY = ("sec", "mentions", "conc")
GRAPH_PATH = "data/processed/graphs/test_graphs.pt"


def _load_one_test_graph():
    graphs = torch.load(GRAPH_PATH, map_location="cpu")
    return graphs[0]


def _randomize_operators(g, seed=42):
    """Randomise r2 operators in place."""
    if R2_KEY in g.edge_types:
        store = g[R2_KEY]
        if hasattr(store, "operator") and store.operator is not None:
            rng = np.random.RandomState(seed)
            n = store.operator.numel()
            store.operator = torch.tensor(
                rng.randint(0, 4, size=n),
                dtype=torch.long,
                device=store.operator.device,
            )
    return g


def _forward(model, g):
    x_dict = {nt: g[nt].x for nt in g.node_types}
    ei_dict = {et: g[et].edge_index for et in g.edge_types}
    edge_attr_dict = None

    if R2_KEY in g.edge_types and g[R2_KEY].edge_index.size(1) > 0:
        r2 = g[R2_KEY]
        attrs = {
            "operator": r2.operator,
            "priority": r2.priority,
        }
        for k in ("auth_type", "auth_level", "auth_recency"):
            if hasattr(r2, k):
                attrs[k] = getattr(r2, k)
        edge_attr_dict = {R2_KEY: attrs}

    h, _ = model(x_dict, ei_dict, edge_attr_dict)
    return h


def test_F1_operators_reach_sections():
    """Randomising operators must change section embeddings."""
    from src.model.jusdef import JusDef

    torch.manual_seed(0)
    model = JusDef(
        in_dim=768,
        hidden_dim=512,
        num_layers=2,
        use_dmp=True,
        use_authority=True,
    ).eval()

    g_orig = _load_one_test_graph()
    h_orig = _forward(model, g_orig)

    g_rand = _randomize_operators(_load_one_test_graph(), seed=42)
    h_rand = _forward(model, g_rand)

    sec_diff = (h_orig["sec"] - h_rand["sec"]).abs().max().item()
    assert sec_diff > 1e-4, (
        f"Section embeddings are operator-blind (max |Δ|={sec_diff:.2e}). "
        "F1 fix not applied: add reverse mention edge."
    )


def test_F1b_operators_reach_predictions():
    """Operators must change the actual prediction scores."""
    from src.model.jusdef import JusDef

    torch.manual_seed(0)
    model = JusDef(
        in_dim=768,
        hidden_dim=512,
        num_layers=2,
        use_dmp=True,
        use_authority=True,
    ).eval()

    g_orig = _load_one_test_graph()
    h_orig = _forward(model, g_orig)

    g_rand = _randomize_operators(_load_one_test_graph(), seed=42)
    h_rand = _forward(model, g_rand)

    with torch.no_grad():
        doc_orig = model.pool_document(h_orig["sec"])
        doc_rand = model.pool_document(h_rand["sec"])
        scores_orig = model.score(doc_orig, h_orig["label"])
        scores_rand = model.score(doc_rand, h_rand["label"])

    score_diff = (scores_orig - scores_rand).abs().max().item()
    assert score_diff > 1e-4, (
        f"Operators do not reach prediction scores (max |Δ|={score_diff:.2e})"
    )


def test_F2_authority_scorer_in_gradient_path():
    """Authority scorer parameters must receive non-zero gradient."""
    from src.model.jusdef import JusDef

    torch.manual_seed(0)
    model = JusDef(
        in_dim=768,
        hidden_dim=512,
        num_layers=2,
        use_dmp=True,
        use_authority=True,
    )
    model.train()

    g = _load_one_test_graph()
    h = _forward(model, g)
    loss = sum(v.sum() for v in h.values())
    loss.backward()

    auth_params = list(model.authority_scorer.parameters())
    assert len(auth_params) > 0, "No authority_scorer params found"

    nonzero_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in auth_params
    )
    assert nonzero_grad, (
        "Authority scorer params received zero gradient. "
        "F2 fix not applied: wire authority_scorer into forward."
    )


def test_F3a_concept_ontology_edge_present():
    """r3 (conc -> ontology -> conc) should exist in the graph."""
    g = _load_one_test_graph()
    needed = ("conc", "ontology", "conc")
    assert needed in set(g.edge_types), (
        f"Missing edge type {needed}. "
        "F3a fix not applied: add EuroVoc ontology edges to constructed graphs."
    )


@pytest.mark.xfail(reason="Deferred until full authority graph rebuild on Ampere.")
def test_F3b_authority_hierarchy_edge_present():
    """r5 (auth -> hierarchy -> auth) should exist in the graph."""
    g = _load_one_test_graph()
    needed = ("auth", "hierarchy", "auth")
    assert needed in set(g.edge_types), (
        f"Missing edge type {needed}. "
        "F3b fix not applied: add authority hierarchy edges."
    )


@pytest.mark.xfail(reason="Deferred until full authority-concept graph rebuild on Ampere.")
def test_F3c_authority_concept_edge_present():
    """r6 (auth -> relates_to -> conc) should exist in the graph."""
    g = _load_one_test_graph()
    needed = ("auth", "relates_to", "conc")
    assert needed in set(g.edge_types), (
        f"Missing edge type {needed}. "
        "F3c fix not applied: add authority-concept edges."
    )


def test_F5_dmp_reduces_when_all_aff():
    """When all operators are AFF, the defeat mask must be all-1s."""
    from src.model.dmp_layer import compute_defeat_mask

    n = 100
    ops = torch.zeros(n, dtype=torch.long)
    pri = torch.zeros(n, dtype=torch.float)
    dst = torch.zeros(n, dtype=torch.long)
    mask = compute_defeat_mask(ops, pri, dst, temperature=5.0)

    assert (mask >= 0.99).all(), (
        f"Defeat mask not all-1 with uniform AFF: {mask.mean().item():.4f}"
    )
