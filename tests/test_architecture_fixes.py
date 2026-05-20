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

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_one_test_graph():
    graphs = torch.load("data/processed/graphs/test_graphs.pt", map_location="cpu")
    return graphs[0]


def _randomize_operators(g, seed=42):
    """Randomise r2 operators in place."""
    r2_key = ("sec", "mentions", "conc")
    if r2_key in g.edge_types:
        store = g[r2_key]
        if hasattr(store, "operator") and store.operator is not None:
            rng = np.random.RandomState(seed)
            n = store.operator.numel()
            store.operator = torch.tensor(
                rng.randint(0, 4, size=n), dtype=torch.long
            )
    return g


def _forward(model, g):
    """Run the model forward and return per-node-type hidden states."""
    r2_key = ("sec", "mentions", "conc")
    x_dict = {nt: g[nt].x for nt in g.node_types}
    ei_dict = {et: g[et].edge_index for et in g.edge_types}
    edge_attr_dict = None
    if r2_key in g.edge_types and g[r2_key].edge_index.size(1) > 0:
        edge_attr_dict = {
            r2_key: {
                "operator": g[r2_key].operator,
                "priority": g[r2_key].priority,
            }
        }
    h, _ = model(x_dict, ei_dict, edge_attr_dict)
    return h


# -----------------------------------------------------------------------------
# F1: operators must reach section embeddings
# -----------------------------------------------------------------------------
def test_F1_operators_reach_sections():
    """Randomising operators must change section embeddings.

    On v1 (no reverse edge), sec embeddings are operator-blind. This test
    fails until the reverse mention edge is added.
    """
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


# -----------------------------------------------------------------------------
# F2: authority scorer must actually receive gradients
# -----------------------------------------------------------------------------
def test_F2_authority_scorer_in_gradient_path():
    """The authority scorer's parameters must receive non-zero gradient.

    On v1, AuthorityScorer is instantiated but never called. Its params
    have grad=None or grad=0. This test fails until the scorer is wired
    into the forward pass.
    """
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

    nonzero_grad = False
    for p in auth_params:
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            nonzero_grad = True
            break

    assert nonzero_grad, (
        "Authority scorer params received zero gradient. "
        "F2 fix not applied: wire authority_scorer into forward."
    )


# -----------------------------------------------------------------------------
# F3: missing edge types should exist on the constructed graphs
# -----------------------------------------------------------------------------
def test_F3_paper_edge_types_present():
    """All 8 edge types from paper Table 1(a) should be in the graph."""
    g = _load_one_test_graph()
    needed = {
        ("doc", "has_section", "sec"),   # r1
        ("sec", "mentions", "conc"),     # r2
        ("conc", "ontology", "conc"),    # r3 — MISSING in v1
        ("sec", "cites", "auth"),        # r4
        ("auth", "hierarchy", "auth"),   # r5 — MISSING in v1
        ("auth", "relates_to", "conc"),  # r6 — MISSING in v1
        ("label", "maps_to", "conc"),    # r7
        ("label", "parent_of", "label"), # r8
    }
    present = set(g.edge_types)
    missing = needed - present
    assert not missing, f"Edge types missing from constructed graph: {missing}"


# -----------------------------------------------------------------------------
# F5: DMP reduction to R-GCN (sanity check, should always pass)
# -----------------------------------------------------------------------------
def test_F5_dmp_reduces_when_all_aff():
    """When all operators are AFF, the defeat mask must be all-1s."""
    from src.model.dmp_layer import compute_defeat_mask

    n = 100
    ops = torch.zeros(n, dtype=torch.long)   # all AFF
    pri = torch.zeros(n, dtype=torch.float)  # all equal priority
    dst = torch.zeros(n, dtype=torch.long)   # same concept
    mask = compute_defeat_mask(ops, pri, dst, temperature=5.0)

    assert (mask >= 0.99).all(), (
        f"Defeat mask not all-1 with uniform AFF: {mask.mean().item():.4f}"
    )