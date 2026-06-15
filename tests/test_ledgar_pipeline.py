"""
Unit tests for the LEDGAR pipeline (preprocess + model).

Tests are designed to run on the laptop without requiring the cluster
or full LEDGAR preprocessing.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.jusdef_ledgar import JusDefLEDGAR


def _make_dummy_batch(n_paragraphs=3, sents_per_para=(5, 3, 7), in_dim=768):
    """Make a synthetic batch matching the trainer's collate output."""
    sent_embs_list = []
    sent_to_para_list = []
    ops_list = []
    for i, n in enumerate(sents_per_para):
        sent_embs_list.append(torch.randn(n, in_dim))
        sent_to_para_list.append(torch.full((n,), i, dtype=torch.long))
        ops_list.append(torch.randint(0, 4, (n,)))
    sent_embs = torch.cat(sent_embs_list, dim=0)
    sent_to_para = torch.cat(sent_to_para_list, dim=0)
    operators = torch.cat(ops_list, dim=0)
    return sent_embs, sent_to_para, operators, n_paragraphs


def test_ledgar_v3_forward_runs():
    """JusDefLEDGAR with v3 variant must produce (B, num_classes) logits."""
    model = JusDefLEDGAR(
        in_dim=768, hidden_dim=128, num_classes=100,
        num_layers=1, dmp_variant="v3"
    ).eval()

    sent_embs, sent_to_para, operators, num_paras = _make_dummy_batch()
    logits = model(sent_embs, sent_to_para, operators, num_paras)

    assert logits.shape == (num_paras, 100), (
        f"Wrong output shape: {logits.shape}, expected ({num_paras}, 100)"
    )
    assert not torch.isnan(logits).any(), "NaN in logits"


def test_ledgar_mean_forward_runs():
    """JusDefLEDGAR with mean variant produces same shape (baseline check)."""
    model = JusDefLEDGAR(
        in_dim=768, hidden_dim=128, num_classes=100,
        num_layers=1, dmp_variant="mean"
    ).eval()

    sent_embs, sent_to_para, operators, num_paras = _make_dummy_batch()
    logits = model(sent_embs, sent_to_para, operators, num_paras)

    assert logits.shape == (num_paras, 100)


def test_ledgar_v3_coef_regulariser():
    """V3 variant should expose coef_regulariser; mean variant returns zero."""
    model_v3 = JusDefLEDGAR(
        in_dim=768, hidden_dim=128, num_classes=100,
        num_layers=2, dmp_variant="v3"
    )
    reg_v3 = model_v3.v3_coef_regulariser()
    assert reg_v3.numel() == 1
    # At init, coefs = init values, regulariser should be near 0
    assert reg_v3.item() < 1e-3

    model_mean = JusDefLEDGAR(
        in_dim=768, hidden_dim=128, num_classes=100,
        num_layers=1, dmp_variant="mean"
    )
    reg_mean = model_mean.v3_coef_regulariser()
    assert reg_mean.item() == 0.0


def test_ledgar_v3_backward():
    """Gradient must flow through V3Layer and reach classifier weights."""
    model = JusDefLEDGAR(
        in_dim=768, hidden_dim=128, num_classes=100,
        num_layers=1, dmp_variant="v3"
    ).train()

    sent_embs, sent_to_para, operators, num_paras = _make_dummy_batch()
    sent_embs.requires_grad_(False)

    logits = model(sent_embs, sent_to_para, operators, num_paras)
    targets = torch.randint(0, 100, (num_paras,))
    loss = nn.functional.cross_entropy(logits, targets) + model.v3_coef_regulariser()
    loss.backward()

    # Classifier weights got gradient
    assert model.classifier.weight.grad is not None
    assert model.classifier.weight.grad.abs().sum().item() > 0

    # V3 op_coef got gradient
    for layer in model.agg_layers:
        assert layer.op_coef.grad is not None


def test_ledgar_v3_responds_to_operator_changes():
    """Changing operator labels must change model output (signal preservation)."""
    torch.manual_seed(0)
    model = JusDefLEDGAR(
        in_dim=768, hidden_dim=128, num_classes=100,
        num_layers=1, dmp_variant="v3"
    ).eval()

    sent_embs, sent_to_para, _, num_paras = _make_dummy_batch()

    # Two different operator assignments
    ops_a = torch.zeros(sent_embs.size(0), dtype=torch.long)  # all AFF
    ops_b = torch.full((sent_embs.size(0),), 1, dtype=torch.long)  # all NEG

    out_a = model(sent_embs, sent_to_para, ops_a, num_paras)
    out_b = model(sent_embs, sent_to_para, ops_b, num_paras)

    # AFF=+1 vs NEG=-1 should produce very different outputs
    diff = (out_a - out_b).abs().max().item()
    assert diff > 1e-2, (
        f"Model insensitive to operator changes (max |Δ|={diff:.2e})"
    )
