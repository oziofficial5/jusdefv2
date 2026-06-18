"""
V4 routing modules. Two routers that gate the v3 update against the
mean-aggregation baseline on a per-paragraph basis.

The v4 architecture exploits the empirical operating regime identified in
Chapter 8: v3 outperforms the mean baseline on the 10-20% non-AFF density
bin (peaking 12-18%, hard upper boundary 20%) and underperforms on the
0% (AFF-only) bin and the >=20% dense bin. v4 routes each paragraph to
the appropriate aggregator based on observable density.

Two router variants:

  HardDensityRouter: inference-time hard gate on the per-paragraph
      non-AFF density. No learned parameters. Routes to v3 if density
      is in [lo, hi); otherwise routes to mean baseline.

  SoftDensityRouter: learned per-paragraph soft gate from a small MLP
      over [density, n_non_aff, n_total]. Initialised to alpha ~ 1, so
      training starts as v3 and learns to suppress the v3 contribution
      on paragraphs where mean is better.

Both routers produce a gate alpha in (num_paragraphs, 1) that the caller
applies to the v3 update via h_para = h_mean + alpha * (h_v3 - h_mean).
"""

import torch
import torch.nn as nn


class HardDensityRouter(nn.Module):
    """Per-paragraph hard gate on observable non-AFF density.

    Routes to v3 when density in [density_lo, density_hi), else to mean.
    No learned parameters; routing depends only on the input.
    """

    def __init__(self, density_lo=0.10, density_hi=0.20):
        super().__init__()
        self.density_lo = float(density_lo)
        self.density_hi = float(density_hi)

    def forward(self, density):
        """
        Args:
            density: (num_paragraphs,) float tensor of non-AFF density
        Returns:
            gate: (num_paragraphs, 1) in {0.0, 1.0}
        """
        in_regime = (density >= self.density_lo) & (density < self.density_hi)
        return in_regime.float().unsqueeze(-1)


class SoftDensityRouter(nn.Module):
    """Per-paragraph learned soft gate from density features.

    Input features: [density, n_non_aff, n_total]. A 2-layer MLP outputs
    a single logit; sigmoid produces alpha in (0, 1). The final linear
    layer is initialised so that initial alpha is approximately 1, i.e.
    training starts as v3 and learns to suppress the v3 contribution on
    paragraphs where mean is better.
    """

    NUM_FEATURES = 3  # [density, n_non_aff, n_total]

    def __init__(self, hidden_dim=32, init_bias=5.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self.NUM_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialise the final layer so sigmoid(output) ~ 1 at init
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.fill_(float(init_bias))

    def forward(self, density, n_non_aff, n_total):
        """
        Args:
            density:    (num_paragraphs,)
            n_non_aff:  (num_paragraphs,)
            n_total:    (num_paragraphs,)
        Returns:
            gate: (num_paragraphs, 1) in (0, 1)
        """
        feats = torch.stack([density, n_non_aff, n_total], dim=-1)
        logit = self.mlp(feats)
        return torch.sigmoid(logit)
