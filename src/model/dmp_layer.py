"""
Defeasible Message Passing (DMP) layer for JusDef.

This is the core contribution. DMP differs from standard GNN aggregation
in one key way: messages can be DEFEATED (excluded from aggregation)
based on operator precedence and authority priority.

Key concepts:
  - Each r2 edge (sec -> conc) carries an operator (AFF/NEG/EXC/OVR)
    and an authority priority score
  - Message m_i defeats m_j if:
    (1) op_precedence(m_i) > op_precedence(m_j), OR
    (2) same precedence AND priority(m_i) > priority(m_j)
    AND both target the same concept node
  - Defeated messages are excluded via a binary mask
  - A Straight-Through Estimator (STE) makes the discrete mask differentiable

When no defeats occur (all AFF, equal priorities), DMP reduces exactly
to standard relational attention — making JusDef a strict generalisation.

PERFORMANCE NOTE:
  The defeat mask and aggregation are fully vectorized using
  scatter_reduce and scatter operations. No Python loops over
  edges or concept nodes. This is critical for EUR-LEX scale
  (200-1200 r2 edges per document, 55K documents).

Reference: JusDef paper Section 4.3, Equations 1-3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


OPERATOR_PRECEDENCE = {0: 0, 1: 1, 2: 2, 3: 3}


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator for differentiable hard defeat masking.

    Forward pass: hard binary mask (0 or 1)
    Backward pass: smooth sigmoid gradient

    Reference: Bengio et al., 2013 (arXiv:1308.3432)
    """

    @staticmethod
    def forward(ctx, priority_gap, temperature=5.0):
        ctx.save_for_backward(priority_gap)
        ctx.temperature = temperature
        return (priority_gap <= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        priority_gap, = ctx.saved_tensors
        tau = ctx.temperature
        sig = torch.sigmoid(-tau * priority_gap)
        sigmoid_grad = tau * sig * (1 - sig)
        return grad_output * sigmoid_grad, None


def compute_defeat_mask(operators, priorities, dst_nodes, temperature=5.0):
    """
    Vectorized defeat mask computation.

    Combines operator precedence and priority into a single defeat score:
        defeat_score = operator * 1000 + priority

    This encoding preserves the two-part defeat rule exactly:
      - Higher operator always wins (gap >= 1000 >> max priority)
      - Same operator: higher priority wins (gap = pri_j - pri_i > 0)
      - Equal operator AND equal priority: both survive (gap = -eps < 0)

    Uses scatter_reduce to find the group maximum in O(E) time,
    replacing the original O(E^2) pairwise Python loops.

    Args:
        operators: tensor (E,) int — operator type (0=AFF..3=OVR)
        priorities: tensor (E,) float — authority priority
        dst_nodes: tensor (E,) int — grouping key (concept node id)
        temperature: float — STE temperature

    Returns:
        mask: tensor (E,) float — 1.0=active, 0.0=defeated
    """
    E = operators.size(0)
    if E == 0:
        return torch.ones(0, device=operators.device)

    # Combine into single score: operator dominates via 1000x scaling
    defeat_score = operators.float() * 1000.0 + priorities

    # Non-in-place scatter_reduce so autograd tracks gradients properly
    num_groups = dst_nodes.max().item() + 1
    base = torch.full((num_groups,), -1e9, device=operators.device)
    group_max = base.scatter_reduce(
        0, dst_nodes, defeat_score, reduce="amax", include_self=True
    )

    # Positive gap = stronger competing message exists -> defeated
    # Zero/negative gap = strongest or tied -> active
    priority_gap = group_max[dst_nodes] - defeat_score - 0.001

    mask = StraightThroughEstimator.apply(priority_gap, temperature)
    return mask


class DMPLayer(nn.Module):
    """
    Defeasible Message Passing layer (vectorized).

    Only operates on r2 (sec -> conc) edges. All other edge types
    use standard HeteroConv aggregation in the JusDef model.

    Architecture per layer:
      1. Apply operator-specific weight matrices W_omega to source embeddings
      2. Compute defeat mask via STE (vectorized scatter_reduce)
      3. Soft-mask messages
      4. Attention-weighted aggregation of active messages per concept node

    When all operators are AFF and priorities are equal, DMP reduces to
    standard attention aggregation.

    Reference: JusDef paper Section 4.3, Equation 2
    """

    NUM_OPERATORS = 4

    def __init__(self, in_dim=512, out_dim=512, temperature=5.0, dropout=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.temperature = temperature

        self.W_op = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(self.NUM_OPERATORS)]
        )

        self.attn = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_embs, dst_node_ids, operators, priorities, concept_ids, num_dst):
        """
        Fully vectorized forward pass.

        Args:
            src_embs: tensor (E, in_dim)
            dst_node_ids: tensor (E,) — target concept node per edge
            operators: tensor (E,) int — operator type per edge
            priorities: tensor (E,) float — authority priority per edge
            concept_ids: tensor (E,) int — concept ID for defeat scoping
            num_dst: int — total number of concept nodes

        Returns:
            out: tensor (num_dst, out_dim) — updated concept node embeddings
        """
        E = src_embs.size(0)
        device = src_embs.device

        if E == 0:
            return torch.zeros(num_dst, self.out_dim, device=device)

        # Step 1: Operator-specific transforms
        msg = torch.zeros(E, self.out_dim, device=device)
        for op_idx in range(self.NUM_OPERATORS):
            op_mask = (operators == op_idx)
            if op_mask.any():
                msg[op_mask] = self.W_op[op_idx](src_embs[op_mask])

        # Step 2: Differentiable defeat mask
        defeat_mask = compute_defeat_mask(
            operators, priorities, concept_ids, self.temperature
        )

        # Step 3: Soft-mask messages
        active_msg = msg * defeat_mask.unsqueeze(-1)

        # Step 4: Attention aggregation with soft defeat weighting
        # Compute raw attention from original msg to avoid hard gating here
        attn_raw = self.attn(msg).squeeze(-1)

        # Max per group for numerical stability
        attn_base = torch.full((num_dst,), -1e9, device=device)
        attn_max = attn_base.scatter_reduce(
            0, dst_node_ids, attn_raw, reduce="amax", include_self=True
        )
        attn_stable = attn_raw - attn_max[dst_node_ids]

        # Exponentiate and softly weight by defeat_mask
        attn_exp = torch.exp(attn_stable) * defeat_mask
        attn_sum = torch.zeros(num_dst, device=device)
        attn_sum.scatter_add_(0, dst_node_ids, attn_exp)
        attn_sum = attn_sum.clamp(min=1e-8)

        # Normalize within destination groups
        attn_weights = attn_exp / attn_sum[dst_node_ids]

        # Weighted aggregation
        weighted_msg = active_msg * attn_weights.unsqueeze(-1)
        out = torch.zeros(num_dst, self.out_dim, device=device)
        out.scatter_add_(
            0,
            dst_node_ids.unsqueeze(-1).expand(-1, self.out_dim),
            weighted_msg,
        )

        return self.dropout(out)

    def get_active_defeated_embeddings(
        self, src_embs, dst_node_ids, operators, priorities, concept_ids
    ):
        """
        Helper for L_defeat loss.

        Returns:
            active_embs: tensor (P, out_dim)
            defeated_embs: tensor (Q, out_dim)
        """
        E = src_embs.size(0)
        if E == 0:
            return (
                torch.zeros(0, self.out_dim, device=src_embs.device),
                torch.zeros(0, self.out_dim, device=src_embs.device),
            )

        msg = torch.zeros(E, self.out_dim, device=src_embs.device)
        for op_idx in range(self.NUM_OPERATORS):
            op_mask = (operators == op_idx)
            if op_mask.any():
                msg[op_mask] = self.W_op[op_idx](src_embs[op_mask])

        defeat_mask = compute_defeat_mask(
            operators, priorities, concept_ids, self.temperature
        )

        active = msg[defeat_mask > 0.5]
        defeated = msg[defeat_mask <= 0.5]

        return active, defeated