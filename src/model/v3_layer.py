"""
V3 — Signal-Preserving Defeasible Aggregation Layer.

Architectural design grounded in three NAMED prior failure modes (see thesis Ch. 7):

  1. STE bias on hard gates
     -> Jang et al. (Gumbel-Softmax) ICLR 2017
     -> "Gapped Straight-Through Estimator" ICML 2022
     FIX: replace hard binary mask with soft sigmoid attention (no STE).

  2. Signed-GNN sign cancellation across depth
     -> Zhu et al. "Sign is Not a Remedy" 2024
     -> "A Signed Graph Approach to Understanding and Mitigating Oversmoothing
        in GNNs" arXiv:2502.11394 (2025)
     FIX: per-operator signed coefficient with L2 drift regulariser to the
     semantic init values; prevents coefficients from collapsing to zero.

  3. Operator-specific weight asymmetric training
     -> Diagnosed empirically in this thesis (Ch. 6):
        W_AFF gets ~13M training examples per epoch; W_OVR gets ~100k.
        The rare W matrices are undertrained and contribute noise.
     FIX: single shared W_shared transform that receives gradient from every
     message. Operator semantics enter via (a) signed coefficient at aggregation
     and (b) operator embedding in the attention conditioner.

Theoretical claims (proofs in thesis Appendix C):

  Proposition 2 (Reduction to R-GCN):
    With op_coef = (1, 1, 1, 1), uniform attention logits, and W_shared matching
    the R-GCN weight on the `mentions` relation, V3Layer's update is identical
    to the R-GCN update over that relation.

  Proposition 3 (Hard-DMP limit case):
    As attention sharpens to one-hot on the maximum-priority message and
    op_coef -> (1, 0, 0, 0), V3Layer recovers the surviving-message structure of
    hard DMP without the W-undertraining noise (W_shared is well-trained).

  Proposition 4 (Strict generalisation):
    V3Layer's hypothesis class strictly contains both R-GCN's (via Prop 2) and
    hard DMP's (as Prop 3 limit). There exist parameter configurations in V3Layer
    achievable by neither baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class V3Layer(nn.Module):
    """
    Signal-preserving defeasible aggregation.

    Forward signature matches the existing JusDef integration except for one
    extra argument: the current destination embeddings (h["conc"]). This is
    required because v3's attention conditions on the destination context
    while v2's DMP did not.
    """

    NUM_OPERATORS = 4  # AFF=0, NEG=1, EXC=2, OVR=3
    NUM_AUTH_TYPES = 6

    def __init__(
        self,
        in_dim=512,
        out_dim=512,
        op_emb_dim=64,
        auth_emb_dim=64,
        init_coefs=(1.0, -1.0, -0.5, 1.0),  # AFF, NEG, EXC, OVR
        coef_reg_strength=0.01,
        dropout=0.3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.coef_reg_strength = coef_reg_strength

        # Single shared transform — gets gradient from every message
        # (Addresses failure mode 3: W-undertraining)
        self.W_shared = nn.Linear(in_dim, out_dim, bias=False)

        # Operator embedding for attention conditioning
        self.op_emb = nn.Embedding(self.NUM_OPERATORS, op_emb_dim)

        # Authority embedding (preserves F2)
        self.auth_emb = nn.Embedding(self.NUM_AUTH_TYPES, auth_emb_dim)

        # Per-operator signed coefficient — learned but initialised with
        # semantically meaningful priors.
        # (Addresses failure mode 2: paired with the drift regulariser below)
        self.op_coef = nn.Parameter(torch.tensor(init_coefs, dtype=torch.float32))
        self.register_buffer(
            "op_coef_init", torch.tensor(init_coefs, dtype=torch.float32)
        )

        # Attention conditioner MLP
        # Input: [shared_msg | op_emb | auth_emb | dst_emb]
        attn_in_dim = out_dim + op_emb_dim + auth_emb_dim + out_dim
        self.attn_mlp = nn.Sequential(
            nn.Linear(attn_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def coef_regulariser(self):
        """
        L2 penalty pulling op_coef toward semantic init values.

        Addresses failure mode 2 (signed-GNN sign cancellation): without this,
        the per-operator coefficients drift toward zero across depth, recovering
        a degenerate sign-blind aggregation. The regulariser preserves the
        AFF/NEG distinction during training.

        Loss caller should add: total_loss += layer.coef_regulariser()
        """
        return self.coef_reg_strength * ((self.op_coef - self.op_coef_init) ** 2).sum()

    def forward(
        self,
        src_embs,
        dst_emb,
        dst_node_ids,
        operators,
        auth_types,
        num_dst,
    ):
        """
        Forward pass.

        Args:
            src_embs:     (E, in_dim) source node embeddings (sec embeddings)
            dst_emb:      (N_dst, out_dim) current destination embeddings
            dst_node_ids: (E,) destination node index per edge
            operators:    (E,) integer operator id per edge (0..3)
            auth_types:   (E,) integer authority type id per edge (0..5)
            num_dst:      int total number of destination nodes

        Returns:
            out: (num_dst, out_dim) aggregated destination updates
        """
        E = src_embs.size(0)
        device = src_embs.device

        if E == 0:
            return torch.zeros(num_dst, self.out_dim, device=device)

        # 1. Shared transform — single W gets every message
        msg = self.W_shared(src_embs)  # (E, out_dim)

        # 2. Feature conditioning
        op_feat = self.op_emb(operators)  # (E, op_emb_dim)
        auth_clamped = auth_types.clamp(0, self.NUM_AUTH_TYPES - 1)
        auth_feat = self.auth_emb(auth_clamped)  # (E, auth_emb_dim)
        dst_feat = dst_emb[dst_node_ids]  # (E, out_dim)

        # 3. Attention conditioned on (msg, op, auth, dst)
        attn_input = torch.cat([msg, op_feat, auth_feat, dst_feat], dim=-1)
        attn_logits = self.attn_mlp(attn_input).squeeze(-1)  # (E,)

        # 4. Scatter-softmax per destination, numerically stable
        attn_max = torch.full((num_dst,), -1e9, device=device)
        attn_max = attn_max.scatter_reduce(
            0, dst_node_ids, attn_logits, reduce="amax", include_self=True
        )
        attn_stable = attn_logits - attn_max[dst_node_ids]
        attn_exp = torch.exp(attn_stable)
        attn_sum = torch.zeros(num_dst, device=device)
        attn_sum.scatter_add_(0, dst_node_ids, attn_exp)
        attn_sum = attn_sum.clamp(min=1e-8)
        attn_weights = attn_exp / attn_sum[dst_node_ids]  # (E,)

        # 5. Per-operator signed coefficient
        coefs = self.op_coef[operators]  # (E,)

        # 6. Final aggregation: signed, attention-weighted, ALL messages contribute
        combined_weights = (attn_weights * coefs).unsqueeze(-1)  # (E, 1)
        weighted_msg = msg * combined_weights

        out = torch.zeros(num_dst, self.out_dim, device=device)
        out.scatter_add_(
            0,
            dst_node_ids.unsqueeze(-1).expand(-1, self.out_dim),
            weighted_msg,
        )

        return self.dropout(out)

    def get_active_defeated_embeddings(
        self, src_embs, dst_node_ids, operators, auth_types, num_dst
    ):
        """
        Compatibility shim for the L_defeat loss interface.

        v3 has no defeat events (all messages contribute via soft attention).
        Returns empty tensors so L_defeat reduces to zero contribution.
        """
        device = src_embs.device
        empty = torch.zeros(0, self.out_dim, device=device)
        return empty, empty
