"""
JusDef: Full model assembling all components.

Architecture:
1. Input projection (all node types: 768 -> hidden_dim)
2. Standard HeteroConv for non-r2 edges
3. DMP for r2 edges (sec -> conc, with defeat)
4. Reverse mention propagation (conc -> sec)
5. Section-level attention pooling -> document embedding
6. Dot-product logits against label embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

from src.model.authority_scorer import AuthorityScorer
from src.model.dmp_layer import DMPLayer


class JusDef(nn.Module):
    def __init__(
        self,
        in_dim=768,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3,
        temperature=5.0,
        use_dmp=True,
        use_authority=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_dmp = use_dmp
        self.use_authority = use_authority

        node_types = ["doc", "sec", "conc", "auth", "label"]
        self.input_proj = nn.ModuleDict(
            {nt: Linear(in_dim, hidden_dim) for nt in node_types}
        )

        if use_authority:
            self.authority_scorer = AuthorityScorer(type_emb_dim=8)

        if use_dmp:
            self.dmp_layers = nn.ModuleList(
                [
                    DMPLayer(hidden_dim, hidden_dim, temperature, dropout)
                    for _ in range(num_layers)
                ]
            )

        self.hetero_convs = nn.ModuleList()
        self.hetero_conv_edge_types = []
        for _ in range(num_layers):
            conv_dict = {
                ("doc", "has_section", "sec"): SAGEConv(hidden_dim, hidden_dim),
                ("label", "maps_to", "conc"): SAGEConv(hidden_dim, hidden_dim),
                ("label", "parent_of", "label"): SAGEConv(hidden_dim, hidden_dim),
                ("sec", "cites", "auth"): SAGEConv(hidden_dim, hidden_dim),
                ("conc", "mentions_rev", "sec"): SAGEConv(hidden_dim, hidden_dim),
            }

            if not use_dmp:
                conv_dict[("sec", "mentions", "conc")] = SAGEConv(
                    hidden_dim, hidden_dim
                )

            self.hetero_conv_edge_types.append(set(conv_dict.keys()))
            self.hetero_convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.out_proj = Linear(hidden_dim, hidden_dim)
        self.sec_attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h = {}
        for nt, x in x_dict.items():
            if nt in self.input_proj:
                h[nt] = F.relu(self.input_proj[nt](x))
            else:
                h[nt] = x

        defeat_info = {"active_embs": [], "defeated_embs": []}
        r2_key = ("sec", "mentions", "conc")

        for layer_idx, hetero_conv in enumerate(self.hetero_convs):
            conv_edge_types = self.hetero_conv_edge_types[layer_idx]

            if self.use_dmp:
                candidate_edges = {
                    k: v
                    for k, v in edge_index_dict.items()
                    if k != r2_key and v.size(1) > 0 and k in conv_edge_types
                }
            else:
                candidate_edges = {
                    k: v
                    for k, v in edge_index_dict.items()
                    if v.size(1) > 0 and k in conv_edge_types
                }

            new_h = {}
            if candidate_edges:
                new_h = hetero_conv(h, candidate_edges)
                new_h = {k: self.dropout(F.relu(v)) for k, v in new_h.items()}

            for nt in h:
                if nt not in new_h:
                    new_h[nt] = h[nt]
            h = new_h

            if self.use_dmp and r2_key in edge_index_dict and edge_attr_dict is not None:
                r2_ei = edge_index_dict[r2_key]
                if r2_ei.size(1) > 0:
                    r2_ops = edge_attr_dict[r2_key]["operator"]
                    r2_pri_raw = edge_attr_dict[r2_key]["priority"]

                    if (
                        self.use_authority
                        and r2_key in edge_attr_dict
                        and "auth_type" in edge_attr_dict[r2_key]
                        and "auth_level" in edge_attr_dict[r2_key]
                        and "auth_recency" in edge_attr_dict[r2_key]
                    ):
                        r2_pri = self.authority_scorer(
                            edge_attr_dict[r2_key]["auth_type"],
                            edge_attr_dict[r2_key]["auth_level"],
                            edge_attr_dict[r2_key]["auth_recency"],
                        )
                    else:
                        r2_pri = r2_pri_raw

                    src_embs = h["sec"][r2_ei[0]]
                    dst_ids = r2_ei[1]
                    concept_ids = dst_ids
                    num_conc = h["conc"].size(0)

                    dmp_out = self.dmp_layers[layer_idx](
                        src_embs, dst_ids, r2_ops, r2_pri, concept_ids, num_conc
                    )

                    h["conc"] = h["conc"] + dmp_out

                    active, defeated = self.dmp_layers[
                        layer_idx
                    ].get_active_defeated_embeddings(
                        src_embs, dst_ids, r2_ops, r2_pri, concept_ids
                    )

                    defeat_info["active_embs"].append(active)
                    defeat_info["defeated_embs"].append(defeated)

            h = {k: self.out_proj(v) for k, v in h.items()}

        if defeat_info["active_embs"]:
            defeat_info["active_embs"] = torch.cat(defeat_info["active_embs"], dim=0)
            defeat_info["defeated_embs"] = torch.cat(
                defeat_info["defeated_embs"], dim=0
            )
        else:
            defeat_info = None

        return h, defeat_info

    def pool_document(self, sec_embs):
        attn = self.sec_attention(sec_embs)
        weights = torch.softmax(attn, dim=0)
        return (weights * sec_embs).sum(dim=0, keepdim=True)

    def score(self, doc_emb, label_embs):
        return doc_emb @ label_embs.T