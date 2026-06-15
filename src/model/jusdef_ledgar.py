"""
JusDef-LEDGAR — paragraph-level classifier for LEDGAR contract clauses.

LEDGAR (LexGLUE) is single-label 100-class classification:
- One paragraph per instance
- Sentences within paragraph carry operator labels (from neural detector)
- Defeasible aggregation produces a paragraph representation
- Linear classifier produces label logits

This is a simpler architecture than EUR-Lex JusDef because:
- No EuroVoc concepts (label IS the prediction target)
- No multi-section structure
- No authority graph
- Pure operator-aware sentence aggregation

Architecture:

  sentence_emb (768) --> input_proj --> shared_msg (hidden)
                                    \
                                     V3 layer (operator-aware aggregation)
                                    /
  paragraph_init (mean of sents) --
                                    \
                                     residual_add --> paragraph_rep --> classifier --> 100 logits

The V3 layer here is the same code as `src/model/v3_layer.py` (the EUR-Lex v3).
Reusing the same layer is intentional — the architectural claim is that v3
generalises across corpora when the operator-density condition is satisfied.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.v3_layer import V3Layer


class JusDefLEDGAR(nn.Module):
    """
    Paragraph-level defeasibility-aware classifier for LEDGAR.

    Forward signature works on batches of (sentence_embeddings, operators,
    sentence_to_paragraph_index).
    """

    def __init__(
        self,
        in_dim=768,
        hidden_dim=512,
        num_classes=100,
        num_layers=1,
        dropout=0.3,
        dmp_variant="v3",  # 'v3' uses V3Layer; 'mean' uses simple mean aggregation
        v3_init_coefs=(1.0, -1.0, -0.5, 1.0),
        v3_coef_reg_strength=0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dmp_variant = dmp_variant

        # Sentence embedding -> hidden
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Aggregation layers
        if dmp_variant == "v3":
            self.agg_layers = nn.ModuleList(
                [
                    V3Layer(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        init_coefs=v3_init_coefs,
                        coef_reg_strength=v3_coef_reg_strength,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif dmp_variant == "mean":
            self.agg_layers = nn.ModuleList()  # mean aggregation, no params
        else:
            raise ValueError(f"Unknown dmp_variant: {dmp_variant}")

        # Per-paragraph classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def v3_coef_regulariser(self):
        """Sum of per-layer coef regularisers (zero for non-v3 variants)."""
        if self.dmp_variant != "v3":
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return sum(layer.coef_regulariser() for layer in self.agg_layers)

    def _scatter_mean(self, src, index, dim_size):
        """Scatter mean: out[i] = mean(src[j] for j where index[j] == i)."""
        out = torch.zeros(dim_size, src.size(-1), device=src.device)
        counts = torch.zeros(dim_size, device=src.device)
        out.scatter_add_(
            0,
            index.unsqueeze(-1).expand(-1, src.size(-1)),
            src,
        )
        counts.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float))
        return out / counts.clamp(min=1.0).unsqueeze(-1)

    def forward(self, sent_embs, sent_to_para, operators, num_paragraphs):
        """
        Args:
            sent_embs: (TotalSents, in_dim) per-sentence LegalBERT [CLS] embeddings
            sent_to_para: (TotalSents,) which paragraph each sentence belongs to
            operators: (TotalSents,) integer operator id per sentence (0..3)
            num_paragraphs: int total number of paragraphs in batch

        Returns:
            logits: (num_paragraphs, num_classes)
        """
        # Project sentence embeddings into hidden space
        h_sent = F.relu(self.input_proj(sent_embs))  # (TotalSents, hidden)

        # Initial paragraph representation: mean of sentences
        h_para = self._scatter_mean(h_sent, sent_to_para, num_paragraphs)

        if self.dmp_variant == "v3":
            # Auth types: LEDGAR has no authority structure, use AFF (0) sentinel
            auth_types = torch.zeros_like(operators)
            for layer in self.agg_layers:
                update = layer(
                    h_sent,
                    h_para,
                    sent_to_para,
                    operators,
                    auth_types,
                    num_paragraphs,
                )
                h_para = h_para + update  # residual
        # "mean" variant: just use h_para as-is (already scatter_mean)

        h_para = self.dropout(h_para)
        logits = self.classifier(h_para)
        return logits
