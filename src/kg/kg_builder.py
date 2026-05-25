import torch
from torch_geometric.data import HeteroData
from typing import Dict, Any, List, Optional

OPERATOR_TO_INT = {
    "AFF": 0,
    "NEG": 1,
    "EXC": 2,
    "OVR": 3,
}

AUTH_TYPE_TO_INT = {
    "REGULATION": 0,
    "DIRECTIVE": 1,
    "DECISION": 2,
    "ARTICLE": 3,
    "CASE": 4,
}
AUTH_UNKNOWN = 5  # sentinel for sections with no citing authority


def build_document_graph(
    doc_struct: Dict[str, Any],
    doc_emb: torch.Tensor,
    section_embs: torch.Tensor,
    label_embs: torch.Tensor,
    label_adj: Optional[torch.Tensor] = None,
) -> HeteroData:
    """
    Build a JusDef v2 HeteroData graph for a single document.

    v2 additions over v1:
      - F1: reverse mention edge ('conc','mentions_rev','sec') with same
            operator/priority as the forward r2 edge.
      - F2: per-r2-edge authority features (auth_type, auth_level, auth_recency)
            mirrored on both forward and reverse r2 edges.
      - F3a: ontology edges ('conc','ontology','conc') from `label_adj`
            restricted to concepts mentioned in this document.

    doc_struct: one entry from *_processed.pkl (see v1 for structure).
    label_adj: optional [n_labels, n_labels] adjacency matrix (EuroVoc).
               If None, r3 edges are empty.
    """
    data = HeteroData()

    sections: List[Dict[str, Any]] = doc_struct["sections"]
    n_sec = len(sections)

    # --- DOC node -----------------------------------------------------------
    data["doc"].x = doc_emb.unsqueeze(0)  # (1, 768)

    # --- SEC nodes ----------------------------------------------------------
    if section_embs is not None and section_embs.size(0) == n_sec:
        data["sec"].x = section_embs  # (N_sec, 768)
    else:
        data["sec"].x = doc_emb.unsqueeze(0).expand(n_sec, -1).clone()

    sec_types = [sec.get("type_int", 2) for sec in sections]
    data["sec"].type_id = torch.tensor(sec_types, dtype=torch.long)

    # --- CONC nodes (unique label_idx per doc) ------------------------------
    conc_id_map: Dict[int, int] = {}
    conc_embs: List[torch.Tensor] = []

    for sec in sections:
        for c in sec.get("concepts", []):
            label_idx = c.get("label_idx")
            if label_idx is None:
                continue
            if label_idx not in conc_id_map and 0 <= label_idx < label_embs.size(0):
                conc_id_map[label_idx] = len(conc_id_map)
                conc_embs.append(label_embs[label_idx])

    has_real_concepts = bool(conc_embs)
    if has_real_concepts:
        data["conc"].x = torch.stack(conc_embs, dim=0)
        data["conc"].global_id = torch.tensor(
            list(conc_id_map.keys()), dtype=torch.long
        )
    else:
        data["conc"].x = torch.zeros(1, label_embs.size(1))
        data["conc"].global_id = torch.tensor([0], dtype=torch.long)
        conc_id_map[0] = 0

    # --- AUTH nodes ---------------------------------------------------------
    auth_map: Dict[str, int] = {}
    auth_features: List[List[float]] = []

    for sec in sections:
        for a in sec.get("authorities", []):
            key = a.get("text", "")
            if not key:
                continue
            if key not in auth_map:
                auth_map[key] = len(auth_map)
                atype = a.get("type", "ARTICLE")
                level = float(a.get("level", 2.0))
                recency = float(a.get("recency", 0.5))
                auth_features.append(
                    [float(AUTH_TYPE_TO_INT.get(atype, 3)), level, recency]
                )

    if auth_features:
        data["auth"].x = torch.zeros(len(auth_features), label_embs.size(1))
        data["auth"].features = torch.tensor(auth_features, dtype=torch.float)
    else:
        data["auth"].x = torch.zeros(1, label_embs.size(1))
        data["auth"].features = torch.tensor([[3.0, 2.0, 0.5]], dtype=torch.float)

    # --- LABEL nodes --------------------------------------------------------
    data["label"].x = label_embs  # (100, 768)

    # ════════════════════════════════════════════════════════════════════════
    #                                EDGES
    # ════════════════════════════════════════════════════════════════════════

    # --- r1: doc -> sec (containment) ---------------------------------------
    if n_sec > 0:
        src = torch.zeros(n_sec, dtype=torch.long)
        dst = torch.arange(n_sec, dtype=torch.long)
        data["doc", "has_section", "sec"].edge_index = torch.stack([src, dst], dim=0)
    else:
        data["doc", "has_section", "sec"].edge_index = torch.zeros(
            (2, 0), dtype=torch.long
        )

    # --- F2 prep: per-section dominant authority features -------------------
    #   For each section we pick its highest-level cited authority and use
    #   its (type, level, recency) as the auth features on every r2 edge
    #   that emanates from that section. Sections with no citing authority
    #   get the UNKNOWN sentinel.
    sec_to_auth_feat: Dict[int, tuple] = {}
    for si, sec in enumerate(sections):
        auths = sec.get("authorities", [])
        if not auths:
            continue
        dom = max(auths, key=lambda a: float(a.get("level", 0.0)))
        a_type = AUTH_TYPE_TO_INT.get(dom.get("type", "ARTICLE"), 3)
        a_level = float(dom.get("level", 2.0))
        a_recency = float(dom.get("recency", 0.5))
        sec_to_auth_feat[si] = (a_type, a_level, a_recency)

    # --- r2: sec -> conc (concept mention) AND r2_rev: conc -> sec ----------
    r2_src, r2_dst, r2_ops, r2_pri = [], [], [], []
    r2_auth_type, r2_auth_level, r2_auth_recency = [], [], []

    for si, sec in enumerate(sections):
        auths = sec.get("authorities", [])
        pri = max((float(a.get("level", 1.0)) for a in auths), default=1.0)

        # Per-section auth features (F2)
        if si in sec_to_auth_feat:
            sec_at, sec_al, sec_ar = sec_to_auth_feat[si]
        else:
            sec_at, sec_al, sec_ar = AUTH_UNKNOWN, 0.0, 0.0

        for c in sec.get("concepts", []):
            label_idx = c.get("label_idx")
            if label_idx is None:
                continue
            local_idx = conc_id_map.get(label_idx)
            if local_idx is None:
                continue

            op_str = c.get("operator", "AFF")
            op_id = OPERATOR_TO_INT.get(op_str, 0)

            r2_src.append(si)
            r2_dst.append(local_idx)
            r2_ops.append(op_id)
            r2_pri.append(pri)
            r2_auth_type.append(sec_at)
            r2_auth_level.append(sec_al)
            r2_auth_recency.append(sec_ar)

    if r2_src:
        # Forward edge sec -> conc
        data["sec", "mentions", "conc"].edge_index = torch.tensor(
            [r2_src, r2_dst], dtype=torch.long
        )
        data["sec", "mentions", "conc"].operator = torch.tensor(
            r2_ops, dtype=torch.long
        )
        data["sec", "mentions", "conc"].priority = torch.tensor(
            r2_pri, dtype=torch.float
        )
        # F2: real authority features
        data["sec", "mentions", "conc"].auth_type = torch.tensor(
            r2_auth_type, dtype=torch.long
        )
        data["sec", "mentions", "conc"].auth_level = torch.tensor(
            r2_auth_level, dtype=torch.float
        )
        data["sec", "mentions", "conc"].auth_recency = torch.tensor(
            r2_auth_recency, dtype=torch.float
        )

        # F1: reverse edge conc -> sec with mirrored attributes
        data["conc", "mentions_rev", "sec"].edge_index = torch.tensor(
            [r2_dst, r2_src], dtype=torch.long
        )
        data["conc", "mentions_rev", "sec"].operator = torch.tensor(
            r2_ops, dtype=torch.long
        )
        data["conc", "mentions_rev", "sec"].priority = torch.tensor(
            r2_pri, dtype=torch.float
        )
        data["conc", "mentions_rev", "sec"].auth_type = torch.tensor(
            r2_auth_type, dtype=torch.long
        )
        data["conc", "mentions_rev", "sec"].auth_level = torch.tensor(
            r2_auth_level, dtype=torch.float
        )
        data["conc", "mentions_rev", "sec"].auth_recency = torch.tensor(
            r2_auth_recency, dtype=torch.float
        )
    else:
        # Empty branches — mirror attribute shape for downstream code that
        # checks hasattr(...).
        for et in [
            ("sec", "mentions", "conc"),
            ("conc", "mentions_rev", "sec"),
        ]:
            data[et].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data[et].operator = torch.zeros(0, dtype=torch.long)
            data[et].priority = torch.zeros(0, dtype=torch.float)
            data[et].auth_type = torch.zeros(0, dtype=torch.long)
            data[et].auth_level = torch.zeros(0, dtype=torch.float)
            data[et].auth_recency = torch.zeros(0, dtype=torch.float)

    # --- F3a: r3 conc -> conc (EuroVoc ontology) ----------------------------
    #   For each pair of concepts mentioned in this document that are
    #   adjacent in the global EuroVoc graph, add a directed edge between
    #   their local node indices.
    r3_src, r3_dst = [], []
    if (
        has_real_concepts
        and label_adj is not None
        and label_adj.numel() > 0
    ):
        max_lbl = label_adj.size(0)
        mentioned = [lbl for lbl in conc_id_map.keys() if lbl < max_lbl]
        for la in mentioned:
            for lb in mentioned:
                if la == lb:
                    continue
                if label_adj[la, lb].item() > 0:
                    r3_src.append(conc_id_map[la])
                    r3_dst.append(conc_id_map[lb])

    if r3_src:
        data["conc", "ontology", "conc"].edge_index = torch.tensor(
            [r3_src, r3_dst], dtype=torch.long
        )
    else:
        data["conc", "ontology", "conc"].edge_index = torch.zeros(
            (2, 0), dtype=torch.long
        )

    # --- r4: sec -> auth (citation) -----------------------------------------
    r4_src, r4_dst = [], []
    for si, sec in enumerate(sections):
        for a in sec.get("authorities", []):
            key = a.get("text", "")
            if key in auth_map:
                ai = auth_map[key]
                r4_src.append(si)
                r4_dst.append(ai)

    if r4_src:
        data["sec", "cites", "auth"].edge_index = torch.tensor(
            [r4_src, r4_dst], dtype=torch.long
        )
    else:
        data["sec", "cites", "auth"].edge_index = torch.zeros(
            (2, 0), dtype=torch.long
        )

    # --- r7: label -> conc (concept-label mapping) --------------------------
    r7_src, r7_dst = [], []
    for label_idx, local_idx in conc_id_map.items():
        r7_src.append(label_idx)
        r7_dst.append(local_idx)

    if r7_src:
        data["label", "maps_to", "conc"].edge_index = torch.tensor(
            [r7_src, r7_dst], dtype=torch.long
        )
    else:
        data["label", "maps_to", "conc"].edge_index = torch.zeros(
            (2, 0), dtype=torch.long
        )

    # --- r8: label -> label (EuroVoc hierarchy, placeholder) ----------------
    data["label", "parent_of", "label"].edge_index = torch.zeros(
        (2, 0), dtype=torch.long
    )

    # ════════════════════════════════════════════════════════════════════════
    #                            TARGET LABELS
    # ════════════════════════════════════════════════════════════════════════
    target = torch.zeros(label_embs.size(0), dtype=torch.float)
    for l in doc_struct.get("labels", []):
        if 0 <= l < label_embs.size(0):
            target[l] = 1.0
    data.y = target

    return data