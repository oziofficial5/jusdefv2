import os
import sys
import pickle
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.kg.kg_builder import build_document_graph
from src.kg.graph_utils import validate_graph, print_graph_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Use small subset of docs")
    args = parser.parse_args()

    emb_dir = Path("data/processed/embeddings")
    proc_dir = Path("data/processed")
    graph_dir = Path("data/processed/graphs")
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Load label embeddings
    label_embs_data = torch.load(emb_dir / "label_embs.pt", map_location="cpu")
    if isinstance(label_embs_data, dict):
        label_embs = label_embs_data["embeddings"]
    else:
        label_embs = label_embs_data

    # Load label adjacency for F3a ontology edges (optional)
    label_adj_path = proc_dir / "label_adj.pt"
    if label_adj_path.exists():
        label_adj = torch.load(label_adj_path, map_location="cpu")
        print(f"Loaded label adjacency from {label_adj_path} with shape {tuple(label_adj.shape)}")
    else:
        label_adj = None
        print(f"WARNING: {label_adj_path} not found; ontology edges will be empty")

    for split in ["train", "validation", "test"]:
        print("=" * 50)
        print(f"Building graphs for split: {split}")
        print("=" * 50)

        pkl_path = proc_dir / f"{split}_processed.pkl"
        if not pkl_path.exists():
            print(" SKIP:", pkl_path, "not found")
            continue

        with open(pkl_path, "rb") as f:
            docs = pickle.load(f)

        if args.debug:
            docs = docs[:10]

        # Doc embeddings
        doc_data = torch.load(emb_dir / f"{split}_doc_embs.pt", map_location="cpu")
        if isinstance(doc_data, dict):
            doc_embs = doc_data["embeddings"]
        else:
            doc_embs = doc_data

        # Section embeddings (optional)
        sec_path = emb_dir / f"{split}_section_embs.pt"
        sec_embs_all, sec_doc_indices = None, None
        if sec_path.exists():
            sec_data = torch.load(sec_path, map_location="cpu")
            if isinstance(sec_data, dict):
                sec_embs_all = sec_data.get("section_embs", None)
                sec_doc_indices = sec_data.get("section_doc_indices", None)

        graphs = []

        for doc in tqdm(docs, desc=split):
            doc_idx = doc["doc_id"]

            # doc embedding
            if doc_idx < doc_embs.size(0):
                doc_emb = doc_embs[doc_idx]
            else:
                doc_emb = torch.zeros(label_embs.size(1))

            # section embeddings for this doc
            if sec_embs_all is not None and sec_doc_indices is not None:
                mask = [j for j, di in enumerate(sec_doc_indices) if di == doc_idx]
                if mask:
                    sec_embs = sec_embs_all[mask]
                else:
                    sec_embs = None
            else:
                sec_embs = None

            g = build_document_graph(
                doc,
                doc_emb,
                sec_embs,
                label_embs,
                label_adj=label_adj,
            )
            graphs.append(g)

        print("\nFirst 3 graphs:")
        for i in range(min(3, len(graphs))):
            print(f"Graph {i}:")
            ok = validate_graph(graphs[i])
            print(" Valid:", ok)
            print_graph_stats(graphs[i])
            print("-" * 40)

        out_path = graph_dir / f"{split}_graphs.pt"
        torch.save(graphs, out_path)
        print(f"Saved {len(graphs)} graphs to {out_path}")


if __name__ == "__main__":
    main()