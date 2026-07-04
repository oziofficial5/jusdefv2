"""
Preprocess ContractNLI for the operator-aware NLI experiment (Pivot B).

ContractNLI (Koreeda & Manning, 2021): each contract document is annotated against
17 fixed hypotheses with a 3-way label {NotMentioned, Entailment, Contradiction}.
Exceptions frequently flip Entailment<->Contradiction ("shall not disclose, EXCEPT
to affiliates"), so this is a task where defeasible operators can determine the
label --- the natural testbed for the JusDef operator prior.

Download the dataset from https://stanfordnlp.github.io/contract-nli/ and place
train.json / dev.json / test.json in --raw_dir (default data/contract_nli_raw/).

For each document we sentence-split, LegalBERT-embed each sentence, and detect a
per-sentence operator with the EUR-Lex-trained neural detector (same as LEDGAR).
Each hypothesis is embedded once. Output per split:
  { "docs": [{"embeddings": Tensor[Ns,768], "operators": [Ns]}],
    "hyp_emb": {hyp_key: Tensor[768]},
    "examples": [{"doc": doc_idx, "hyp": hyp_key, "label": int}] }
Label map: NotMentioned=0, Entailment=1, Contradiction=2.

Usage:
  python scripts/preprocess_contractnli.py            # full
  python scripts/preprocess_contractnli.py --debug    # 30 docs/split smoke
"""
import os, re, sys, json, argparse, pickle
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

LABEL_MAP = {"NotMentioned": 0, "Entailment": 1, "Contradiction": 2}
SPLIT_FILES = {"train": "train.json", "validation": "dev.json", "test": "test.json"}


class OperatorClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(self.dropout(out.last_hidden_state[:, 0]))


def sentence_split(text):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


@torch.no_grad()
def embed_and_detect(sentences, tok, enc, det, device, bs=64, max_len=128):
    embs, ops = [], []
    for i in range(0, len(sentences), bs):
        b = sentences[i:i + bs]
        e = tok(b, truncation=True, padding="max_length", max_length=max_len,
                return_tensors="pt")
        ii, am = e["input_ids"].to(device), e["attention_mask"].to(device)
        embs.append(enc(input_ids=ii, attention_mask=am).last_hidden_state[:, 0].cpu())
        ops.append(det(ii, am).argmax(-1).cpu())
    return torch.cat(embs), torch.cat(ops)


@torch.no_grad()
def embed_texts(texts, tok, enc, device, max_len=64):
    e = tok(texts, truncation=True, padding="max_length", max_length=max_len,
            return_tensors="pt")
    return enc(input_ids=e["input_ids"].to(device),
               attention_mask=e["attention_mask"].to(device)).last_hidden_state[:, 0].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/contract_nli_raw")
    ap.add_argument("--out_dir", default="data/processed_contractnli")
    ap.add_argument("--detector_ckpt", default="outputs/checkpoints/operator_detector_neural.pt")
    ap.add_argument("--max_sents", type=int, default=128, help="cap sentences per doc")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.detector_ckpt, map_location=device)
    backbone = state["config"]["backbone"]
    print(f"Device {device} | detector backbone {backbone}", flush=True)
    tok = AutoTokenizer.from_pretrained(backbone)
    enc = AutoModel.from_pretrained(backbone).to(device).eval()
    det = OperatorClassifier(backbone, 4).to(device).eval()
    det.load_state_dict(state["state_dict"])

    for split, fname in SPLIT_FILES.items():
        path = raw / fname
        if not path.exists():
            print(f"[skip] {path} not found (download ContractNLI JSON there)"); continue
        data = json.load(open(path, encoding="utf-8"))
        documents = data["documents"]
        labels = data["labels"]  # {hyp_key: {"hypothesis": text}}
        if args.debug:
            documents = documents[:30]
        print(f"\n=== {split}: {len(documents)} docs, {len(labels)} hypotheses ===", flush=True)

        # hypothesis embeddings (once per split)
        hyp_keys = list(labels.keys())
        hyp_texts = [labels[k]["hypothesis"] for k in hyp_keys]
        hyp_embs = embed_texts(hyp_texts, tok, enc, device)
        hyp_emb = {k: hyp_embs[i] for i, k in enumerate(hyp_keys)}

        docs, examples = [], []
        nonaff = tot = 0
        for d in tqdm(documents, desc=split):
            sents = sentence_split(d["text"])[:args.max_sents]
            if not sents:
                continue
            embs, ops = embed_and_detect(sents, tok, enc, det, device)
            di = len(docs)
            docs.append({"embeddings": embs, "operators": ops.tolist()})
            nonaff += int((ops != 0).sum()); tot += len(ops)
            # one example per annotated hypothesis
            aset = d.get("annotation_sets", [{}])
            anns = aset[0].get("annotations", {}) if aset else {}
            for k in hyp_keys:
                a = anns.get(k)
                if a is None:
                    continue
                choice = a.get("choice", "NotMentioned")
                if choice not in LABEL_MAP:
                    continue
                examples.append({"doc": di, "hyp": k, "label": LABEL_MAP[choice]})

        pickle.dump({"docs": docs, "hyp_emb": hyp_emb, "examples": examples},
                    open(out / f"{split}_processed.pkl", "wb"))
        from collections import Counter
        lc = Counter(e["label"] for e in examples)
        print(f"  {len(docs)} docs, {len(examples)} examples | "
              f"labels NM/Ent/Con = {lc[0]}/{lc[1]}/{lc[2]} | "
              f"non-AFF density {nonaff/max(tot,1)*100:.1f}%", flush=True)
    print("\nDone. Output in", out)


if __name__ == "__main__":
    main()
