## LegalBERT cosine baseline (zero-shot)



- Embeddings: precomputed LegalBERT CLS embeddings for validation/test documents and label descriptions.

- Similarity: cosine(doc_emb, label_emb).

- Threshold: single global cosine threshold tuned on the validation set.



Validation (n=5000):

- Macro-F1: 0.0877

- Micro-F1: 0.0935



Test (n=5000):

- Macro-F1: 0.0906

- Micro-F1: 0.0967



Config:

- Code: scripts/run_cosine_baseline.py

- Log: outputs/logs/baseline_legalbert_cosine.json (generated on Ampere)

