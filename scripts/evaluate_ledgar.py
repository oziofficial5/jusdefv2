import sys, re
import torch
import torch.nn as nn
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, '.')

LABEL_TO_INT = {'AFF': 0, 'NEG': 1, 'EXC': 2, 'OVR': 3}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

class OperatorDetector(nn.Module):
    def __init__(self, backbone_name, n=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(h, n)

    def forward(self, input_ids, attention_mask):
        o = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(self.dropout(o.last_hidden_state[:, 0]))

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model state
    checkpoint_path = 'outputs/checkpoints/operator_detector_neural.pt'
    print(f"Loading checkpoint from {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location=device)
    backbone_name = state['config']['backbone']

    tok = AutoTokenizer.from_pretrained(backbone_name)
    model = OperatorDetector(backbone_name).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()

    # Load dataset
    print('Loading LEDGAR dataset...')
    ds = load_dataset('lex_glue', 'ledgar', split='test')
    print(f'Test set: {len(ds)} instances')

    # Extract and split sentences
    sents = []
    for inst in ds:
        text = inst.get('text', inst.get('contract', ''))
        # Split on periods or semicolons
        sents.extend([s.strip() for s in re.split(r'(?<=[.;])\s+', text) if s.strip()])

    sents = sents[:30000]
    print(f'Evaluating {len(sents)} sentences...')

    # Inference
    with torch.no_grad():
        preds = []
        batch_size = 64
        for k in range(0, len(sents), batch_size):
            batch = sents[k:k+batch_size]
            enc = tok(batch, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
            logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            preds.extend(logits.argmax(-1).cpu().tolist())

    # Metrics calculation
    c = Counter(INT_TO_LABEL[p] for p in preds)
    t = sum(c.values())

    print('\nLEDGAR density:')
    for op in ['AFF', 'NEG', 'EXC', 'OVR']:
        n = c.get(op, 0)
        print(f'  {op}: {n} ({n/t*100:.2f}%)')

    nonaff = t - c.get('AFF', 0)
    print(f'\nNON-AFF: {nonaff/t*100:.2f}%')
    print('Compare: EUR-Lex 0.71%, ECtHR 0.58%')

    if nonaff / t > 0.03:
        print('==> LEDGAR pivot justified')
    elif nonaff / t > 0.01:
        print('==> Borderline')
    else:
        print('==> LEDGAR no denser, commit to negative result')

if __name__ == "__main__":
    main()
