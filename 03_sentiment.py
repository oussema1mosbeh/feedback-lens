"""Step 3: fine-tune a transformer to read customer sentiment.

Same recipe as Mood Reader: a pretrained transformer + a new 3-way layer
(negative / neutral / positive), class weights so the rarer classes count.

Run with:  .venv/Scripts/python.exe 03_sentiment.py mini
           .venv/Scripts/python.exe 03_sentiment.py distilbert
"""
import json
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import SENTIMENTS, load

PRESETS = {
    "distilbert": dict(hub="distilbert-base-uncased", epochs=3, lr=4e-5, batch=32),
    "mini": dict(hub="google/bert_uncased_L-4_H-256_A-4", epochs=8, lr=1e-4, batch=32),
}
name = sys.argv[1] if len(sys.argv) > 1 else "mini"
cfg = PRESETS[name]
MAX_LEN = 64
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
to_id = {s: i for i, s in enumerate(SENTIMENTS)}

train, val = load("train"), load("validation")
tok = AutoTokenizer.from_pretrained(cfg["hub"])
model = AutoModelForSequenceClassification.from_pretrained(
    cfg["hub"], num_labels=3, id2label=dict(enumerate(SENTIMENTS)), label2id=to_id).to(device)
print(f"{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")


def encode(df):
    enc = tok(list(df.clean), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor([to_id[s] for s in df.sentiment])


tr, va = encode(train), encode(val)
counts = torch.bincount(tr[2], minlength=3).float()
class_weight = ((counts.sum() / (3 * counts)) ** 0.5).to(device)


@torch.no_grad()
def evaluate(ids, mask, y):
    model.eval()
    p = torch.cat([model(input_ids=ids[i:i + 256].to(device), attention_mask=mask[i:i + 256].to(device)).logits.argmax(1).cpu()
                   for i in range(0, len(ids), 256)])
    return accuracy_score(y, p), f1_score(y, p, average="macro")


opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
steps = cfg["epochs"] * (len(tr[0]) // cfg["batch"])
sched = get_linear_schedule_with_warmup(opt, int(steps * .06), steps)
scaler = torch.amp.GradScaler(enabled=device == "cuda")
out_dir = Path("models") / f"sentiment-{name}"
best, history, start = 0, [], time.time()

for epoch in range(1, cfg["epochs"] + 1):
    model.train()
    order = torch.randperm(len(tr[0]))
    total = 0
    for i in range(len(order) // cfg["batch"]):
        idx = order[i * cfg["batch"]:(i + 1) * cfg["batch"]]
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(input_ids=tr[0][idx].to(device), attention_mask=tr[1][idx].to(device)).logits
        loss = torch.nn.functional.cross_entropy(logits.float(), tr[2][idx].to(device), weight=class_weight)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item()
    acc, f1 = evaluate(*va)
    history.append({"epoch": epoch, "loss": total / (len(order) // cfg["batch"]), "val_acc": acc, "val_f1": f1})
    saved = ""
    if f1 > best:
        best = f1
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        saved = "  <- best so far, saved"
    print(f"epoch {epoch} | loss {history[-1]['loss']:.3f} | val accuracy {acc:.1%} | val macro F1 {f1:.1%}{saved}")

(out_dir / "training.json").write_text(json.dumps({"hub": cfg["hub"], "seconds": round(time.time() - start), "history": history}, indent=1))
print(f"\nBest validation macro F1: {best:.1%}  ({time.time() - start:.0f}s) -> {out_dir}/")
