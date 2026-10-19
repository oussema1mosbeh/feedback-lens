"""Step 5: final sentiment test, then export everything the browser needs.

Run with:  .venv/Scripts/python.exe 05_evaluate_export.py
"""
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import SENTIMENTS, load

train, test = load("train"), load("test")
y = test.sentiment.map({s: i for i, s in enumerate(SENTIMENTS)}).values
OUT = Path("demo/models")
rows = []

# --- 1. Baseline on the test set -------------------------------------------------------------
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
base = LogisticRegression(max_iter=2000, C=5, class_weight="balanced").fit(vec.fit_transform(train.clean), train.sentiment)
bp = base.predict(vec.transform(test.clean))
rows.append(("TF-IDF baseline", "-", accuracy_score(test.sentiment, bp), f1_score(test.sentiment, bp, average="macro")))

# --- 2. Each transformer: export to ONNX, quantize to 8-bit, score the browser version ----------
for name in ["mini", "distilbert"]:
    src, out = Path("models") / f"sentiment-{name}", OUT / f"sentiment-{name}"
    if out.exists():
        shutil.rmtree(out)
    (out / "onnx").mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    sample = tok(["an example"], return_tensors="pt")
    names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in sample]

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args):
            return self.m(**dict(zip(names, args))).logits

    fp32, int8 = out / "onnx" / "model.onnx", out / "onnx" / "model_quantized.onnx"
    torch.onnx.export(Wrapper(model), tuple(sample[k] for k in names), fp32, input_names=names, output_names=["logits"],
                      opset_version=17, dynamo=False,
                      dynamic_axes={**{k: {0: "batch", 1: "sequence"} for k in names}, "logits": {0: "batch"}})
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    fp32.unlink()
    model.config.save_pretrained(out)
    tok.save_pretrained(out)

    sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    inputs = [i.name for i in sess.get_inputs()]
    pred = []
    for i in range(0, len(test), 64):
        enc = tok(list(test.clean[i:i + 64]), padding=True, truncation=True, max_length=64, return_tensors="np")
        pred.append(sess.run(None, {k: enc[k].astype(np.int64) for k in inputs})[0].argmax(1))
    pred = np.concatenate(pred)
    rows.append((f"{name} (browser, 8-bit)", f"{int8.stat().st_size / 1e6:.0f} MB",
                 accuracy_score(y, pred), f1_score(y, pred, average="macro")))

print(f"\nSentiment on {len(test):,} test messages:")
print(f"  {'model':<26} {'size':>6} {'accuracy':>9} {'macro F1':>9}")
for r in rows:
    print(f"  {r[0]:<26} {r[1]:>6} {r[2]:>9.1%} {r[3]:>9.1%}")

# --- 3. The embedding model for topics, in its ready-made browser format -------------------------
emb_out = OUT / "embedder"
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model_quantized.onnx"]:
    p = hf_hub_download("Xenova/all-MiniLM-L6-v2", f)
    (emb_out / Path(f).parent).mkdir(parents=True, exist_ok=True)
    shutil.copy(p, emb_out / f)
print(f"\nEmbedding model: {(emb_out / 'onnx/model_quantized.onnx').stat().st_size / 1e6:.0f} MB -> {emb_out}/")
