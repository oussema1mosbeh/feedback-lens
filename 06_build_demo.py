"""Step 6: pre-analyse the sample dataset so the dashboard opens instantly.

Every message gets the model's sentiment; every complaint gets a topic.
(Uploaded files are analysed live in the browser with the same models.)

Run with:  .venv/Scripts/python.exe 06_build_demo.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import SENTIMENTS, load

df = load().sort_values("created").reset_index(drop=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- sentiment for every message, with the accurate model ------------------------------------
tok = AutoTokenizer.from_pretrained("models/sentiment-distilbert")
model = AutoModelForSequenceClassification.from_pretrained("models/sentiment-distilbert").to(device).eval()
probs = []
with torch.no_grad():
    for i in range(0, len(df), 256):
        enc = tok(list(df.clean[i:i + 256]), padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        probs.append(model(**enc).logits.softmax(1).cpu().numpy())
probs = np.concatenate(probs)
df["pred"] = probs.argmax(1)
df["conf"] = (probs.max(1) * 100).round().astype(int)

# --- topics for every complaint: nearest of the topic centres found in step 4 -----------------
info = json.loads(Path("models/topics.json").read_text())
centres = np.load("models/topic_centres.npy")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
neg = df.pred == SENTIMENTS.index("negative")
emb = encoder.encode(list(df.clean[neg]), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
sims = emb @ (centres / np.linalg.norm(centres, axis=1, keepdims=True)).T
df["topic"] = -1
df.loc[neg, "topic"] = sims.argmax(1)

# Coherence = how tightly a topic's messages sit around its centre. The loosest topics
# are general venting ("worst airline ever") rather than one specific problem.
coherence = {t: float(sims[sims.argmax(1) == t, t].mean()) for t in range(len(centres))}
# Checked by eye: only the loosest topic is pure venting. (Taking the 2 loosest wrongly
# caught "email, website", a real problem, so a human review of the rule was needed.)
vague = sorted(coherence, key=coherence.get)[:1]
print("Topic coherence (lowest = vaguest):")
for t in sorted(coherence, key=coherence.get):
    print(f"  {coherence[t]:.3f}  {', '.join(info['topics'][t]['words'][:4])}{'   <- general frustration' if t in vague else ''}")

# Readable names for the sample data. The AI proposes the keywords; I wrote short titles
# from them (the app lets users rename topics the same way on their own data).
NAMES = {"delayed": "Delays", "bag": "Baggage", "cancelled": "Cancellations", "customer": "Customer service",
         "hold": "Long hold times", "phone": "Phone calls & callbacks", "ticket": "Refunds, tickets & miles",
         "email": "Website, email & app", "gate": "Gate agents & boarding", "seat": "Seats & reservations",
         "weather": "Stranded & weather", "flight": "Getting help with a flight", "airline": "Overall bad experience"}
topics = []
for tp in info["topics"]:
    t = tp["id"]
    name = "General frustration" if t in vague else NAMES.get(tp["words"][0], tp["words"][0].title())
    # "general" = venting with nothing specific to fix: listed, but never headlined as the #1 issue
    topics.append({"id": t, "name": name, "keywords": tp["words"][:5], "coherence": round(coherence[t], 3),
                   "general": name in ("General frustration", "Overall bad experience")})
# merge the vague topics into one "General frustration" group
keep = vague[0]
df.loc[df.topic.isin(vague), "topic"] = keep
topics = [tp for tp in topics if tp["id"] not in vague[1:]]

# --- compact JSON for the dashboard -------------------------------------------------------------
airlines = sorted(df.airline.unique())
t0 = df.created.min()
rows = [[airlines.index(r.airline), int((r.created - t0).total_seconds() // 60), int(r.pred), int(r.conf), int(r.topic), r.text]
        for r in df.itertuples()]
out = {"meta": {"source": "Twitter US Airline Sentiment (CC BY-NC-SA 4.0)", "label": "Customer messages to 6 US airlines",
                "start": t0.isoformat(), "messages": len(df), "model": "DistilBERT (fine-tuned), all-MiniLM-L6-v2 topics"},
       "companies": airlines, "sentiments": SENTIMENTS, "topics": topics, "rows": rows}
Path("demo/data").mkdir(parents=True, exist_ok=True)
Path("demo/data/airlines.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
size = Path("demo/data/airlines.json").stat().st_size / 1e6
print(f"\n{len(rows):,} messages, {len(topics)} topics -> demo/data/airlines.json ({size:.1f} MB)")
print("Sentiment (model):", {s: f"{(df.pred == i).mean():.1%}" for i, s in enumerate(SENTIMENTS)})
