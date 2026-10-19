"""Step 4: discover complaint topics automatically - no labels needed.

1. Embeddings: each message becomes 384 numbers that capture its MEANING
   ("flight delayed 3 hours" and "still stuck at the gate" end up close together).
2. Clustering (k-means): group messages whose meanings are close.
3. Naming: each group is named by the words most typical of it (class-based TF-IDF).
4. Honest check: we never show the human "reason" labels to the algorithm, but we
   use them afterwards to measure how well the groups match real complaint types.

Run with:  .venv/Scripts/python.exe 04_topics.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from common import load

df = load()
neg = df[df.sentiment == "negative"].reset_index(drop=True)
print(f"{len(neg):,} complaints to organise\n")

# --- 1. Embeddings -------------------------------------------------------------------------
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = encoder.encode(list(neg.clean), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
print(f"Each complaint is now {emb.shape[1]} numbers.\n")

# --- 2. How many topics? Let the data decide (silhouette = how well separated the groups are)
print("Choosing the number of topics:")
scores = {}
for k in range(6, 15):
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(emb)
    scores[k] = silhouette_score(emb, labels, sample_size=4000, random_state=0)
    print(f"  k={k:2d}  silhouette {scores[k]:.3f}")
K = max(scores, key=scores.get)
print(f"-> best: {K} topics\n")
km = KMeans(K, n_init=20, random_state=0).fit(emb)
neg["topic"] = km.labels_

# --- 3. Name each topic by its most distinctive words (class-based TF-IDF) ------------------
cv = CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
counts = cv.fit_transform(neg.clean)
words = np.array(cv.get_feature_names_out())
per_topic = np.vstack([np.asarray(counts[neg.topic.values == t].sum(0)).ravel() for t in range(K)]).astype(float)
tf = per_topic / per_topic.sum(1, keepdims=True)
idf = np.log(1 + per_topic.sum(1).mean() / (per_topic.sum(0) + 1))
ctfidf = tf * idf

topics = []
for t in range(K):
    members = neg[neg.topic == t]
    top_words = [w for w in words[ctfidf[t].argsort()[::-1]] if len(w) > 2][:5]
    dist = emb[members.index] @ km.cluster_centers_[t]
    examples = members.text.iloc[dist.argsort()[::-1][:3]].tolist()       # closest to the centre = most typical
    reason, n = Counter(members.reason.fillna("?")).most_common(1)[0]
    topics.append({"id": t, "size": len(members), "words": top_words, "examples": examples,
                   "human_reason": reason, "purity": n / len(members)})

print("Topics found (biggest first), compared with the human label that dominates each:")
for tp in sorted(topics, key=lambda x: -x["size"]):
    print(f"  {tp['size']:5,} | {', '.join(tp['words'][:4]):<52} | humans said: {tp['human_reason']} ({tp['purity']:.0%})")
    print(f"        e.g. {tp['examples'][0][:100]}")

# --- 4. Honest check against the human reasons ------------------------------------------------
known = neg.reason.notna() & (neg.reason != "Can't Tell")
truth = neg.reason[known]
print(f"\nAgreement with the human complaint reasons ({known.sum():,} labelled complaints):")
print("  (ARI and NMI: 0 = random grouping, 1 = identical to the human labels)")

tfidf = TfidfVectorizer(stop_words="english", min_df=3).fit_transform(neg.clean)
words_km = KMeans(K, n_init=10, random_state=0).fit_predict(tfidf)
for name, lab in [("word counting (TF-IDF) + k-means", words_km), ("meaning (embeddings) + k-means", neg.topic.values)]:
    print(f"  {name:<36} ARI {adjusted_rand_score(truth, lab[known]):.3f}   NMI {normalized_mutual_info_score(truth, lab[known]):.3f}")

Path("models").mkdir(exist_ok=True)
Path("models/topics.json").write_text(json.dumps({"k": K, "silhouette": scores, "topics": topics}, indent=1))
np.save("models/topic_centres.npy", km.cluster_centers_)
print("\nSaved models/topics.json")
