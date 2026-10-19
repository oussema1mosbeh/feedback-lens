"""Step 2: sentiment baseline - word counting + a linear model.

Run with:  .venv/Scripts/python.exe 02_baseline.py
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from common import load

train, test = load("train"), load("test")
print(f"Always say 'negative': {(test.sentiment == 'negative').mean():.1%} accuracy\n")

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
model = LogisticRegression(max_iter=2000, C=5, class_weight="balanced")
model.fit(vec.fit_transform(train.clean), train.sentiment)
pred = model.predict(vec.transform(test.clean))

print("TF-IDF + logistic regression")
print(f"  accuracy: {accuracy_score(test.sentiment, pred):.1%}")
print(f"  macro F1: {f1_score(test.sentiment, pred, average='macro'):.1%}\n")
print(classification_report(test.sentiment, pred, digits=3))
