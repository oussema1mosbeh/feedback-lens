"""Step 1: what's in the feedback?

Run with:  .venv/Scripts/python.exe 01_explore.py
"""
from common import load

df = load()
print(f"{len(df):,} messages to {df.airline.nunique()} airlines, "
      f"{df.created.min():%d %b %Y} to {df.created.max():%d %b %Y}\n")

print("Sentiment (labelled by people):")
for s, share in df.sentiment.value_counts(normalize=True).items():
    print(f"  {s:>8}: {share:5.1%}  {'#' * int(share * 60)}")

print("\nShare of negative messages per airline:")
neg = df.groupby("airline").sentiment.apply(lambda s: (s == "negative").mean()).sort_values(ascending=False)
for a, share in neg.items():
    print(f"  {a:>15}: {share:5.1%}  ({(df.airline == a).sum():,} messages)")

print("\nWhy people complain (reason given by the labellers):")
for r, n in df.reason.value_counts().items():
    print(f"  {r:>28}: {n:5,}")

print("\nA few messages:")
for _, row in df.sample(5, random_state=4).iterrows():
    print(f"  [{row.sentiment:>8}] {row.text[:110]}")
