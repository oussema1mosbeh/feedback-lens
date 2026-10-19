"""Step 0: download real customer feedback and prepare it.

Twitter US Airline Sentiment (CrowdFlower / Figure Eight "Data for Everyone"),
CC BY-NC-SA 4.0 - fine for a portfolio demo; for client work you use the client's data.
14,640 messages to 6 US airlines (Feb 2015). People labelled each one positive,
neutral or negative, and gave a reason for every complaint ("Late Flight", ...).

Run with:  .venv/Scripts/python.exe 00_download.py
"""
import numpy as np
from datasets import load_dataset

from common import DATA, clean

DATA.mkdir(exist_ok=True)
df = load_dataset("osanseviero/twitter-airline-sentiment")["train"].to_pandas()

df = df.rename(columns={"airline_sentiment": "sentiment", "airline_sentiment_confidence": "confidence",
                        "negativereason": "reason", "tweet_created": "created"})
df = df[["tweet_id", "airline", "created", "text", "sentiment", "confidence", "reason"]]
df["created"] = df.created.str.slice(0, 19)                 # "2015-02-24 11:35:52 -0800" -> local time

# Data leak fix: whoever prepared this dataset replaced "cancel" with "Cancelled Flight" and
# "late" with "Late Flight" in the text ("Cancelled Flightled", "Late Flightr" = later).
# Those are the complaint category names! Undo it so models see the customers' real words.
df["text"] = (df.text.str.replace("Cancelled Flight", "Cancel", case=False, regex=False)
                     .str.replace("Late Flight", "late", case=False, regex=False))
df["clean"] = df.text.map(clean)
df = df[df.clean.str.len() > 0].drop_duplicates("clean").reset_index(drop=True)

# 80% train / 10% validation / 10% test, shuffled with a fixed seed
rng = np.random.default_rng(42)
r = rng.random(len(df))
df["split"] = np.where(r < .8, "train", np.where(r < .9, "validation", "test"))
df.to_csv(DATA / "feedback.csv", index=False)

print(f"{len(df):,} messages (duplicates removed) -> data/feedback.csv")
print(df.split.value_counts().to_string())
