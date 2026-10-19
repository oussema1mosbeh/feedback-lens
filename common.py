"""Shared helpers, so every step cleans text and splits data the same way."""
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
SENTIMENTS = ["negative", "neutral", "positive"]


def clean(text):
    """Text the models see. Every message starts with the airline's handle (@united...).
    If we kept those, the AI would group messages by airline instead of by topic."""
    text = re.sub(r"@\w+", " ", str(text))          # @mentions
    text = re.sub(r"https?://\S+", " ", text)       # links
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", text).strip()


def load(split=None):
    df = pd.read_csv(DATA / "feedback.csv", parse_dates=["created"])
    return df if split is None else df[df.split == split].reset_index(drop=True)
