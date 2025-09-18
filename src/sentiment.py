from typing import List
import pandas as pd
from transformers import pipeline

# Use binary sentiment model
BINARY_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

def batch_sentiment(
    texts: List[str],
    model_name: str = BINARY_SENTIMENT_MODEL,
    batch_size: int = 64,
    max_length: int = 256
) -> List[str]:
    """
    Run sentiment analysis in batch using a binary model (POSITIVE/NEGATIVE only).
    """
    clf = pipeline("sentiment-analysis", model=model_name)
    results = clf(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=max_length
    )
    labels = [r["label"] for r in results]
    return labels


def add_sentiment_column(
    df: pd.DataFrame,
    text_col: str,
    model_name: str = BINARY_SENTIMENT_MODEL,
    batch_size: int = 64,
    max_length: int = 256
) -> pd.DataFrame:
    """
    Add a binary sentiment column (POSITIVE/NEGATIVE) to a DataFrame.
    """
    texts = df[text_col].astype(str).tolist()
    labels = batch_sentiment(texts, model_name, batch_size, max_length)
    out = df.copy()
    out["sentiment"] = labels
    return out
