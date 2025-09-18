from typing import List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def get_embeddings(texts: List[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb)


def kmeans_cluster(embeddings: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(embeddings)
    return labels


def topic_samples(df: pd.DataFrame, text_col: str, label_col: str, k: int = 5) -> pd.DataFrame:
    rows = []
    for t in sorted(df[label_col].unique()):
        sample = df[df[label_col] == t][text_col].head(k).tolist()
        rows.append({"topic": t, "samples": sample})
    return pd.DataFrame(rows)
