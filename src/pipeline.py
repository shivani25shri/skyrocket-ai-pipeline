# pipeline.py
# ------------------------------------------------------------------------------
# Orchestrates the analytics pipeline with an optional LLM generation step.
# New entrypoint: `run_with_llm_generation` which:
#   1) reads queries (Excel per config)
#   2) uses generate_responses.py to produce (query, response, scores) in memory
#   3) preprocesses texts
#   4) runs embeddings / clustering / entities / sentiment on queries+responses
#   5) returns a rich artifacts dict that includes LLM quality scores
# ------------------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import yaml
import pandas as pd

# Local modules
from generate_responses import load_queries_from_excel, generate_for_queries
import preprocess as _pre
import embeddings as _emb
import sentiment as _sent
import entities as _ent


def _load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _maybe_get(cfg: Dict[str, Any], *keys: str, default=None):
    """Safely reach nested config keys."""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, {})
    return cur if cur else default


def _clean_list(texts: List[str]) -> List[str]:
    """Run your preprocessing normalization step if available."""
    if hasattr(_pre, "clean_texts"):
        return _pre.clean_texts(texts)
    if hasattr(_pre, "clean_text"):
        return [_pre.clean_text(t) for t in texts]
    # fallback no-op
    return [str(t).strip() for t in texts]


def _embed_list(texts: List[str], cfg: Dict[str, Any]) -> List[List[float]]:
    """Compute embeddings with your embeddings module based on config model."""
    model_name = _maybe_get(cfg, "models", "sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    batch_size = _maybe_get(cfg, "inference", "batch_size", default=64)
    max_len = _maybe_get(cfg, "inference", "max_length", default=256)
    if hasattr(_emb, "encode_texts"):
        return _emb.encode_texts(texts, model_name=model_name, batch_size=batch_size, max_length=max_len)
    if hasattr(_emb, "build_embeddings"):
        return _emb.build_embeddings(texts, model_name=model_name, batch_size=batch_size, max_length=max_len)
    raise RuntimeError("embeddings.py must expose encode_texts(...) or build_embeddings(...).")


def _cluster(emb, cfg: Dict[str, Any]) -> List[int]:
    """Cluster embeddings; use your module if it has a helper, else do a local KMeans."""
    if hasattr(_emb, "cluster_embeddings"):
        n = int(_maybe_get(cfg, "clustering", "n_clusters", default=10))
        rs = int(_maybe_get(cfg, "clustering", "random_state", default=42))
        return _emb.cluster_embeddings(emb, n_clusters=n, random_state=rs)

    # Fallback simple KMeans (scikit-learn)
    try:
        from sklearn.cluster import KMeans
        n = int(_maybe_get(cfg, "clustering", "n_clusters", default=10))
        rs = int(_maybe_get(cfg, "clustering", "random_state", default=42))
        labels = KMeans(n_clusters=n, random_state=rs, n_init="auto").fit_predict(emb).tolist()
        return labels
    except Exception as e:
        raise RuntimeError(
            "No cluster_embeddings(...) in embeddings.py and scikit-learn is not available."
        ) from e


def _analyze_sentiment(texts: List[str], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run sentiment classification using your sentiment.py adapter."""
    model_name = _maybe_get(cfg, "models", "sentiment", default="cardiffnlp/twitter-roberta-base-sentiment")
    batch_size = _maybe_get(cfg, "inference", "batch_size", default=64)
    max_len = _maybe_get(cfg, "inference", "max_length", default=256)
    if hasattr(_sent, "analyze_sentiment"):
        return _sent.analyze_sentiment(texts, model_name=model_name, batch_size=batch_size, max_length=max_len)
    if hasattr(_sent, "predict"):
        return _sent.predict(texts, model_name=model_name, batch_size=batch_size, max_length=max_len)
    raise RuntimeError("sentiment.py must expose analyze_sentiment(...) or predict(...).")


def _extract_entities(texts: List[str], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Entity extraction step."""
    if hasattr(_ent, "extract_entities"):
        return _ent.extract_entities(texts, taxonomy=_maybe_get(cfg, "taxonomy", "top_categories", default=None))
    if hasattr(_ent, "predict"):
        return _ent.predict(texts, taxonomy=_maybe_get(cfg, "taxonomy", "top_categories", default=None))
    # fallback no-op
    return [{"entities": []} for _ in texts]


def run_with_llm_generation(
    config_path: str = "config.yaml",
    chat_model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    End-to-end pipeline with LLM response generation + scoring (in memory).
    Returns a dict of artifacts:
    {
      "queries": [...],
      "responses": [...],
      "llm_scores": [{"relevance":..., "helpfulness":..., "tone":..., "explanation":...}, ...],
      "preprocessed_queries": [...],
      "preprocessed_responses": [...],
      "query_embeddings": [[...], ...],
      "response_embeddings": [[...], ...],
      "query_clusters": [...],
      "response_sentiment": [...],
      "query_entities": [...],
    }
    """
    cfg = _load_config(config_path)

    # 1) Load queries from Excel (config-driven)
    queries = load_queries_from_excel(config_path)

    # 2) Generate responses + LLM-as-judge scores (in-memory)
    gen = generate_for_queries(queries, model=chat_model)
    responses = [g["response"] for g in gen]
    llm_scores = [g["scores"] for g in gen]

    # 3) Preprocess
    q_clean = _clean_list(queries)
    r_clean = _clean_list(responses)

    # 4) Embeddings + clustering (on queries)
    q_emb = _embed_list(q_clean, cfg)
    q_clusters = _cluster(q_emb, cfg)

    # 5) Optional embeddings on responses (useful for similarity or analysis)
    r_emb = _embed_list(r_clean, cfg)

    # 6) Entities on queries
    q_ents = _extract_entities(q_clean, cfg)

    # 7) Sentiment on responses
    r_senti = _analyze_sentiment(r_clean, cfg)

    artifacts = {
        "queries": queries,
        "responses": responses,
        "llm_scores": llm_scores,
        "preprocessed_queries": q_clean,
        "preprocessed_responses": r_clean,
        "query_embeddings": q_emb,
        "response_embeddings": r_emb,
        "query_clusters": q_clusters,
        "query_entities": q_ents,
        "response_sentiment": r_senti,
    }
    return artifacts


if __name__ == "__main__":
    result = run_with_llm_generation()
    print(
        f"Pipeline completed: {len(result['queries'])} queries, "
        f"{len(result['responses'])} responses."
    )
