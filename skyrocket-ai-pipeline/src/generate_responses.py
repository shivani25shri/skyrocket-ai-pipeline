# generate_responses.py
# ------------------------------------------------------------------------------
# Generates one customer-support-style response per query using gpt-4o,
# then LLM-scores each response (relevance, helpfulness, tone).
# Returns everything in memory (no writes to Excel).
# ------------------------------------------------------------------------------

from __future__ import annotations

import os
import json
from typing import Dict, List, Any, Iterable, Optional, Tuple

import yaml
import pandas as pd

from llm_utils import generate_chat_response, score_chat_response


def _load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_queries_from_excel(
    config_path: str = "config.yaml",
    excel_path: Optional[str] = None,
    queries_sheet: Optional[str] = None,
    text_column: str = "query",
) -> List[str]:
    """
    Load queries from Excel based on config defaults unless explicit overrides provided.
    By default, reads config['data']['excel_path'] (if nested) or top-level 'excel_path'.
    """
    cfg = _load_config(config_path)

    # Backward-compatible: allow both flat and nested structures
    excel_file = (
        excel_path
        or cfg.get("excel_path")
        or (cfg.get("data", {}) or {}).get("excel_path")
    )
    sheet = (
        queries_sheet
        or cfg.get("queries_sheet")
        or (cfg.get("data", {}) or {}).get("queries_sheet")
        or "Queries"
    )

    if not excel_file:
        raise ValueError(
            "Excel path not found. Provide excel_path or set it in config.yaml under "
            "'excel_path' or 'data.excel_path'."
        )

    df = pd.read_excel(excel_file, sheet_name=sheet)
    if text_column not in df.columns:
        # Try a few typical names
        for candidate in ("query", "Query", "text", "Text", "customer_query"):
            if candidate in df.columns:
                text_column = candidate
                break
        else:
            raise KeyError(
                f"Could not find a text column in '{sheet}'. "
                "Expected 'query' or a common variant."
            )

    queries: List[str] = [str(x).strip() for x in df[text_column].astype(str).tolist()]
    return queries


def generate_for_queries(
    queries: Iterable[str],
    model: str = "gpt-4o",
    response_temperature: float = 0.2,
    judge_temperature: float = 0.0,
    max_tokens: int = 350,
) -> List[Dict[str, Any]]:
    """
    For each query:
      - Generate a support-style response with gpt-4o
      - Score with gpt-4o-as-judge
    Returns a list of dicts with keys: query, response, scores(dict).
    """
    results: List[Dict[str, Any]] = []

    for q in queries:
        # 1) Generate response
        response_text = generate_chat_response(
            query=q,
            model=model,
            temperature=response_temperature,
            max_tokens=max_tokens,
        )

        # 2) Score the response (LLM-as-judge)
        scores = score_chat_response(
            query=q,
            response=response_text,
            model=model,
            temperature=judge_temperature,
        )

        results.append(
            {
                "query": q,
                "response": response_text,
                "scores": scores,  # {'relevance': int, 'helpfulness': int, 'tone': int, 'explanation': str}
            }
        )

    return results


if __name__ == "__main__":
    """
    Example CLI run:
        OPENAI_API_KEY=sk-... python generate_responses.py
    This reads queries from Excel (per config) and prints a small sample.
    """
    cfg_path = "config.yaml"
    qs = load_queries_from_excel(cfg_path)
    out = generate_for_queries(qs)

    print(f"Generated {len(out)} responses.")
    print(json.dumps(out[:3], indent=2, ensure_ascii=False))
