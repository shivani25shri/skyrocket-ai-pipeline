# evaluate.py
# ------------------------------------------------------------------------------
# Combines:
#   - Existing plotting/category normalization utilities
#   - New LLM response generation + quality evaluation
# ------------------------------------------------------------------------------

import os
from typing import List, Any, Dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as stats

from pipeline import run_with_llm_generation


# -------------------------------------------------------------------
# Existing helpers: normalize category, plots, export
# -------------------------------------------------------------------

def normalize_category(row: pd.Series) -> str:
    cat = str(row.get("category", "")).upper()
    sub = str(row.get("Sub Category", "")).lower()
    if "refund" in sub:
        return "REFUND"
    if "cancel" in sub:
        return "CANCEL"
    if "ship" in sub or "delivery" in sub:
        return "SHIPPING"
    if "account" in sub or "password" in sub:
        return "ACCOUNT"
    return cat if cat else "OTHER"


def add_normalized_category(responses_df: pd.DataFrame) -> pd.DataFrame:
    out = responses_df.copy()
    out["normalized_category"] = out.apply(normalize_category, axis=1)
    return out


def plot_length_distribution(queries_df: pd.DataFrame, outputs_dir: str):
    plt.figure(figsize=(7, 4))
    sns.histplot(queries_df["length"], bins=30)
    plt.title("Distribution of Query Lengths")
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "query_length_distribution.png"))
    plt.close()


def plot_sentiment_by_category(responses_df: pd.DataFrame, outputs_dir: str):
    plt.figure(figsize=(9, 5))
    sns.countplot(
        x="normalized_category",
        hue="sentiment",
        data=responses_df,
        order=responses_df["normalized_category"].value_counts().index,
    )
    plt.title("Sentiment Distribution by Normalized Category")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "sentiment_by_category.png"))
    plt.close()


def export_topic_samples(topic_df: pd.DataFrame, outputs_dir: str):
    topic_df.to_json(
        os.path.join(outputs_dir, "topic_samples.json"),
        orient="records",
        indent=2,
    )


# -------------------------------------------------------------------
# New: LLM response quality evaluation
# -------------------------------------------------------------------

def _avg(vals: List[float]) -> float:
    return round(float(sum(vals) / max(1, len(vals))), 3)


def evaluate_llm_quality(llm_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate LLM-as-judge metrics over the dataset.
    Returns averages + distributions.
    """
    if not llm_scores:
        return {
            "avg_relevance": 0.0,
            "avg_helpfulness": 0.0,
            "avg_tone": 0.0,
            "n": 0,
            "dist": {"relevance": {}, "helpfulness": {}, "tone": {}},
        }

    rel = [int(s.get("relevance", 0)) for s in llm_scores]
    helpf = [int(s.get("helpfulness", 0)) for s in llm_scores]
    tone = [int(s.get("tone", 0)) for s in llm_scores]

    def _dist(xs: List[int]) -> Dict[int, int]:
        d: Dict[int, int] = {}
        for x in xs:
            d[x] = d.get(x, 0) + 1
        return {k: d[k] for k in sorted(d)}

    summary = {
        "avg_relevance": _avg(rel),
        "avg_helpfulness": _avg(helpf),
        "avg_tone": _avg(tone),
        "n": len(llm_scores),
        "dist": {
            "relevance": _dist(rel),
            "helpfulness": _dist(helpf),
            "tone": _dist(tone),
        },
    }
    return summary


def print_overall_report(artifacts: Dict[str, Any]) -> None:
    """
    Console-friendly report tying together the new LLM metrics with existing steps.
    """
    print("\n=== LLM Response Quality (LLM-as-judge) ===")
    q = evaluate_llm_quality(artifacts.get("llm_scores", []))
    print(f"Items: {q['n']}")
    print(f"Avg Relevance:   {q['avg_relevance']}   Dist: {q['dist']['relevance']}")
    print(f"Avg Helpfulness: {q['avg_helpfulness']} Dist: {q['dist']['helpfulness']}")
    print(f"Avg Tone:        {q['avg_tone']}        Dist: {q['dist']['tone']}")

    if "query_clusters" in artifacts:
        unique_clusters = len(set(artifacts["query_clusters"]))
        print("\n=== Clustering Snapshot ===")
        print(f"Unique clusters: {unique_clusters}")

    if "response_sentiment" in artifacts:
        print("\n=== Response Sentiment (sample of 5) ===")
        sample = artifacts["response_sentiment"][:5]
        for i, s in enumerate(sample, 1):
            print(f"{i:02d}. {s}")

    print("\n=== Example pairs (first 3) ===")
    for i in range(min(3, len(artifacts["queries"]))):
        print(f"Q{i+1}: {artifacts['queries'][i]}")
        print(f"A{i+1}: {artifacts['responses'][i]}")
        print(f"S{i+1}: {artifacts['llm_scores'][i]}")
        print("---")


# -------------------------------------------------------------------
# Main entrypoint
# -------------------------------------------------------------------

if __name__ == "__main__":
    artifacts = run_with_llm_generation()
    print_overall_report(artifacts)
