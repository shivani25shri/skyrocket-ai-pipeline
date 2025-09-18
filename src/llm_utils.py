# llm_utils.py
# ------------------------------------------------------------------------------
# Wrapper around the OpenAI API for:
#  - Generating a customer-support-style reply for a single query
#  - Scoring that reply with LLM-as-judge (relevance/helpfulness/tone)
#  - Classifying queries into categories
#  - Generating synthetic queries
#  - Evaluating a manual chatbot response
# ------------------------------------------------------------------------------

import os
import json
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI

# Initialize client (API key picked from environment or Streamlit secrets)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

SUPPORT_SYSTEM_PROMPT = (
    "You are a helpful, empathetic, and concise customer support agent. "
    "Always be polite, acknowledge the user's concern, explain the next steps clearly, "
    "and, when helpful, provide numbered steps or short bullets. "
    "Avoid over-promising. Prefer ~2–6 sentences unless more detail is truly needed."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. Given a customer query and a proposed reply, "
    "score the reply on three 1–5 integer scales (higher is better):\n"
    "- relevance: Does the reply directly and correctly address the user's issue?\n"
    "- helpfulness: Is the reply actionable, specific, and non-generic?\n"
    "- tone: Is the reply polite, empathetic, and professional?\n"
    "Return ONLY a compact JSON object with keys: "
    '{"relevance": int, "helpfulness": int, "tone": int, "explanation": string}.'
)

CLASSIFY_SYSTEM_PROMPT = (
    "You are a classifier. Categorize queries into one of: ORDER, SHIPPING, REFUND, ACCOUNT, CANCEL."
)

SYNTHETIC_SYSTEM_PROMPT = (
    "You are a generator of synthetic customer support queries. "
    "Produce diverse, realistic queries for testing."
)


def _chat_completions(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 350,
    response_format_json: bool = False,
    max_retries: int = 5,
    retry_base_delay: float = 1.25,
) -> str:
    """Wrapper for Chat Completions (OpenAI v1)."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response_format_json:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(retry_base_delay * (2 ** attempt))
    raise RuntimeError(f"OpenAI ChatCompletion failed after {max_retries} retries: {last_err}")


# ------------------------------
# New pipeline helpers
# ------------------------------

def generate_chat_response(query: str, model: str = DEFAULT_CHAT_MODEL) -> str:
    """Generate a single customer-support-style response for a given query."""
    messages = [
        {"role": "system", "content": SUPPORT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Customer query:\n{query}"},
    ]
    return _chat_completions(messages, model=model, temperature=0.2, max_tokens=350)


def _coerce_score_int(x: Any, lo: int = 1, hi: int = 5) -> int:
    try:
        v = int(x)
    except Exception:
        v = lo
    return max(lo, min(hi, v))


def score_chat_response(query: str, response: str, model: str = DEFAULT_CHAT_MODEL) -> Dict[str, Any]:
    """Use LLM-as-judge to score the response."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"query": query, "reply": response}, ensure_ascii=False)},
    ]
    raw = _chat_completions(messages, model=model, temperature=0.0, max_tokens=250, response_format_json=True)
    try:
        data = json.loads(raw)
    except Exception:
        data = {}
    rel = _coerce_score_int(data.get("relevance", 3))
    helpf = _coerce_score_int(data.get("helpfulness", 3))
    tone = _coerce_score_int(data.get("tone", 4))
    expl = data.get("explanation") or "No explanation provided."
    return {"relevance": rel, "helpfulness": helpf, "tone": tone, "explanation": expl}


# ------------------------------
# Old helpers for Streamlit UI
# ------------------------------

def classify_query(query: str, model: str = DEFAULT_CHAT_MODEL) -> str:
    """Classify a query into ORDER, SHIPPING, REFUND, ACCOUNT, CANCEL."""
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    return _chat_completions(messages, model=model, temperature=0.0, max_tokens=50)


def generate_synthetic_queries(topic: str, n: int = 5, model: str = DEFAULT_CHAT_MODEL) -> List[str]:
    """Generate synthetic queries for a given topic."""
    messages = [
        {"role": "system", "content": SYNTHETIC_SYSTEM_PROMPT},
        {"role": "user", "content": f"Generate {n} diverse queries for the topic: {topic}"},
    ]
    text = _chat_completions(messages, model=model, temperature=0.8, max_tokens=300)
    return [line.strip("0123456789. ") for line in text.split("\n") if line.strip()]


def evaluate_response(query: str, response: str, model: str = DEFAULT_CHAT_MODEL) -> Dict[str, Any]:
    """Evaluate a manual chatbot response (UI helper)."""
    return score_chat_response(query, response, model=model)
