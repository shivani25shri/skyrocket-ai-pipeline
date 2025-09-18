import re
from typing import List, Tuple
import pandas as pd
import dateparser

ORDER_ID_RE = re.compile(r"\b\d{5,}\b")
MONEY_RE = re.compile(r"\$?\b\d+(?:\.\d{1,2})?\b")

ACCOUNT_ACTIONS = [
    "reset password", "forgot password", "delete account", "close account",
    "change address", "update address", "update email", "edit personal information"
]

def extract_regex_entities(text: str) -> List[Tuple[str, str]]:
    s = str(text).lower()
    entities = []

    # Order IDs
    if m := ORDER_ID_RE.search(s):
        entities.append(("OrderID", m.group(0)))

    # Money amounts
    if m := MONEY_RE.search(s):
        entities.append(("Amount", m.group(0)))

    # Dates (only if explicit terms exist)
    d = dateparser.parse(s, settings={"STRICT_PARSING": False})
    if d:
        if any(tok in s for tok in [
            "today", "tomorrow", "yesterday", "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday", "sunday",
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        ]):
            entities.append(("Date", d.strftime("%Y-%m-%d")))

    # Account actions
    for act in ACCOUNT_ACTIONS:
        if act in s:
            entities.append(("AccountAction", act))

    # NEW: Domain-specific intents
    if "refund" in s:
        entities.append(("Intent", "Refund"))
    if "order" in s:
        entities.append(("Intent", "Order"))
    if "account" in s:
        entities.append(("Intent", "Account"))
    if "payment" in s or "pay" in s:
        entities.append(("Intent", "Payment"))
    if "ship" in s or "delivery" in s:
        entities.append(("Intent", "Shipping"))

    return entities


def add_entities_column(df: pd.DataFrame, text_col: str = "clean") -> pd.DataFrame:
    out = df.copy()
    out["entities"] = out[text_col].map(extract_regex_entities)
    return out
