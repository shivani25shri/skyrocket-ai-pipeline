import re
import os
import pandas as pd
from typing import Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# 🔹 Ensure NLTK data is available (safe for Streamlit Cloud)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)  # optional: wordnet synonyms support

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
SPELL = SpellChecker()

# Regex for punctuation / digits
_punct_re = re.compile(r"[^\w\s]")
_num_re = re.compile(r"\d+")

# Custom typo dictionary
CUSTOM_MAP = {
    "acn": "can",
    "ya": "you",
    "pls": "please",
    "plz": "please",
    "u": "you",
    "ur": "your",
    "thx": "thanks",
    "tnx": "thanks",
    "im": "i am",
    "dont": "do not",
    "wanna": "want to",
    "gonna": "going to",
}

def load_data(excel_path: str, queries_sheet: str, responses_sheet: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load queries and responses from Excel."""
    queries_df = pd.read_excel(excel_path, sheet_name=queries_sheet)
    responses_df = pd.read_excel(excel_path, sheet_name=responses_sheet)
    queries_df.columns = [c.strip() for c in queries_df.columns]
    responses_df.columns = [c.strip() for c in responses_df.columns]
    return queries_df, responses_df

def clean_text(text: str) -> str:
    """Advanced text cleaning with typo correction + lemmatization."""
    text = str(text).lower()
    text = _punct_re.sub(" ", text)   # remove punctuation
    text = _num_re.sub(" ", text)     # remove numbers
    tokens = text.split()

    # Custom typo normalization
    tokens = [CUSTOM_MAP.get(t, t) for t in tokens]

    # Spell correction (skip single-char tokens)
    corrected = []
    for t in tokens:
        if len(t) > 1 and t not in SPELL:
            suggestion = SPELL.correction(t)
            corrected.append(suggestion if suggestion else t)
        else:
            corrected.append(t)

    # Stopwords removal
    tokens = [t for t in corrected if t not in STOPWORDS]

    # Lemmatization
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)

def prepare_queries(queries_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare queries by cleaning + adding length column."""
    col = "Queries" if "Queries" in queries_df.columns else list(queries_df.columns)[0]
    queries_df = queries_df.copy()
    queries_df["clean"] = queries_df[col].astype(str).map(clean_text)
    queries_df["length"] = queries_df["clean"].str.split().map(len)
    return queries_df

def ensure_outputs_dir(path: str = "outputs"):
    os.makedirs(path, exist_ok=True)
