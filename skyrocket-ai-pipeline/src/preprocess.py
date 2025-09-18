import re
import os
import pandas as pd
from typing import Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# Download resources if missing
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
SPELL = SpellChecker()

# Regex for punctuation / digits
_punct_re = re.compile(r"[^\w\s]")
_num_re = re.compile(r"\d+")

# Custom dictionary for common typos/slang in support queries
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
    """Advanced text cleaning: lowercase, normalize typos, spell-correct, remove stopwords, lemmatize."""
    text = str(text).lower()
    text = _punct_re.sub(" ", text)   # remove punctuation
    text = _num_re.sub(" ", text)     # remove numbers
    tokens = text.split()

    # Step 1: Apply custom typo/slang normalization
    tokens = [CUSTOM_MAP.get(t, t) for t in tokens]

    # Step 2: Spell correction (skip if word is already correct or single char)
    corrected_tokens = []
    for t in tokens:
        if len(t) > 1 and t not in SPELL:
            suggestion = SPELL.correction(t)
            corrected_tokens.append(suggestion if suggestion else t)
        else:
            corrected_tokens.append(t)
    tokens = corrected_tokens

    # Step 3: Remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]

    # Step 4: Lemmatize
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
