"""Dataset loading for BANKING77 (PolyAI) customer-query intents.

BANKING77: 13,083 real customer-service queries labelled with 77 intents
(Casanueva et al., 2020 — "Efficient Intent Detection with Dual Sentence
Encoders", https://arxiv.org/abs/2003.04807).

Loading strategy (no data is ever committed to this repo):
1. Try the Hugging Face hub via ``datasets`` (``PolyAI/banking77``).
   Recent ``datasets`` versions dropped script-based datasets, so this may
   fail on newer installs — that is expected, not an error.
2. Fall back to the canonical CSVs published in the authors' GitHub repo
   (the exact files the HF loading script itself downloads), cached under
   ``data/cache/`` (git-ignored) or ``$BANKING77_CACHE``.

We keep all 77 intents (no grouping): fine-grained routing *is* the product
problem — each intent maps to a distinct support queue / playbook entry, and
grouping them would hide exactly the confusions a triage system must handle.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd

TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"
    "/master/banking_data/train.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"
    "/master/banking_data/test.csv"
)

TEXT_COL = "text"
LABEL_COL = "category"


def _cache_dir() -> Path:
    root = os.environ.get("BANKING77_CACHE")
    if root:
        return Path(root)
    # repo_root/data/cache  (this file lives in repo_root/src/)
    return Path(__file__).resolve().parent.parent / "data" / "cache"


def _load_via_huggingface() -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset  # imported lazily; optional path

    ds = load_dataset("PolyAI/banking77")
    train = pd.DataFrame({"text": ds["train"]["text"], "category": ds["train"]["label_name"]}) \
        if "label_name" in ds["train"].column_names else None
    if train is None:
        # Older datasets versions expose the raw integer label; map via features.
        names = ds["train"].features["label"].names
        train = pd.DataFrame(
            {"text": ds["train"]["text"],
             "category": [names[i] for i in ds["train"]["label"]]}
        )
        test = pd.DataFrame(
            {"text": ds["test"]["text"],
             "category": [names[i] for i in ds["test"]["label"]]}
        )
    else:
        test = pd.DataFrame({"text": ds["test"]["text"], "category": ds["test"]["label_name"]})
    return train, test


def _load_via_csv() -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, url in (("train.csv", TRAIN_URL), ("test.csv", TEST_URL)):
        path = cache / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        frames.append(pd.read_csv(path))
    return frames[0], frames[1]


def load_banking77() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) with columns [text, category] using the
    official BANKING77 train/test split (10,003 train / 3,080 test)."""
    try:
        train, test = _load_via_huggingface()
    except Exception:
        train, test = _load_via_csv()
    for df in (train, test):
        df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
        df[LABEL_COL] = df[LABEL_COL].astype(str)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def intent_names(train: pd.DataFrame) -> list[str]:
    """Sorted list of the 77 intent names (stable ordering everywhere)."""
    return sorted(train[LABEL_COL].unique().tolist())
