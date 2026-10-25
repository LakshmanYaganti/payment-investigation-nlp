"""Baseline model: TF-IDF (word n-grams) + Logistic Regression.

The workhorse baseline for intent classification. On short customer-service
queries it is fast to train (~seconds on BANKING77), fully deterministic,
and embarrassingly hard to beat by a large margin.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=10.0,
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
