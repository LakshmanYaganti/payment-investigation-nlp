"""Advanced model: sentence-transformer embeddings + Logistic Regression.

Each message is encoded with a pretrained sentence encoder
(``all-MiniLM-L6-v2``, 384-dim) and a multinomial Logistic Regression head is
trained on the frozen embeddings. Semantic embeddings generalise beyond the
exact wording a TF-IDF model memorises, which helps on paraphrases —
customers rarely describe the same problem the same way twice.

Fallback: if ``sentence-transformers`` (and its torch dependency) is not
installed or the encoder cannot be downloaded, ``build_advanced`` returns a
TF-IDF model combining word n-grams with character n-grams instead. Char
n-grams make the model robust to typos and morphological variants
("charged", "charging", "charges"). Whichever backend actually runs is
reported by ``backend_name`` and recorded in reports/metrics.json — we never
label the fallback as an embeddings model.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingLogReg(BaseEstimator, ClassifierMixin):
    """Logistic Regression on frozen sentence-transformer embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, C: float = 10.0,
                 max_iter: int = 1000, batch_size: int = 64):
        self.model_name = model_name
        self.C = C
        self.max_iter = max_iter
        self.batch_size = batch_size

    def _encode(self, texts):
        return self.encoder_.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def fit(self, X, y):
        from sentence_transformers import SentenceTransformer

        self.encoder_ = SentenceTransformer(self.model_name)
        embeddings = self._encode(X)
        self.clf_ = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")
        self.clf_.fit(embeddings, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        return self.clf_.predict(self._encode(X))

    def predict_proba(self, X):
        return self.clf_.predict_proba(self._encode(X))


def build_char_tfidf_model() -> Pipeline:
    """Fallback 'advanced' model: word + char TF-IDF union + LogReg."""
    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(
                            lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
                        ("char", TfidfVectorizer(
                            lowercase=True, analyzer="char_wb",
                            ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
                    ]
                ),
            ),
            ("clf", LogisticRegression(C=10.0, max_iter=1000, solver="lbfgs")),
        ]
    )


def build_advanced():
    """Return (model, backend_name). Prefers sentence-transformer embeddings;
    falls back to word+char TF-IDF if the dependency/weights are unavailable.
    The returned name describes what *actually* will run."""
    try:
        import sentence_transformers  # noqa: F401
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(EMBEDDING_MODEL_NAME)  # probe: weights downloadable?
        return EmbeddingLogReg(), f"sentence-transformers:{EMBEDDING_MODEL_NAME}"
    except Exception:
        return build_char_tfidf_model(), "tfidf-word-char-ngrams (fallback)"
