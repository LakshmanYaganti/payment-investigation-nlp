"""FastAPI service: classify a payment/banking investigation message.

POST /classify {"message": "..."} ->
    {intent, confidence, top_intents, entities{amounts, dates, card_last4,
     transaction_ids}}

Model lifecycle: the TF-IDF + Logistic Regression baseline is used because
it trains on BANKING77 in a few seconds and needs no GPU or model download.
On first use the service loads ``models/baseline.joblib`` if present,
otherwise it trains the baseline on the fly (see src/train.py to pre-train).
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
_MODEL = None
_LOCK = threading.Lock()


def get_model():
    """Load the pre-trained baseline, or train it on first use."""
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                artifact = ROOT / "models" / "baseline.joblib"
                if artifact.exists():
                    import joblib

                    _MODEL = joblib.load(artifact)
                else:
                    from src.baseline import build_baseline
                    from src.data import LABEL_COL, TEXT_COL, load_banking77

                    train_df, _ = load_banking77()
                    model = build_baseline()
                    model.fit(train_df[TEXT_COL], train_df[LABEL_COL])
                    _MODEL = model
    return _MODEL


class ClassifyRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=[
        "I was charged $59.99 twice on March 3 for order A1B2C3D4, card ending in 4321"
    ])


app = FastAPI(
    title="Payment Investigation NLP",
    description="Routes payment/banking investigation messages to the right "
                "intent queue and extracts amounts, dates, card and reference IDs.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/classify")
def classify(req: ClassifyRequest) -> dict:
    from src.predict import predict_message

    return predict_message(get_model(), req.message)
