"""Single-message inference: intent + confidence + extracted entities."""

from __future__ import annotations

import numpy as np

from .extract import extract_entities


def predict_message(model, text: str, top_k: int = 3) -> dict:
    """Classify one message with a fitted sklearn-style pipeline.

    Returns the top intent, its confidence (max predicted probability), the
    top-k candidate intents, and the rule-extracted entities — everything a
    triage UI or routing rule needs in one payload.
    """
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    order = np.argsort(proba)[::-1]
    top = [
        {"intent": classes[i], "confidence": float(proba[i])}
        for i in order[:top_k]
    ]
    return {
        "intent": top[0]["intent"],
        "confidence": top[0]["confidence"],
        "top_intents": top,
        "entities": extract_entities(text),
    }
