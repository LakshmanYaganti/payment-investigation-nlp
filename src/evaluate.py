"""Evaluation helpers: headline metrics + confusion analysis.

Accuracy alone hides the operationally important story in a 77-way routing
problem: *which* queues get confused with each other. ``most_confused_pairs``
surfaces the top off-diagonal (true -> predicted) pairs from the confusion
matrix so error analysis can target real playbook/wording collisions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_summary(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_class_report(y_true, y_pred, labels: list[str]) -> dict:
    """Per-intent precision/recall/F1/support as a plain dict (JSON-safe)."""
    rep = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        label: {
            "precision": float(rep[label]["precision"]),
            "recall": float(rep[label]["recall"]),
            "f1": float(rep[label]["f1-score"]),
            "support": int(rep[label]["support"]),
        }
        for label in labels
    }


def most_confused_pairs(y_true, y_pred, labels: list[str], top_k: int = 10) -> list[dict]:
    """Top off-diagonal confusion pairs, most-confused first."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "true_intent": true_label,
                        "predicted_intent": pred_label,
                        "count": int(cm[i, j]),
                    }
                )
    pairs.sort(key=lambda p: (-p["count"], p["true_intent"], p["predicted_intent"]))
    return pairs[:top_k]
