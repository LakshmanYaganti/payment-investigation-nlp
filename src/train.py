"""Train + evaluate both models on the official BANKING77 split.

Usage:  python -m src.train
Writes: reports/metrics.json  (the single source of truth for README numbers)
        models/baseline.joblib (git-ignored artifact; the API can also
        retrain the baseline on first use)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from .baseline import build_baseline
from .data import LABEL_COL, TEXT_COL, intent_names, load_banking77
from .embeddings_model import build_advanced
from .evaluate import classification_summary, most_confused_pairs, per_class_report

ROOT = Path(__file__).resolve().parent.parent


def run() -> dict:
    train_df, test_df = load_banking77()
    labels = intent_names(train_df)
    X_train, y_train = train_df[TEXT_COL], train_df[LABEL_COL]
    X_test, y_test = test_df[TEXT_COL], test_df[LABEL_COL]

    results: dict = {
        "dataset": {
            "name": "BANKING77 (PolyAI)",
            "source": "Hugging Face hub 'PolyAI/banking77' / authors' GitHub CSVs",
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "num_intents": len(labels),
        }
    }

    baseline = build_baseline()
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    results["baseline_tfidf_logreg"] = {
        **classification_summary(y_test, base_pred),
        "most_confused_pairs": most_confused_pairs(y_test, base_pred, labels, top_k=10),
    }

    advanced, backend = build_advanced()
    advanced.fit(X_train, y_train)
    adv_pred = advanced.predict(X_test)
    results["advanced"] = {
        "model": backend,
        **classification_summary(y_test, adv_pred),
        "most_confused_pairs": most_confused_pairs(y_test, adv_pred, labels, top_k=10),
    }

    # Per-class detail for the winning model, kept for README/error analysis.
    winner = "advanced" if results["advanced"]["macro_f1"] >= results["baseline_tfidf_logreg"]["macro_f1"] else "baseline_tfidf_logreg"
    winner_pred = adv_pred if winner == "advanced" else base_pred
    report = per_class_report(y_test, winner_pred, labels)
    worst = sorted(report.items(), key=lambda kv: kv[1]["f1"])[:10]
    results["winner"] = winner
    results["winner_worst_classes_by_f1"] = [
        {"intent": k, **v} for k, v in worst
    ]
    results["winner_per_class"] = report

    (ROOT / "reports").mkdir(exist_ok=True)
    with open(ROOT / "reports" / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(baseline, ROOT / "models" / "baseline.joblib")

    print(json.dumps({k: results[k] for k in ("baseline_tfidf_logreg", "advanced")},
                     indent=2, default=str)[:1200])
    print("winner:", winner)
    return results


if __name__ == "__main__":
    run()
