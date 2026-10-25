# Payment Investigation NLP

**Automatically route incoming payment & banking investigation messages to the
right queue — and pull out the amount, date, card and transaction ID an
investigator needs — with an NLP intent classifier trained on BANKING77.**

## The problem

Every customer payment query ("charged twice", "refund never arrived",
"transfer hasn't reached the recipient") lands in a support/investigations
inbox and is **read and routed by a human**. Manual triage is slow, it is
inconsistent between agents, and it delays precisely the cases — suspected
fraud, duplicate charges, failed transfers — where response time matters most.

This project automates the first pass:

1. **Classify** the message into one of 77 intents (the routing queue).
2. **Extract** the structured facts from the text: amount, date, card
   last-4, transaction/reference ID.
3. **Expose confidence** so low-certainty messages go to a human with the
   top-3 suggested queues attached, instead of being silently misrouted.

## Approach

| Stage | What happens |
|---|---|
| Data | BANKING77 official train/test split, all 77 intents kept (no grouping — each intent is a distinct resolution queue) |
| Baseline | TF-IDF word n-grams (1–2) + Logistic Regression |
| Advanced | **Sentence-transformer embeddings** (`all-MiniLM-L6-v2`, 384-dim) + Logistic Regression |
| Entities | Deterministic regex extraction (amounts, dates, card last-4, TXN/REF IDs) — exact and unit-tested |
| Serving | FastAPI `POST /classify` → intent, confidence, top-3 intents, entities |

**Why embeddings for the advanced model:** TF-IDF matches *wording*;
embeddings encode *meaning*. Customers paraphrase the same problem endlessly
("charged twice" / "double charge" / "billed two times"), and semantic
embeddings generalise across those phrasings — worth **+3.6 points** of
accuracy and macro-F1 here. `src/embeddings_model.py` uses the embeddings
model whenever `sentence-transformers` is installed and falls back to a
TF-IDF word+char n-gram model otherwise; whichever backend actually runs is
recorded in `reports/metrics.json`. The results below are the embeddings run
(`"model": "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2"`).

## Dataset

**BANKING77** (Casanueva et al., 2020) — 13,083 real, anonymised
customer-service queries across 77 banking intents.

- Official split: **10,003 train / 3,080 test** (exactly 40 test per intent)
- Source: [Hugging Face `PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) /
  [authors' GitHub](https://github.com/PolyAI-LDN/task-specific-datasets)
- The dataset is **downloaded at runtime** by `src/data.py` (cached under
  `data/cache/`, git-ignored) — no data files are committed to this repo.
  See [`data/README.md`](data/README.md).

## Results (official test set, 3,080 messages)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| TF-IDF (word 1–2g) + LogReg — baseline | 0.8938 | 0.8942 | 0.8942 |
| MiniLM embeddings + LogReg — advanced | **0.9299** | **0.9298** | **0.9298** |

Full per-intent precision/recall/F1 and confusion detail:
[`reports/metrics.json`](reports/metrics.json). Reproduce with
`python -m src.train`.

## Error analysis — most confused intent pairs

Top (true → predicted) confusions of the advanced (embeddings) model:

| True intent | Predicted as | Count |
|---|---|---|
| `balance_not_updated_after_bank_transfer` | `transfer_not_received_by_recipient` | 5 |
| `declined_transfer` | `declined_card_payment` | 5 |
| `card_arrival` | `card_delivery_estimate` | 4 |
| `card_payment_not_recognised` | `compromised_card` | 4 |
| `pending_transfer` | `transfer_not_received_by_recipient` | 4 |
| `why_verify_identity` | `verify_my_identity` | 4 |

These are **taxonomy collisions, not random noise**: the residual errors
concentrate in the **transfer-status cluster** — `balance_not_updated…`,
`pending_transfer` and `transfer_not_received_by_recipient` are different
stages of the same journey, described by customers in nearly identical
words — and in sibling pairs like `declined_transfer` / `declined_card_payment`
that differ only in the payment rail. The weakest single intent is
`transfer_not_received_by_recipient` (F1 0.776), the sink of that cluster.
Production takeaway: route by **confidence threshold with human review**,
not argmax alone.

## Entity extraction

`src/extract.py` — high-precision regexes (all unit-tested) for:

- **Amounts** — `$59.99`, `USD 1,250.00`, `45 dollars`
- **Dates** — `2026-03-14`, `03/14/2026`, `14 March 2026`, `March 3rd`
- **Card last-4** — `card ending in 4321`, `**** **** **** 4321`
- **Transaction / reference IDs** — `TXN-8F3K2X9`, `ref #A1B2C3D4`

```json
POST /classify  {"message": "I was charged $59.99 twice on March 3, card ending in 4321, txn TXN-8F3K2X9"}
→ {
    "intent": "transaction_charged_twice",
    "confidence": 0.89,
    "entities": {
      "amounts": [{"raw": "$59.99", "value": 59.99}],
      "dates": ["March 3"],
      "card_last4": ["4321"],
      "transaction_ids": ["TXN-8F3K2X9"]
    }
  }
```

## Project structure

```
├── api/app.py                 # FastAPI service (/health, /classify)
├── src/
│   ├── data.py                # BANKING77 loading (HF hub → canonical CSV fallback)
│   ├── baseline.py            # TF-IDF word n-grams + LogReg
│   ├── embeddings_model.py    # sentence-embeddings model + char-TFIDF fallback
│   ├── extract.py             # rule-based entity extraction
│   ├── evaluate.py            # accuracy / macro-F1 / per-class / confusion pairs
│   ├── predict.py             # message → intent + confidence + entities
│   └── train.py               # train + evaluate both models → reports/metrics.json
├── notebooks/01_message_classification.ipynb   # executed EDA + modelling notebook
├── tests/test_pipeline.py     # entities, data loading, model + API tests
├── reports/metrics.json       # real measured results (source of truth)
└── data/README.md             # dataset provenance
```

## How to run

```bash
pip install -r requirements.txt

python -m src.train            # train both models, write reports/metrics.json
pytest                         # 9 tests: extraction, data, model, API
uvicorn api.app:app --reload   # serve on http://127.0.0.1:8000  (docs at /docs)
```

The API serves the **baseline** model: it trains on BANKING77 in a few
seconds, needs no GPU or model download, and is loaded from
`models/baseline.joblib` if you have run training — otherwise it trains on
first use.

## Tech stack

Python · scikit-learn · sentence-transformers · pandas · FastAPI ·
Hugging Face `datasets` · matplotlib/seaborn · pytest

## License

MIT — see [`LICENSE`](LICENSE). The BANKING77 dataset itself is released by
PolyAI for research purposes; see the dataset card for its terms.

## Author

**Lakshman Sai Yaganti** — Data Scientist @ Square
Fraud detection, credit risk, and payment investigations analytics.
