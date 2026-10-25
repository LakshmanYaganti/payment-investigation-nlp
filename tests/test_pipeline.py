"""Tests: entity extraction, dataset loading, model + API behaviour."""

import pandas as pd
import pytest

from src.baseline import build_baseline
from src.extract import extract_entities
from src.predict import predict_message

# ------------------------------------------------------------ entities

def test_extract_amount_with_cents():
    ents = extract_entities("I was charged $59.99 twice for the same order")
    assert ents["amounts"], "expected an amount"
    assert ents["amounts"][0]["value"] == pytest.approx(59.99)


def test_extract_amount_thousands_and_currency_word():
    ents = extract_entities("A payment of USD 1,250.00 left my account, about 45 dollars in fees")
    values = {a["value"] for a in ents["amounts"]}
    assert 1250.00 in values
    assert 45.0 in values


def test_extract_dates_multiple_formats():
    text = "Charged on 2026-03-14 and again on 03/20/2026, statement dated 14 March 2026"
    ents = extract_entities(text)
    assert "2026-03-14" in ents["dates"]
    assert "03/20/2026" in ents["dates"]
    assert any("March" in d for d in ents["dates"])


def test_extract_card_last4():
    assert "4321" in extract_entities("my card ending in 4321 was charged")["card_last4"]
    assert "9876" in extract_entities("card **** **** **** 9876 declined")["card_last4"]


def test_extract_transaction_id():
    ents = extract_entities("Please check transaction TXN-8F3K2X9, it looks wrong")
    assert any("8F3K2X9" in t for t in ents["transaction_ids"])
    ents2 = extract_entities("reference #A1B2C3D4 for the duplicate charge")
    assert any("A1B2C3D4" in t for t in ents2["transaction_ids"])


def test_extract_empty_and_plain_text():
    empty = extract_entities("")
    assert empty == {"amounts": [], "dates": [], "card_last4": [], "transaction_ids": []}
    plain = extract_entities("I just have a question about my account")
    assert plain["card_last4"] == []
    assert plain["amounts"] == []


# ------------------------------------------------------------ dataset

def test_banking77_loads_with_official_split():
    from src.data import load_banking77

    try:
        train, test = load_banking77()
    except Exception as exc:  # offline environment — dataset cannot be fetched
        pytest.skip(f"BANKING77 unavailable in this environment: {exc}")
    assert len(train) == 10003
    assert len(test) == 3080
    assert train["category"].nunique() == 77
    assert set(train["category"]) == set(test["category"])


# ------------------------------------------------------------ model + API

TINY = pd.DataFrame(
    {
        "text": [
            "my card was charged twice for the same purchase",
            "I see a duplicate charge on my card statement",
            "charged two times for one order, please refund",
            "where is my refund for the returned item",
            "refund has not arrived after ten days",
            "still waiting for my money back from the refund",
            "I need to activate my new debit card",
            "how do I activate the card you sent me",
            "my new card will not activate in the app",
        ],
        "category": (
            ["lost_or_stolen_card"] * 0
            + ["duplicate_charge"] * 3
            + ["refund_not_showing"] * 3
            + ["activate_card"] * 3
        ),
    }
)


@pytest.fixture(scope="module")
def tiny_model():
    model = build_baseline()
    model.fit(TINY["text"], TINY["category"])
    return model


def test_baseline_predicts_and_predict_message_shape(tiny_model):
    out = predict_message(tiny_model, "I was charged twice, card ending in 4321, $25.00 on March 3")
    assert out["intent"] == "duplicate_charge"
    assert 0.0 < out["confidence"] <= 1.0
    assert len(out["top_intents"]) == 3
    assert out["entities"]["card_last4"] == ["4321"]
    assert out["entities"]["amounts"][0]["value"] == pytest.approx(25.00)


def test_api_health_and_classify(tiny_model):
    from fastapi.testclient import TestClient

    import api.app as app_module

    app_module._MODEL = tiny_model  # inject tiny model; skips BANKING77 training
    client = TestClient(app_module.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    resp = client.post("/classify", json={"message": "where is my refund?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "refund_not_showing"
    assert "entities" in body and "confidence" in body
