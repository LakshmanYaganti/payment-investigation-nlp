"""Rule-based entity extraction for payment investigation messages.

A triage agent (human or model) needs more than an intent label: the amount
in dispute, when it happened, which card was used, and any reference ID.
These are high-precision, regex-friendly patterns, so rules beat ML here —
they are exact, auditable, and unit-testable.

Extracted entities:
- ``amounts``          — "$1,250.00", "USD 45", "45 dollars"
- ``dates``            — "2026-03-14", "03/14/2026", "14 March 2026", "March 3"
- ``card_last4``       — "card ending in 4321", "**** **** **** 4321", "last 4: 4321"
- ``transaction_ids``  — "TXN-8F3K2", "ref #A1B2C3D4", "transaction id: ZX88..." —
                         alphanumeric reference tokens of length >= 6 that carry
                         a TXN/REF prefix or a digit+letter mix.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- amounts
_AMOUNT_RE = re.compile(
    r"""
    (?:USD|GBP|EUR|\$|£|€)\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?   # $1,250.00 / USD 45
    |
    \d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?(?:dollars|pounds|euros|USD|GBP|EUR)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ------------------------------------------------------------------ dates
_DATE_RES = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                                  # 2026-03-14
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),                            # 03/14/2026
    re.compile(
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)(?:\s+\d{4})?\b",
        re.IGNORECASE,
    ),                                                                     # 14 March 2026
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?"
        r"(?:,?\s+\d{4})?\b",
        re.IGNORECASE,
    ),                                                                     # March 3rd, 2026
]

# ------------------------------------------------------------- card last-4
_CARD_RES = [
    re.compile(r"(?:ending in|ends in|last 4(?: digits)?(?: is|:)?)\s*\*{0,4}\s*(\d{4})\b",
               re.IGNORECASE),
    re.compile(r"(?:\*{2,}\s*){2,}\*{0,2}\s*(\d{4})\b"),                   # **** **** **** 4321
    re.compile(r"\bcard\s+(?:number\s+)?(?:ending|ends)\s+(?:in\s+)?(\d{4})\b",
               re.IGNORECASE),
]

# --------------------------------------------------------- transaction IDs
_TXN_PREFIX_RE = re.compile(
    r"\b(?:TXN|REF|REFERENCE|TRANSACTION(?:\s+ID)?|CONFIRMATION(?:\s+CODE)?)"
    r"\s*[#:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-]{5,})\b",
    re.IGNORECASE,
)
# Bare reference-looking token: >=8 chars, mixed letters+digits, not a word.
_BARE_ID_RE = re.compile(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{8,}\b")


def _amount_to_float(raw: str) -> float | None:
    nums = re.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?", raw)
    if not nums:
        return None
    try:
        return float(nums[0].replace(",", ""))
    except ValueError:
        return None


def extract_entities(text: str) -> dict:
    """Extract structured entities from a free-text investigation message.

    Returns a dict with keys ``amounts`` (list of {"raw", "value"}),
    ``dates`` (list of str), ``card_last4`` (list of str) and
    ``transaction_ids`` (list of str). All lists are de-duplicated and keep
    first-seen order.
    """
    if not text:
        return {"amounts": [], "dates": [], "card_last4": [], "transaction_ids": []}

    amounts, seen_amt = [], set()
    for m in _AMOUNT_RE.finditer(text):
        raw = m.group(0).strip()
        if raw.lower() not in seen_amt:
            seen_amt.add(raw.lower())
            amounts.append({"raw": raw, "value": _amount_to_float(raw)})

    dates, seen_date = [], set()
    for rx in _DATE_RES:
        for m in rx.finditer(text):
            raw = m.group(0).strip()
            if raw.lower() not in seen_date:
                seen_date.add(raw.lower())
                dates.append(raw)

    cards, seen_card = [], set()
    for rx in _CARD_RES:
        for m in rx.finditer(text):
            last4 = m.group(1)
            if last4 not in seen_card:
                seen_card.add(last4)
                cards.append(last4)

    txns, seen_txn = [], set()
    for m in _TXN_PREFIX_RE.finditer(text):
        tid = m.group(1).strip("-")
        if tid.upper() not in seen_txn:
            seen_txn.add(tid.upper())
            txns.append(tid)
    for m in _BARE_ID_RE.finditer(text):
        tid = m.group(0)
        # skip things that look like amounts/dates caught above or plain words
        if tid.upper() not in seen_txn and not re.fullmatch(r"\d+", tid):
            seen_txn.add(tid.upper())
            txns.append(tid)

    return {
        "amounts": amounts,
        "dates": dates,
        "card_last4": cards,
        "transaction_ids": txns,
    }
