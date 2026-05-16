import numpy as np
from typing import Any

from app.services.cache_service import CacheService
from app.api.transactions import fetch_customer_transactions

SEQ_LEN = 50
FEATURE_DIM = 32


def _normalize_amount(amount: float, max_amount: float = 5_000_000.0) -> float:
    return min(amount / max_amount, 1.0)


def _encode_hour(hour: int) -> list[float]:
    import math
    return [math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)]


MERCHANT_CATEGORIES = [
    "transfer", "airtime", "data", "bills", "shopping",
    "food", "transport", "crypto", "pos", "unknown"
]


def _encode_merchant_category(category: str) -> list[float]:
    vec = [0.0] * len(MERCHANT_CATEGORIES)
    idx = MERCHANT_CATEGORIES.index(category) if category in MERCHANT_CATEGORIES else -1
    if idx >= 0:
        vec[idx] = 1.0
    return vec


# Squad card webhook transaction_type values
_CARD_TX_TYPES = {"card", "transfer", "bank", "ussd", "merchantussd", "virtualaccount"}


def _transaction_to_vector(txn: dict) -> np.ndarray:
    """
    Convert a single transaction dict to a fixed-size feature vector.
    Handles two formats:
      1. Squad card/transfer webhook body fields
      2. Squad virtual account transaction list fields
    Output shape: (FEATURE_DIM,)
    """
    from datetime import datetime

    # ── Amount ───────────────────────────────────────────────────────────────
    # Card webhook: 'amount' in kobo (int)
    # VA transaction list: 'principal_amount' in naira (string)
    raw_amount = txn.get("amount")
    if raw_amount is None:
        principal_str = txn.get("principal_amount", "0") or "0"
        raw_amount = float(principal_str) * 100   # convert naira → kobo
    amount = _normalize_amount(float(raw_amount or 0) / 100)

    # ── Timestamp ─────────────────────────────────────────────────────────────
    # Card webhook: 'created_at'
    # VA transaction list: 'transaction_date'
    created_at = txn.get("created_at") or txn.get("transaction_date", "")
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        hour_enc = _encode_hour(dt.hour)
        day_of_week = dt.weekday() / 6.0
        is_weekend = float(dt.weekday() >= 5)
    except Exception:
        hour_enc = [0.0, 0.0]
        day_of_week = 0.0
        is_weekend = 0.0

    # ── Merchant category ─────────────────────────────────────────────────────
    category = txn.get("merchant_category", "unknown") or "unknown"
    cat_enc = _encode_merchant_category(category)

    # ── Behavioural flags ─────────────────────────────────────────────────────
    is_new_device = float(txn.get("is_new_device", False))
    is_new_recipient = float(txn.get("is_new_recipient", False))

    # ── Transaction direction ─────────────────────────────────────────────────
    # Card webhooks: transaction_type in ("Card", "Transfer", "Bank", "Ussd", "MerchantUssd")
    # VA transactions: transaction_indicator "C" (credit) or "D" (debit)
    tx_type = str(txn.get("transaction_type", "")).lower()
    tx_indicator = str(txn.get("transaction_indicator", "")).upper()
    is_inbound = float(
        tx_type in _CARD_TX_TYPES or tx_indicator == "C"
    )

    currency_is_ngn = float(txn.get("currency", "NGN") == "NGN")

    # Compose: 1 + 2 + 1 + 1 + 10 + 1 + 1 + 1 + 1 = 19 features
    features = (
        [amount]
        + hour_enc
        + [day_of_week, is_weekend]
        + cat_enc
        + [is_new_device, is_new_recipient, is_inbound, currency_is_ngn]
    )

    # Pad to FEATURE_DIM
    features = features + [0.0] * (FEATURE_DIM - len(features))
    return np.array(features[:FEATURE_DIM], dtype=np.float32)


class SequentialService:
    """
    Builds the behavioral sequence vector for the Transformer model.
    Pulls the user's last 50 transactions from Redis cache or Squad API.
    Handles both card-payment transactions and virtual-account transactions.
    """

    def __init__(self, cache: CacheService):
        self.cache = cache

    async def build(self, body: dict) -> np.ndarray:
        # Prefer customer_email; fall back to customer_identifier (VA webhooks)
        identifier = (
            body.get("customer_email")
            or body.get("customer_identifier")
            or ""
        )
        cache_key = f"seq:{identifier}"

        cached = await self.cache.get_list(cache_key, limit=SEQ_LEN)

        if len(cached) < 5 and identifier:
            # fetch_customer_transactions takes (customer_identifier, limit)
            fetched = await fetch_customer_transactions(identifier, limit=SEQ_LEN)
            transactions = fetched if fetched else []
        else:
            transactions = cached

        vectors = [_transaction_to_vector(t) for t in transactions[-SEQ_LEN:]]

        while len(vectors) < SEQ_LEN:
            vectors.insert(0, np.zeros(FEATURE_DIM, dtype=np.float32))

        if identifier:
            await self.cache.push_to_list(cache_key, body, max_len=SEQ_LEN)

        return np.stack(vectors, axis=0)  # (SEQ_LEN, FEATURE_DIM)