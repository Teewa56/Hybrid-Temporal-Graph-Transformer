import numpy as np
from typing import Any

from app.services.cache_service import CacheService
from app.api.transactions import fetch_customer_transactions

SEQ_LEN = 50
FEATURE_DIM = 32


def _normalize_amount(amount: float, max_amount: float = 5_000_000.0) -> float:
    return min(amount / max_amount, 1.0)


def _encode_hour(hour: int) -> list[float]:
    """Cyclic encoding of hour to preserve 23→0 continuity."""
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


def _transaction_to_vector(txn: dict) -> np.ndarray:
    """
    Convert a single transaction dict to a fixed-size feature vector.
    Output shape: (FEATURE_DIM,)
    """
    from datetime import datetime

    amount = _normalize_amount(float(txn.get("amount", 0) or 0) / 100)

    created_at = txn.get("created_at", "")
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        hour_enc = _encode_hour(dt.hour)
        day_of_week = dt.weekday() / 6.0
        is_weekend = float(dt.weekday() >= 5)
    except Exception:
        hour_enc = [0.0, 0.0]
        day_of_week = 0.0
        is_weekend = 0.0

    category = txn.get("merchant_category", "unknown")
    cat_enc = _encode_merchant_category(category)  # 10-dim

    is_new_device = float(txn.get("is_new_device", False))
    is_new_recipient = float(txn.get("is_new_recipient", False))
    transaction_type = float(txn.get("transaction_type", "debit") == "debit")
    currency_is_ngn = float(txn.get("currency", "NGN") == "NGN")

    # Compose: 1 + 2 + 1 + 1 + 10 + 1 + 1 + 1 + 1 = 19 features
    features = (
        [amount]
        + hour_enc
        + [day_of_week, is_weekend]
        + cat_enc
        + [is_new_device, is_new_recipient, transaction_type, currency_is_ngn]
    )

    # Pad to FEATURE_DIM
    features = features + [0.0] * (FEATURE_DIM - len(features))
    return np.array(features[:FEATURE_DIM], dtype=np.float32)


class SequentialService:
    """
    Builds the behavioral sequence vector for the Transformer model.
    Pulls the user's last 50 transactions from Redis cache or Squad API,
    encodes each as a fixed-size feature vector, and returns
    a (SEQ_LEN, FEATURE_DIM) numpy array.
    """

    def __init__(self, cache: CacheService):
        self.cache = cache

    async def build(self, body: dict) -> np.ndarray:
        email = body.get("customer_email", "")
        cache_key = f"seq:{email}"

        # Try cache first
        cached = await self.cache.get_list(cache_key, limit=SEQ_LEN)

        if len(cached) < 5:
            # Fallback to Squad API if cache is cold
            fetched = await fetch_customer_transactions(email, limit=SEQ_LEN)
            transactions = fetched if fetched else []
        else:
            transactions = cached

        # Encode each transaction
        vectors = [_transaction_to_vector(t) for t in transactions[-SEQ_LEN:]]

        # Pad with zeros if fewer than SEQ_LEN transactions
        while len(vectors) < SEQ_LEN:
            vectors.insert(0, np.zeros(FEATURE_DIM, dtype=np.float32))

        # Cache current transaction for future requests
        await self.cache.push_to_list(cache_key, body, max_len=SEQ_LEN)

        return np.stack(vectors, axis=0)  # (SEQ_LEN, FEATURE_DIM)