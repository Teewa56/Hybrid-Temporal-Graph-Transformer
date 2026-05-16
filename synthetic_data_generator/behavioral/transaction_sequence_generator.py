import uuid
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from synthetic_data_generator.config import CONFIG, NIGERIAN_CARRIERS
from synthetic_data_generator.behavioral.user_profile_generator import UserProfile


def _sample_amount(mean: float, std: float) -> float:
    amount = np.random.normal(mean, std)
    return round(max(100.0, amount), 2)


def _sample_hour(active_hours: list[int]) -> int:
    return random.choice(active_hours)


def _sample_merchant(weights: dict) -> str:
    categories = list(weights.keys())
    probs = list(weights.values())
    total = sum(probs)
    probs = [p / total for p in probs]
    return np.random.choice(categories, p=probs)


def _build_transaction(
    user: UserProfile,
    timestamp: datetime,
    is_new_device: bool = False,
    is_new_recipient: bool = False,
    override_amount: Optional[float] = None,
    override_merchant: Optional[str] = None,
) -> dict:
    merchant = override_merchant or _sample_merchant(user.merchant_weights)
    amount = override_amount or _sample_amount(
        user.typical_amount_mean, user.typical_amount_std
    )
    recipient = (
        str(uuid.uuid4())[:8]
        if is_new_recipient
        else random.choice(user.typical_recipients)
    )
    device = str(uuid.uuid4()) if is_new_device else user.device_id

    return {
        "transaction_ref": str(uuid.uuid4()).replace("-", "")[:12].upper(),
        "user_id": user.user_id,
        "customer_email": user.email,
        "amount": int(amount * 100),           # Kobo, matching the backend amount unit
        "currency": "NGN",
        "merchant_category": merchant,
        "recipient_id": recipient,
        "device_id": device,
        "is_new_device": is_new_device,
        "is_new_recipient": is_new_recipient,
        "ip_address": f"102.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "state": user.state,
        "carrier": user.carrier,
        "channel": random.choice(["app", "ussd", "web", "pos"]),
        "transaction_type": "debit",
        "created_at": timestamp.isoformat() + "Z",
        "label": 0,      # 0 = legit, 1 = fraud (set by anomaly injector)
    }


class TransactionSequenceGenerator:
    """
    Generates time-ordered transaction sequences per user profile.
    Sequences reflect realistic Nigerian FinTech behavioral patterns:
    log-normal amounts, time-of-day curves, merchant category Markov chains.
    """

    # Simple Markov chain: given current merchant, likely next merchant
    MERCHANT_TRANSITIONS = {
        "transfer":  {"transfer": 0.4, "airtime": 0.2, "bills": 0.2, "data": 0.1, "shopping": 0.1},
        "airtime":   {"airtime": 0.3, "data": 0.3, "transfer": 0.2, "bills": 0.1, "food": 0.1},
        "data":      {"data": 0.3, "airtime": 0.2, "transfer": 0.3, "shopping": 0.1, "food": 0.1},
        "bills":     {"bills": 0.2, "transfer": 0.4, "shopping": 0.2, "food": 0.1, "transport": 0.1},
        "shopping":  {"shopping": 0.3, "food": 0.2, "transfer": 0.2, "bills": 0.15, "transport": 0.15},
        "food":      {"food": 0.3, "transport": 0.2, "shopping": 0.2, "airtime": 0.2, "data": 0.1},
        "transport": {"transport": 0.3, "food": 0.3, "airtime": 0.2, "data": 0.1, "bills": 0.1},
        "crypto":    {"crypto": 0.5, "transfer": 0.3, "shopping": 0.1, "bills": 0.1},
        "pos":       {"pos": 0.3, "food": 0.2, "shopping": 0.3, "transport": 0.2},
        "unknown":   {"transfer": 0.4, "airtime": 0.3, "data": 0.2, "unknown": 0.1},
    }

    def __init__(self, seed: int = None):
        seed = seed or CONFIG.behavioral.random_seed
        random.seed(seed)
        np.random.seed(seed)

    def _next_merchant(self, current: str) -> str:
        transitions = self.MERCHANT_TRANSITIONS.get(current, {"transfer": 1.0})
        cats = list(transitions.keys())
        probs = list(transitions.values())
        total = sum(probs)
        return np.random.choice(cats, p=[p / total for p in probs])

    def generate_for_user(
        self,
        user: UserProfile,
        n_transactions: Optional[int] = None,
    ) -> list[dict]:
        if n_transactions is None:
            n_transactions = random.randint(
                CONFIG.behavioral.min_transactions_per_user,
                CONFIG.behavioral.max_transactions_per_user,
            )

        # Start from a random point in the past (up to 180 days ago)
        start_date = datetime.utcnow() - timedelta(days=random.randint(30, 180))
        current_time = start_date
        transactions = []
        current_merchant = random.choice(list(self.MERCHANT_TRANSITIONS.keys()))

        for i in range(n_transactions):
            # Time delta: exponential inter-arrival based on frequency
            hours_until_next = np.random.exponential(
                scale=24.0 / user.transaction_frequency
            )
            current_time += timedelta(hours=hours_until_next)

            # Override hour with user's active hours occasionally
            if random.random() < 0.7:
                hour = _sample_hour(user.active_hours)
                current_time = current_time.replace(hour=hour)

            # Markov merchant transition
            current_merchant = self._next_merchant(current_merchant)

            txn = _build_transaction(
                user=user,
                timestamp=current_time,
                is_new_device=random.random() < 0.02,      # 2% chance new device
                is_new_recipient=random.random() < 0.15,   # 15% chance new recipient
                override_merchant=current_merchant,
            )
            transactions.append(txn)

        return transactions

    def generate_all(self, profiles: list[UserProfile]) -> list[dict]:
        all_transactions = []
        for profile in profiles:
            txns = self.generate_for_user(profile)
            all_transactions.extend(txns)
        return all_transactions