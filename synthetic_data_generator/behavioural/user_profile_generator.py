import uuid
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from synthetic_data_generator.config import (
    CONFIG,
    MERCHANT_CATEGORY_WEIGHTS,
    ACTIVE_HOURS_WEIGHTS,
    NIGERIAN_STATES,
    NIGERIAN_CARRIERS,
    NIGERIAN_TRANSACTION_AMOUNTS,
    DEVICE_OS,
    DEVICE_BRANDS,
)


@dataclass
class UserProfile:
    user_id: str
    email: str
    phone: str
    state: str
    carrier: str
    device_id: str
    device_os: str
    device_brand: str
    is_fraud_user: bool

    # Spending fingerprint
    typical_amount_mean: float
    typical_amount_std: float
    active_hours: list[int]           # Preferred transaction hours
    merchant_weights: dict            # Category probability distribution
    transaction_frequency: float      # Mean txns per day
    typical_recipients: list[str]     # Known recipient account IDs
    home_state: str


def _random_nigerian_phone() -> str:
    prefixes = ["0803", "0806", "0813", "0816", "0703", "0706",
                "0813", "0814", "0903", "0906", "0805", "0705"]
    return random.choice(prefixes) + "".join([str(random.randint(0, 9)) for _ in range(7)])


def _random_email(uid: str) -> str:
    domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]
    names = ["john", "amaka", "chidi", "fatima", "emeka", "bola", "tunde", "ngozi"]
    return f"{random.choice(names)}{uid[:4]}@{random.choice(domains)}"


def _sample_merchant_weights() -> dict:
    """
    Generate a user-specific merchant distribution by perturbing the
    population-level weights — each user has a slightly different spending mix.
    """
    base = list(MERCHANT_CATEGORY_WEIGHTS.values())
    noise = np.random.dirichlet(np.array(base) * 10)  # Concentrate around base
    categories = list(MERCHANT_CATEGORY_WEIGHTS.keys())
    return dict(zip(categories, noise.tolist()))


def _sample_active_hours() -> list[int]:
    """Sample 4-8 preferred transaction hours weighted by Nigerian activity patterns."""
    hours = list(ACTIVE_HOURS_WEIGHTS.keys())
    weights = [ACTIVE_HOURS_WEIGHTS[h] for h in hours]
    weights = np.array(weights) / sum(weights)
    n = random.randint(4, 8)
    return list(np.random.choice(hours, size=n, replace=False, p=weights))


class UserProfileGenerator:
    """
    Generates synthetic user profiles with consistent spending fingerprints.
    Each profile defines what "normal" looks like for that user —
    the baseline the Transformer model learns to defend.
    """

    def __init__(self, seed: int = None):
        seed = seed or CONFIG.behavioral.random_seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_one(self, is_fraud_user: bool = False) -> UserProfile:
        uid = str(uuid.uuid4())[:8]
        state = random.choice(NIGERIAN_STATES)

        # Fraud users tend to have slightly higher, more erratic amounts
        amount_mean = (
            np.random.lognormal(mean=10.5, sigma=0.8)
            if is_fraud_user
            else np.random.lognormal(mean=9.5, sigma=0.6)
        )
        amount_std = amount_mean * random.uniform(0.3, 1.2)

        return UserProfile(
            user_id=uid,
            email=_random_email(uid),
            phone=_random_nigerian_phone(),
            state=state,
            carrier=random.choice(NIGERIAN_CARRIERS),
            device_id=str(uuid.uuid4()),
            device_os=random.choice(DEVICE_OS),
            device_brand=random.choice(DEVICE_BRANDS),
            is_fraud_user=is_fraud_user,
            typical_amount_mean=round(amount_mean, 2),
            typical_amount_std=round(amount_std, 2),
            active_hours=_sample_active_hours(),
            merchant_weights=_sample_merchant_weights(),
            transaction_frequency=round(random.uniform(0.5, 5.0), 2),
            typical_recipients=[str(uuid.uuid4())[:8] for _ in range(random.randint(2, 8))],
            home_state=state,
        )

    def generate_batch(
        self,
        n_users: int = None,
        n_fraud_users: int = None,
    ) -> list[UserProfile]:
        n_users = n_users or CONFIG.behavioral.n_users
        n_fraud = n_fraud_users or CONFIG.behavioral.n_fraud_users

        profiles = (
            [self.generate_one(is_fraud_user=False) for _ in range(n_users - n_fraud)]
            + [self.generate_one(is_fraud_user=True) for _ in range(n_fraud)]
        )
        random.shuffle(profiles)
        return profiles