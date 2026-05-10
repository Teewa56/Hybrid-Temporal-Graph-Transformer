import uuid
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from copy import deepcopy

from synthetic_data_generator.config import CONFIG, NIGERIAN_STATES
from synthetic_data_generator.behavioral.user_profile_generator import UserProfile


class AnomalyInjector:
    """
    Takes clean transaction sequences and injects labeled fraud events.
    Each anomaly type maps to a real-world fraud pattern seen in African FinTechs.

    Anomaly types:
    - large_late_transfer   : Large transfer at unusual hour from new device
    - velocity_spike        : 5+ transactions in under 10 minutes
    - location_jump         : Transaction from a state far from home state
    - new_device_large      : Unusually large transfer from a new device
    - out_of_category       : Sudden crypto/international transfer for a non-crypto user
    - rapid_recipient_churn : Multiple new recipients in rapid succession
    """

    ANOMALY_TYPES = [
        "large_late_transfer",
        "velocity_spike",
        "location_jump",
        "new_device_large",
        "out_of_category",
        "rapid_recipient_churn",
    ]

    def __init__(self, seed: int = None):
        seed = seed or CONFIG.behavioral.random_seed
        random.seed(seed)
        np.random.seed(seed)

    def inject_large_late_transfer(
        self, sequence: list[dict], user: UserProfile
    ) -> list[dict]:
        """3 AM large transfer from a new device to a new recipient."""
        seq = deepcopy(sequence)
        if not seq:
            return seq

        inject_at = random.randint(len(seq) // 2, len(seq) - 1)
        base_time = datetime.fromisoformat(seq[inject_at]["created_at"].replace("Z", ""))
        anomaly_time = base_time.replace(hour=random.randint(1, 4), minute=random.randint(0, 59))

        anomaly = deepcopy(seq[inject_at])
        anomaly.update({
            "transaction_ref": str(uuid.uuid4()).replace("-", "")[:12].upper(),
            "amount": int(user.typical_amount_mean * random.uniform(5, 15) * 100),
            "created_at": anomaly_time.isoformat() + "Z",
            "device_id": str(uuid.uuid4()),
            "is_new_device": True,
            "is_new_recipient": True,
            "recipient_id": str(uuid.uuid4())[:8],
            "merchant_category": "transfer",
            "label": 1,
        })
        seq.insert(inject_at + 1, anomaly)
        return seq

    def inject_velocity_spike(
        self, sequence: list[dict], user: UserProfile
    ) -> list[dict]:
        """5 transactions within 8 minutes — account takeover pattern."""
        seq = deepcopy(sequence)
        if not seq:
            return seq

        inject_at = random.randint(len(seq) // 2, len(seq) - 1)
        base_time = datetime.fromisoformat(seq[inject_at]["created_at"].replace("Z", ""))
        burst = []

        for i in range(5):
            t = base_time + timedelta(minutes=i * random.uniform(0.5, 2.0))
            txn = deepcopy(seq[inject_at])
            txn.update({
                "transaction_ref": str(uuid.uuid4()).replace("-", "")[:12].upper(),
                "amount": int(random.uniform(5000, 50000) * 100),
                "created_at": t.isoformat() + "Z",
                "is_new_recipient": True,
                "recipient_id": str(uuid.uuid4())[:8],
                "label": 1,
            })
            burst.append(txn)

        for i, txn in enumerate(burst):
            seq.insert(inject_at + 1 + i, txn)
        return seq

    def inject_location_jump(
        self, sequence: list[dict], user: UserProfile
    ) -> list[dict]:
        """Transaction from a different Nigerian state than the user's home."""
        seq = deepcopy(sequence)
        if not seq:
            return seq

        inject_at = random.randint(len(seq) // 2, len(seq) - 1)
        foreign_states = [s for s in NIGERIAN_STATES if s != user.home_state]

        anomaly = deepcopy(seq[inject_at])
        anomaly.update({
            "transaction_ref": str(uuid.uuid4()).replace("-", "")[:12].upper(),
            "state": random.choice(foreign_states),
            "ip_address": f"41.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
            "amount": int(user.typical_amount_mean * random.uniform(3, 8) * 100),
            "label": 1,
        })
        seq.insert(inject_at + 1, anomaly)
        return seq

    def inject_new_device_large(
        self, sequence: list[dict], user: UserProfile
    ) -> list[dict]:
        """Large transfer immediately after a device change."""
        seq = deepcopy(sequence)
        if not seq:
            return seq

        inject_at = random.randint(len(seq) // 2, len(seq) - 1)
        new_device_id = str(uuid.uuid4())

        # First: device change event
        device_change = deepcopy(seq[inject_at])
        device_change.update({
            "transaction_ref": str(uuid.uuid4()).replace("-", "")[:12].upper(),
            "device_id": new_device_id,
            "is_new_device": True,
            "amount": int(random.uniform(500, 2000) * 100),
            "label": 0,  # Device change itself isn't fraud
        })

        # Immediately after: large transfer from that new device
        large_transfer = deepcopy(seq[inject_at])
        base_time = datetime.fromisoformat(seq[inject_at]["created_at"].replace("Z", ""))
        large_transfer.update({
            "transaction_ref": str(uuid.uuid4()).replace("-", "")[:12].upper(),
            "device_id": new_device_id,
            "is_new_device": True,
            "is_new_recipient": True,
            "recipient_id": str(uuid.uuid4())[:8],
            "amount": int(user.typical_amount_mean * random.uniform(8, 20) * 100),
            "created_at": (base_time + timedelta(minutes=3)).isoformat() + "Z",
            "merchant_category": "transfer",
            "label": 1,
        })

        seq.insert(inject_at + 1, device_change)
        seq.insert(inject_at + 2, large_transfer)
        return seq

    def inject_out_of_category(
        self, sequence: list[dict], user: UserProfile
    ) -> list[dict]:
        """Crypto transaction for a user who has never done crypto."""
        seq = deepcopy(sequence)
        if not seq:
            return seq

        inject_at = random.randint(len(seq) // 2, len(seq) - 1)
        anomaly = deepcopy(seq[inject_at])
        anomaly.update({
            "transaction_ref": str(uuid.uuid4()).replace("-", "")[:12].upper(),
            "merchant_category": "crypto",
            "amount": int(user.typical_amount_mean * random.uniform(4, 10) * 100),
            "is_new_recipient": True,
            "recipient_id": str(uuid.uuid4())[:8],
            "label": 1,
        })
        seq.insert(inject_at + 1, anomaly)
        return seq

    def inject_rapid_recipient_churn(
        self, sequence: list[dict], user: UserProfile
    ) -> list[dict]:
        """Multiple new recipients in under 30 minutes — money mule pattern."""
        seq = deepcopy(sequence)
        if not seq:
            return seq

        inject_at = random.randint(len(seq) // 2, len(seq) - 1)
        base_time = datetime.fromisoformat(seq[inject_at]["created_at"].replace("Z", ""))

        for i in range(random.randint(4, 7)):
            t = base_time + timedelta(minutes=i * random.uniform(2, 6))
            txn = deepcopy(seq[inject_at])
            txn.update({
                "transaction_ref": str(uuid.uuid4()).replace("-", "")[:12].upper(),
                "is_new_recipient": True,
                "recipient_id": str(uuid.uuid4())[:8],
                "amount": int(random.uniform(10000, 100000) * 100),
                "created_at": t.isoformat() + "Z",
                "merchant_category": "transfer",
                "label": 1,
            })
            seq.insert(inject_at + 1 + i, txn)
        return seq

    def inject(
        self,
        sequence: list[dict],
        user: UserProfile,
        anomaly_type: Optional[str] = None,
    ) -> list[dict]:
        """Inject a random (or specified) anomaly into a sequence."""
        anomaly_type = anomaly_type or random.choice(self.ANOMALY_TYPES)
        handler = {
            "large_late_transfer":   self.inject_large_late_transfer,
            "velocity_spike":        self.inject_velocity_spike,
            "location_jump":         self.inject_location_jump,
            "new_device_large":      self.inject_new_device_large,
            "out_of_category":       self.inject_out_of_category,
            "rapid_recipient_churn": self.inject_rapid_recipient_churn,
        }
        return handler[anomaly_type](sequence, user)

    def inject_batch(
        self,
        sequences_with_users: list[tuple[list[dict], UserProfile]],
        fraud_rate: float = None,
    ) -> list[list[dict]]:
        """
        Inject anomalies into a fraction of sequences.
        Returns all sequences with fraud labels applied.
        """
        fraud_rate = fraud_rate or CONFIG.behavioral.fraud_injection_rate
        result = []
        for sequence, user in sequences_with_users:
            if random.random() < fraud_rate:
                result.append(self.inject(sequence, user))
            else:
                result.append(sequence)
        return result