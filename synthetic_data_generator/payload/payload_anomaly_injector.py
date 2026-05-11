import uuid
import random
import string
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Optional

from synthetic_data_generator.config import CONFIG
from synthetic_data_generator.payload.squad_payload_schema import SquadPayloadSchema


SCHEMA = SquadPayloadSchema()


class PayloadAnomalyInjector:
    """
    Takes legitimate Squad API payloads and injects labeled anomalies.
    Each anomaly type maps to a real payment injection attack pattern.
    """

    ANOMALY_TYPES = list(CONFIG.payload.anomaly_type_distribution.keys())

    def __init__(self, seed: int = None):
        seed = seed or CONFIG.payload.random_seed
        random.seed(seed)

    def inject_replayed_ref(self, payload: dict, seen_refs: set) -> dict:
        """Replay an existing transaction_ref — classic replay attack."""
        p = deepcopy(payload)
        if seen_refs:
            p["transaction_ref"] = random.choice(list(seen_refs))
        p["label"] = 1
        p["anomaly_type"] = "replayed_ref"
        return p

    def inject_type_mismatch(self, payload: dict) -> dict:
        """Send amount as string instead of int — breaks type validation."""
        p = deepcopy(payload)
        p["amount"] = str(p["amount"]) + ".00"
        p["label"] = 1
        p["anomaly_type"] = "type_mismatch"
        return p

    def inject_missing_field(self, payload: dict) -> dict:
        """Drop a required field — tests field presence validation."""
        p = deepcopy(payload)
        field_to_drop = random.choice(SCHEMA.REQUIRED_FIELDS)
        p.pop(field_to_drop, None)
        p["label"] = 1
        p["anomaly_type"] = "missing_field"
        return p

    def inject_negative_amount(self, payload: dict) -> dict:
        """Negative or zero amount — financial logic bypass attempt."""
        p = deepcopy(payload)
        p["amount"] = random.choice([-abs(p["amount"]), 0, -1])
        p["label"] = 1
        p["anomaly_type"] = "negative_amount"
        return p

    def inject_malformed_ip(self, payload: dict) -> dict:
        """Invalid IP address format."""
        p = deepcopy(payload)
        malformed_ips = [
            "999.999.999.999",
            "0.0.0.0",
            "not_an_ip",
            "256.1.1.1",
            "",
            "::1",            # IPv6 in an IPv4-expected field
        ]
        p["ip_address"] = random.choice(malformed_ips)
        p["label"] = 1
        p["anomaly_type"] = "malformed_ip"
        return p

    def inject_timestamp_inconsistency(self, payload: dict) -> dict:
        """Timestamp in the future or impossibly old."""
        p = deepcopy(payload)
        choice = random.randint(0, 2)
        if choice == 0:
            # Far future
            future = datetime.utcnow() + timedelta(days=random.randint(1, 365))
            p["created_at"] = future.isoformat() + "Z"
        elif choice == 1:
            # Before Squad existed
            past = datetime(2015, 1, 1)
            p["created_at"] = past.isoformat() + "Z"
        else:
            # Completely invalid format
            p["created_at"] = "not-a-timestamp"
        p["label"] = 1
        p["anomaly_type"] = "timestamp_inconsistency"
        return p

    def inject_oversized_amount(self, payload: dict) -> dict:
        """Amount exceeding the schema maximum — potential overflow attack."""
        p = deepcopy(payload)
        p["amount"] = random.choice([
            999_999_999_999,
            2**31 - 1,
            2**63,
        ])
        p["label"] = 1
        p["anomaly_type"] = "oversized_amount"
        return p

    def inject(
        self,
        payload: dict,
        anomaly_type: Optional[str] = None,
        seen_refs: set = None,
    ) -> dict:
        """Inject a random (or specified) anomaly."""
        if anomaly_type is None:
            dist = CONFIG.payload.anomaly_type_distribution
            anomaly_type = random.choices(
                list(dist.keys()), weights=list(dist.values()), k=1
            )[0]

        seen_refs = seen_refs or set()

        handler = {
            "replayed_ref":            lambda p: self.inject_replayed_ref(p, seen_refs),
            "type_mismatch":           self.inject_type_mismatch,
            "missing_field":           self.inject_missing_field,
            "negative_amount":         self.inject_negative_amount,
            "malformed_ip":            self.inject_malformed_ip,
            "timestamp_inconsistency": self.inject_timestamp_inconsistency,
            "oversized_amount":        self.inject_oversized_amount,
        }
        return handler[anomaly_type](payload)

    def inject_batch(
        self,
        legit_payloads: list[dict],
        n_anomalous: int = None,
    ) -> list[dict]:
        """
        Generate N anomalous payloads by injecting anomalies into
        copies of legitimate payloads. Returns only the anomalous set.
        """
        n_anomalous = n_anomalous or CONFIG.payload.n_anomalous_payloads
        seen_refs = {p["transaction_ref"] for p in legit_payloads}
        result = []
        for _ in range(n_anomalous):
            base = deepcopy(random.choice(legit_payloads))
            result.append(self.inject(base, seen_refs=seen_refs))
        return result