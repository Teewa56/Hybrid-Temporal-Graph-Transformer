from dataclasses import dataclass, field
from typing import Any


@dataclass
class SquadPayloadSchema:
    """
    Defines the exact Squad API transaction payload schema.
    Ground truth for what a legitimate payload looks like.
    All generators reference this to ensure structural consistency.
    """

    REQUIRED_FIELDS: list[str] = field(default_factory=lambda: [
        "transaction_ref",
        "amount",
        "currency",
        "customer_email",
        "ip_address",
        "device_id",
        "created_at",
    ])

    OPTIONAL_FIELDS: list[str] = field(default_factory=lambda: [
        "merchant_category",
        "channel",
        "meta",
        "customer_name",
        "customer_phone",
        "transaction_type",
        "recipient_account",
        "narration",
    ])

    FIELD_TYPES: dict[str, type] = field(default_factory=lambda: {
        "transaction_ref":  str,
        "amount":           int,
        "currency":         str,
        "customer_email":   str,
        "ip_address":       str,
        "device_id":        str,
        "created_at":       str,
        "merchant_category": str,
        "channel":          str,
        "customer_name":    str,
        "customer_phone":   str,
        "transaction_type": str,
        "narration":        str,
    })

    FIELD_CONSTRAINTS: dict[str, dict] = field(default_factory=lambda: {
        "amount":       {"min": 10000, "max": 500_000_000},  # Kobo
        "currency":     {"allowed": ["NGN", "USD", "GBP"]},
        "channel":      {"allowed": ["app", "ussd", "web", "pos", "payment_link"]},
        "transaction_type": {"allowed": ["debit", "credit"]},
        "transaction_ref": {"length": 12, "pattern": "ALPHANUMERIC_UPPER"},
        "ip_address":   {"pattern": "IPV4"},
    })

    def validate(self, payload: dict) -> tuple[bool, list[str]]:
        """Validate a payload dict against the schema. Returns (is_valid, errors)."""
        errors = []

        for field_name in self.REQUIRED_FIELDS:
            if field_name not in payload:
                errors.append(f"Missing required field: {field_name}")
            elif payload[field_name] is None:
                errors.append(f"Required field is None: {field_name}")

        for field_name, expected_type in self.FIELD_TYPES.items():
            if field_name in payload and payload[field_name] is not None:
                if not isinstance(payload[field_name], expected_type):
                    errors.append(
                        f"Type mismatch on '{field_name}': "
                        f"expected {expected_type.__name__}, "
                        f"got {type(payload[field_name]).__name__}"
                    )

        constraints = self.FIELD_CONSTRAINTS
        if "amount" in payload and isinstance(payload["amount"], int):
            amt = payload["amount"]
            if amt < constraints["amount"]["min"] or amt > constraints["amount"]["max"]:
                errors.append(f"Amount out of range: {amt}")

        if "currency" in payload:
            if payload["currency"] not in constraints["currency"]["allowed"]:
                errors.append(f"Invalid currency: {payload['currency']}")

        return len(errors) == 0, errors