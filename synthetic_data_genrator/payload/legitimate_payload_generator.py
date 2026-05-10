import uuid
import random
from datetime import datetime, timedelta

from synthetic_data_generator.config import CONFIG, NIGERIAN_CARRIERS
from synthetic_data_generator.payload.squad_payload_schema import SquadPayloadSchema


SCHEMA = SquadPayloadSchema()

NIGERIAN_NAMES = [
    "Amaka Okonkwo", "Emeka Eze", "Fatima Bello", "Tunde Adeyemi",
    "Ngozi Obi", "Chidi Nwosu", "Bola Adesanya", "Kemi Afolabi",
    "Musa Ibrahim", "Aisha Suleiman", "Taiwo Olawale", "Seun Adebayo",
]


def _random_transaction_ref() -> str:
    import string
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=12))


def _random_ip() -> str:
    # Nigerian IP ranges (approximate)
    prefixes = ["102.", "41.190.", "105.112.", "197.210.", "154.120."]
    prefix = random.choice(prefixes)
    suffix = ".".join(str(random.randint(0, 255)) for _ in range(4 - prefix.count(".")))
    return prefix + suffix


def _random_email() -> str:
    names = ["emeka", "amaka", "tunde", "ngozi", "fatima", "chidi", "bola", "musa"]
    domains = ["gmail.com", "yahoo.com", "outlook.com"]
    return f"{random.choice(names)}{random.randint(10, 999)}@{random.choice(domains)}"


class LegitimatePayloadGenerator:
    """
    Generates syntactically and semantically valid Squad API payloads.
    Uses realistic Nigerian phone numbers, emails, bank account numbers,
    and transaction references. Forms the negative (non-fraud) class
    for CNN-GNN training.
    """

    def __init__(self, seed: int = None):
        seed = seed or CONFIG.payload.random_seed
        random.seed(seed)

    def generate_one(self, timestamp: datetime = None) -> dict:
        timestamp = timestamp or (
            datetime.utcnow() - timedelta(days=random.randint(0, 90))
        )
        amount = random.randint(10000, 50_000_000)  # 100 NGN to 500,000 NGN in kobo

        payload = {
            "transaction_ref": _random_transaction_ref(),
            "amount": amount,
            "currency": "NGN",
            "customer_email": _random_email(),
            "customer_name": random.choice(NIGERIAN_NAMES),
            "customer_phone": f"080{random.randint(10000000, 99999999)}",
            "ip_address": _random_ip(),
            "device_id": str(uuid.uuid4()),
            "channel": random.choice(["app", "ussd", "web", "pos", "payment_link"]),
            "merchant_category": random.choice([
                "transfer", "airtime", "data", "bills", "shopping", "food"
            ]),
            "transaction_type": "debit",
            "created_at": timestamp.isoformat() + "Z",
            "narration": random.choice([
                "Transfer to friend", "Airtime purchase", "Bill payment",
                "Online shopping", "Data subscription", "Rent payment",
            ]),
            "meta": {
                "receiver_account": str(random.randint(1000000000, 9999999999)),
                "receiver_bank_code": random.choice(["058", "044", "057", "011", "033"]),
            },
            "label": 0,
        }

        is_valid, errors = SCHEMA.validate(payload)
        assert is_valid, f"Generated invalid payload: {errors}"
        return payload

    def generate_batch(self, n: int = None) -> list[dict]:
        n = n or CONFIG.payload.n_legit_payloads
        return [self.generate_one() for _ in range(n)]