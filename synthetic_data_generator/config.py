import os
from pathlib import Path
from dataclasses import dataclass, field


BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BehavioralConfig:
    n_users: int = 2000
    n_fraud_users: int = 200
    min_transactions_per_user: int = 30
    max_transactions_per_user: int = 150
    fraud_injection_rate: float = 0.08     # 8% of sequences contain anomalies
    seq_len: int = 50
    feature_dim: int = 32
    random_seed: int = 42


@dataclass
class GraphConfig:
    n_legit_accounts: int = 5000
    n_mule_accounts: int = 300
    n_fraud_rings: int = 20
    min_ring_size: int = 3
    max_ring_size: int = 15
    mule_hop_depth: int = 3            # How many hops before funds exit
    ba_attachment: int = 3             # Barabasi-Albert m parameter
    random_seed: int = 42


@dataclass
class PayloadConfig:
    n_legit_payloads: int = 10000
    n_anomalous_payloads: int = 2000
    anomaly_type_distribution: dict = field(default_factory=lambda: {
        "replayed_ref": 0.20,
        "type_mismatch": 0.20,
        "missing_field": 0.15,
        "negative_amount": 0.10,
        "malformed_ip": 0.10,
        "timestamp_inconsistency": 0.15,
        "oversized_amount": 0.10,
    })
    random_seed: int = 42


@dataclass
class SIMSwapConfig:
    n_users: int = 2000
    n_sim_swap_events: int = 400
    history_length: int = 30           # Device events per user before swap
    post_swap_window: int = 5          # Transactions after swap to flag
    legitimate_upgrade_rate: float = 0.10   # 10% are real phone upgrades
    random_seed: int = 42


@dataclass
class KYCConfig:
    n_legitimate_docs: int = 5000
    n_forged_docs: int = 1000
    feature_dim: int = 128
    forgery_type_distribution: dict = field(default_factory=lambda: {
        "dpi_inconsistency": 0.20,
        "exif_mismatch": 0.20,
        "font_anomaly": 0.20,
        "biometric_divergence": 0.20,
        "ai_generated": 0.20,
    })
    random_seed: int = 42


@dataclass
class SyntheticDataConfig:
    behavioral: BehavioralConfig = field(default_factory=BehavioralConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    payload: PayloadConfig = field(default_factory=PayloadConfig)
    sim_swap: SIMSwapConfig = field(default_factory=SIMSwapConfig)
    kyc: KYCConfig = field(default_factory=KYCConfig)
    output_dir: Path = OUTPUT_DIR


# Singleton config — import this everywhere
CONFIG = SyntheticDataConfig()


# Nigerian context distributions (calibrated from PaySim + GSMA reports)
NIGERIAN_TRANSACTION_AMOUNTS = {
    "mean": 15000,       # NGN
    "std": 45000,
    "min": 100,
    "max": 5_000_000,
}

MERCHANT_CATEGORY_WEIGHTS = {
    "transfer":   0.40,
    "airtime":    0.15,
    "data":       0.10,
    "bills":      0.10,
    "shopping":   0.08,
    "food":       0.07,
    "transport":  0.05,
    "crypto":     0.02,
    "pos":        0.02,
    "unknown":    0.01,
}

ACTIVE_HOURS_WEIGHTS = {  # Hour: relative weight
    0: 0.5, 1: 0.3, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.3,
    6: 1.0, 7: 2.5, 8: 3.5, 9: 4.0, 10: 4.0, 11: 3.5,
    12: 3.0, 13: 3.5, 14: 4.0, 15: 4.0, 16: 3.5, 17: 3.0,
    18: 3.5, 19: 3.0, 20: 2.5, 21: 2.0, 22: 1.5, 23: 1.0,
}

NIGERIAN_STATES = [
    "Lagos", "Abuja", "Kano", "Rivers", "Oyo", "Kaduna",
    "Anambra", "Delta", "Ogun", "Enugu",
]

NIGERIAN_CARRIERS = ["MTN", "Airtel", "Glo", "9mobile"]

DEVICE_OS = ["Android", "iOS"]
DEVICE_BRANDS = ["Samsung", "Tecno", "Infinix", "iPhone", "Itel", "Nokia"]