import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from app.models.ensemble import EnsembleScores


@dataclass
class DriftEvent:
    feature_index: int
    old_mean: float
    new_mean: float
    magnitude: float
    affected_model: str


class ADWINDetector:
    """
    ADWIN (Adaptive Windowing) drift detector.
    Maintains a sliding window of feature values and detects
    when the mean of the distribution shifts significantly —
    indicating a new fraud pattern is emerging.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window: deque = deque()
        self.variance = 0.0
        self.mean = 0.0
        self.n = 0

    def update(self, value: float) -> bool:
        """
        Add a new value to the window. Returns True if drift detected.
        """
        self.window.append(value)
        self.n += 1

        # Welford's online variance
        old_mean = self.mean
        self.mean += (value - self.mean) / self.n
        self.variance += (value - old_mean) * (value - self.mean)

        if self.n < 30:
            return False

        return self._check_drift()

    def _check_drift(self) -> bool:
        window_arr = np.array(self.window)
        n = len(window_arr)
        half = n // 2

        mean_a = window_arr[:half].mean()
        mean_b = window_arr[half:].mean()
        var_a = window_arr[:half].var() + 1e-10
        var_b = window_arr[half:].var() + 1e-10

        # ADWIN cut condition
        epsilon_cut = np.sqrt(
            (1 / (2 * half)) * np.log(4 * n / self.delta)
        )
        if abs(mean_a - mean_b) >= epsilon_cut:
            # Drift detected — trim old window
            self.window = deque(list(self.window)[half:])
            self.n = len(self.window)
            return True
        return False


class DriftDetector:
    """
    Monitors the statistical distribution of incoming transaction features
    in real-time. When a significant distributional shift is detected,
    uncertain predictions are routed to the Active Learning Queue.
    """

    FEATURE_NAMES = [
        "amount", "hour_sin", "hour_cos", "day_of_week",
        "is_weekend", "is_new_device", "is_new_recipient",
        "transaction_type", "currency_ngn",
    ]

    MODEL_MAP = {
        0: "transformer",
        1: "graphsage",
        2: "cnn_gnn",
        3: "tssgc",
        4: "gan_autoencoder",
    }

    def __init__(self, delta: float = 0.002):
        self.detectors = {
            name: ADWINDetector(delta=delta)
            for name in self.FEATURE_NAMES
        }
        self.score_detectors = {
            model: ADWINDetector(delta=delta)
            for model in self.MODEL_MAP.values()
        }
        self.drift_events: list[DriftEvent] = []
        self.drift_count = 0

    def observe(self, body: dict, scores: EnsembleScores) -> list[DriftEvent]:
        """
        Feed a new transaction and its scores to the drift detectors.
        Returns any drift events detected this cycle.
        """
        new_events = []

        # Monitor raw features
        feature_values = {
            "amount": float(body.get("amount", 0)) / 10_000_000.0,
            "is_new_device": float(body.get("is_new_device", False)),
            "is_new_recipient": float(body.get("is_new_recipient", False)),
            "currency_ngn": float(body.get("currency", "NGN") == "NGN"),
        }
        for name, value in feature_values.items():
            if name in self.detectors:
                if self.detectors[name].update(value):
                    event = DriftEvent(
                        feature_index=list(self.detectors).index(name),
                        old_mean=self.detectors[name].mean,
                        new_mean=value,
                        magnitude=abs(self.detectors[name].mean - value),
                        affected_model=self._infer_affected_model(name),
                    )
                    self.drift_events.append(event)
                    new_events.append(event)
                    self.drift_count += 1
                    print(f" Drift detected on feature '{name}'")

        # Monitor model score distributions
        score_map = {
            "transformer": scores.transformer_score,
            "graphsage": scores.graphsage_score,
            "cnn_gnn": scores.cnn_gnn_score,
            "tssgc": scores.tssgc_score,
            "gan_autoencoder": scores.gan_autoencoder_score,
        }
        for model, score in score_map.items():
            if self.score_detectors[model].update(score):
                print(f" Score distribution drift on model '{model}'")

        return new_events

    def _infer_affected_model(self, feature_name: str) -> str:
        if feature_name in ("amount", "is_new_recipient"):
            return "transformer"
        if feature_name == "is_new_device":
            return "tssgc"
        return "graphsage"

    def is_uncertain(self, scores: EnsembleScores) -> bool:
        """
        Returns True if the unified score is in the uncertainty band (0.45–0.75).
        Uncertain predictions are routed to Active Learning Queue.
        """
        return 0.45 <= scores.unified_score <= 0.75

    def get_summary(self) -> dict:
        return {
            "total_drift_events": self.drift_count,
            "recent_events": [
                {
                    "feature_index": e.feature_index,
                    "magnitude": round(e.magnitude, 4),
                    "affected_model": e.affected_model,
                }
                for e in self.drift_events[-10:]
            ],
        }