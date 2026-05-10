import asyncio
import numpy as np
from dataclasses import dataclass


@dataclass
class EnsembleScores:
    transformer_score: float     # Behavioral anomaly
    graphsage_score: float       # Social engineering / mule network
    cnn_gnn_score: float         # Payment injection
    tssgc_score: float           # SIM swap
    gan_autoencoder_score: float # KYC fraud
    unified_score: float         # Weighted aggregate
    weights: dict

    def to_dict(self) -> dict:
        return {
            "transformer": round(self.transformer_score, 4),
            "graphsage": round(self.graphsage_score, 4),
            "cnn_gnn": round(self.cnn_gnn_score, 4),
            "tssgc": round(self.tssgc_score, 4),
            "gan_autoencoder": round(self.gan_autoencoder_score, 4),
            "unified": round(self.unified_score, 4),
            "weights": self.weights,
        }


# Default weights — can be tuned per deployment context
DEFAULT_WEIGHTS = {
    "transformer": 0.25,
    "graphsage": 0.30,
    "cnn_gnn": 0.15,
    "tssgc": 0.20,
    "gan_autoencoder": 0.10,
}


class ModelEnsemble:
    """
    Runs all five fraud detection sub-architectures in parallel
    and aggregates their outputs into a Unified Fraud Score.
    Weights are configurable and can be adapted per transaction context.
    """

    def __init__(self, weights: dict = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self._normalize_weights()

    def _normalize_weights(self):
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def compute_unified_score(
        self,
        transformer_score: float,
        graphsage_score: float,
        cnn_gnn_score: float,
        tssgc_score: float,
        gan_autoencoder_score: float,
    ) -> EnsembleScores:
        scores = np.array([
            transformer_score,
            graphsage_score,
            cnn_gnn_score,
            tssgc_score,
            gan_autoencoder_score,
        ])
        weight_arr = np.array([
            self.weights["transformer"],
            self.weights["graphsage"],
            self.weights["cnn_gnn"],
            self.weights["tssgc"],
            self.weights["gan_autoencoder"],
        ])

        # Weighted average unified score
        unified = float(np.dot(scores, weight_arr))

        # Boost score if multiple models agree strongly (correlation penalty)
        agreement_bonus = 0.0
        high_confidence = scores[scores > 0.75]
        if len(high_confidence) >= 3:
            agreement_bonus = 0.05  # Lift when 3+ models fire together

        unified = min(unified + agreement_bonus, 1.0)

        return EnsembleScores(
            transformer_score=transformer_score,
            graphsage_score=graphsage_score,
            cnn_gnn_score=cnn_gnn_score,
            tssgc_score=tssgc_score,
            gan_autoencoder_score=gan_autoencoder_score,
            unified_score=unified,
            weights=self.weights,
        )

    def adjust_weights_for_context(self, transaction_body: dict) -> None:
        """
        Dynamically adjust model weights based on transaction context.
        E.g., if KYC was just submitted, boost GAN weight.
            if it's a USSD transfer, boost TSSGC weight.
        """
        channel = transaction_body.get("channel", "")
        is_new_account = transaction_body.get("is_new_account", False)

        weights = self.weights.copy()

        if channel == "ussd":
            weights["tssgc"] += 0.05
            weights["transformer"] -= 0.05

        if is_new_account:
            weights["gan_autoencoder"] += 0.05
            weights["graphsage"] += 0.05
            weights["transformer"] -= 0.10

        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}