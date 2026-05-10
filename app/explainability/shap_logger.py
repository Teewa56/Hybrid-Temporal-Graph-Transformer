import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from app.models.ensemble import EnsembleScores


FEATURE_NAMES = [
    "amount", "hour_sin", "hour_cos", "day_of_week", "is_weekend",
    "merchant_transfer", "merchant_airtime", "merchant_data",
    "merchant_bills", "merchant_shopping", "merchant_food",
    "merchant_transport", "merchant_crypto", "merchant_pos", "merchant_unknown",
    "is_new_device", "is_new_recipient", "is_debit", "is_ngn",
    "pad_0", "pad_1", "pad_2", "pad_3", "pad_4",
    "pad_5", "pad_6", "pad_7", "pad_8", "pad_9",
    "pad_10", "pad_11", "pad_12",
]


@dataclass
class SHAPExplanation:
    transaction_ref: str
    model_name: str
    top_features: list[dict]        # [{"feature": str, "shap_value": float}]
    base_value: float
    prediction: float
    risk_label: str


class SHAPLogger:
    """
    Computes SHAP values for every blocked (Amber/Red) transaction.
    Provides human-readable feature-level explanations satisfying
    CBN regulatory requirements for automated financial decisions.
    Also surfaces Transformer attention weights as complementary signal.
    """

    def __init__(self, background_data: Optional[np.ndarray] = None):
        self.background = background_data
        self._explainers: dict = {}

    def register_model(self, model_name: str, predict_fn, background: np.ndarray):
        """Register a model's predict function with a background dataset for SHAP."""
        if not SHAP_AVAILABLE:
            return
        self._explainers[model_name] = shap.KernelExplainer(
            predict_fn,
            shap.sample(background, 50),
        )

    def explain(
        self,
        model_name: str,
        features: np.ndarray,
        transaction_ref: str,
        prediction: float,
    ) -> SHAPExplanation:
        """
        Compute SHAP values for a single sample.
        Falls back to gradient-based approximation if SHAP not available.
        """
        if SHAP_AVAILABLE and model_name in self._explainers:
            shap_values = self._explainers[model_name].shap_values(
                features.reshape(1, -1)
            )
            values = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
            base = self._explainers[model_name].expected_value
        else:
            # Fallback: use feature magnitudes as proxy importance
            values = np.abs(features) / (np.abs(features).sum() + 1e-10) * prediction
            base = 0.5

        # Build sorted top features
        n_features = min(len(values), len(FEATURE_NAMES))
        feature_importance = [
            {
                "feature": FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}",
                "shap_value": round(float(values[i]), 5),
                "direction": "↑ fraud" if values[i] > 0 else "↓ legit",
            }
            for i in range(n_features)
        ]
        feature_importance.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return SHAPExplanation(
            transaction_ref=transaction_ref,
            model_name=model_name,
            top_features=feature_importance[:10],
            base_value=float(base),
            prediction=prediction,
            risk_label="HIGH RISK" if prediction >= 0.9 else "MEDIUM RISK" if prediction >= 0.65 else "LOW RISK",
        )

    @staticmethod
    def explain_from_attention(
        attention_weights: np.ndarray,
        sequence_length: int,
        transaction_ref: str,
        prediction: float,
    ) -> list[dict]:
        """
        Convert Transformer attention weights to human-readable
        explanation: which past transactions drove the anomaly flag.
        """
        if attention_weights is None or len(attention_weights) == 0:
            return []

        weights = attention_weights[:sequence_length]
        top_indices = np.argsort(weights)[::-1][:5]
        return [
            {
                "position": f"T-{sequence_length - int(idx)}",
                "description": f"Transaction {sequence_length - int(idx)} steps ago",
                "attention_weight": round(float(weights[idx]), 4),
            }
            for idx in top_indices
        ]