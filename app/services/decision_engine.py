import os
from enum import Enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from app.models.ensemble import EnsembleScores

if TYPE_CHECKING:
    from fastapi import Request

NEOBANK_BASE_URL = os.getenv("NEOBANK_BASE_URL", "https://api.neobank.example")
NEOBANK_SECRET_KEY = os.getenv("NEOBANK_SECRET_KEY", "")

GREEN_THRESHOLD = float(os.getenv("GREEN_THRESHOLD", "0.65"))
RED_THRESHOLD = float(os.getenv("RED_THRESHOLD", "0.90"))


class FraudZone(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


@dataclass
class DecisionResult:
    transaction_ref: str
    unified_score: float
    zone: FraudZone
    action_taken: str
    scores_breakdown: dict
    top_signals: list[str]


class DecisionEngine:
    """
    Converts EnsembleScores into a routing decision and takes action.

    """

    def _determine_zone(self, score: float) -> FraudZone:
        if score < GREEN_THRESHOLD:
            return FraudZone.GREEN
        elif score < RED_THRESHOLD:
            return FraudZone.AMBER
        return FraudZone.RED

    def _identify_top_signals(self, scores: EnsembleScores) -> list[str]:
        signal_map = {
            "Behavioral anomaly (Transformer)": scores.transformer_score,
            "Mule network detected (GraphSAGE)": scores.graphsage_score,
            "Payload tampering (CNN-GNN)": scores.cnn_gnn_score,
            "SIM swap event (TSSGC)": scores.tssgc_score,
            "KYC document fraud (GAN)": scores.gan_autoencoder_score,
        }
        firing = {k: v for k, v in signal_map.items() if v >= 0.70}
        return sorted(firing, key=lambda k: firing[k], reverse=True)

    async def _initiate_refund(
        self,
        transaction_ref: str,
        gateway_ref: str,
        reason: str,
    ) -> dict:
        """
        Call the backend refund API to reverse a fraudulent transaction post-settlement.
        Requires gateway_ref from the webhook body — without it, the refund cannot proceed.
        """
        if not gateway_ref:
            print(
                f"  RED ZONE [{transaction_ref}]: No gateway_ref available — "
                f"refund cannot be submitted to the backend. Transaction flagged for manual review."
            )
            return {"status": "skipped", "reason": "gateway_ref missing"}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{NEOBANK_BASE_URL}/transaction/refund",
                    json={
                        "gateway_transaction_ref": gateway_ref,
                        "transaction_ref": transaction_ref,
                        "refund_type": "Full",
                        "reason_for_refund": reason,
                    },
                    headers={
                        "Authorization": f"Bearer {NEOBANK_SECRET_KEY}",
                        "Content-Type": "application/json",
                    },
                )
            result = response.json()
            print(f" RED ZONE [{transaction_ref}]: Refund submitted — {result}")
            return result
        except Exception as e:
            print(f" RED ZONE [{transaction_ref}]: Refund API call failed — {e}. Flagged for manual review.")
            return {"status": "error", "detail": str(e)}

    async def _trigger_step_up_auth(self, transaction_ref: str, body: dict):
        """
        Amber Zone: flag for manual review and notify for re-authentication.
        In production, integrate with your OTP or Face ID middleware here.
        """
        print(f" AMBER [{transaction_ref}]: Flagged for step-up auth. Action required.")
        # TODO: POST to OTP/FaceID service with customer contact details

    async def decide(
        self,
        transaction_ref: str,
        scores: EnsembleScores,
        body: dict,
        request,
    ) -> DecisionResult:
        zone = self._determine_zone(scores.unified_score)
        top_signals = self._identify_top_signals(scores)

        # gateway_ref is extracted during webhook normalisation and stored in body
        gateway_ref = body.get("gateway_ref", "")

        if zone == FraudZone.RED:
            reason = (
                f"TGT RED ZONE: unified score {scores.unified_score:.3f}. "
                f"Signals: {', '.join(top_signals) if top_signals else 'ensemble threshold exceeded'}"
            )
            await self._initiate_refund(transaction_ref, gateway_ref, reason)
            action = f"REFUND_INITIATED — {reason}"

        elif zone == FraudZone.AMBER:
            await self._trigger_step_up_auth(transaction_ref, body)
            action = "STEP_UP_AUTH — transaction flagged, pending re-verification"

        else:
            action = "APPROVED — transaction proceeds normally"

        return DecisionResult(
            transaction_ref=transaction_ref,
            unified_score=scores.unified_score,
            zone=zone,
            action_taken=action,
            scores_breakdown=scores.to_dict(),
            top_signals=top_signals,
        )