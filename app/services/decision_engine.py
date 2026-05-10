import os
from enum import Enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from app.models.ensemble import EnsembleScores

if TYPE_CHECKING:
    from fastapi import Request

SQUAD_BASE_URL = os.getenv("SQUAD_BASE_URL", "https://sandbox-api-d.squadco.com")
SQUAD_SECRET_KEY = os.getenv("SQUAD_SECRET_KEY", "")

GREEN_THRESHOLD = 0.65
RED_THRESHOLD = 0.90


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
    Converts EnsembleScores into a routing decision and takes action:
    - GREEN  (<0.65): Transaction proceeds.
    - AMBER (0.65-0.89): Step-up authentication triggered. Settlement held.
    - RED   (>=0.90): Squad Dispute API called. Funds frozen immediately.
    """

    def _determine_zone(self, score: float) -> FraudZone:
        if score < GREEN_THRESHOLD:
            return FraudZone.GREEN
        elif score < RED_THRESHOLD:
            return FraudZone.AMBER
        return FraudZone.RED

    def _identify_top_signals(self, scores: EnsembleScores) -> list[str]:
        """Return the model names firing above 0.70, sorted by score."""
        signal_map = {
            "Behavioral anomaly (Transformer)": scores.transformer_score,
            "Mule network detected (GraphSAGE)": scores.graphsage_score,
            "Payload tampering (CNN-GNN)": scores.cnn_gnn_score,
            "SIM swap event (TSSGC)": scores.tssgc_score,
            "KYC document fraud (GAN)": scores.gan_autoencoder_score,
        }
        firing = {k: v for k, v in signal_map.items() if v >= 0.70}
        return sorted(firing, key=lambda k: firing[k], reverse=True)

    async def _call_squad_dispute_api(self, transaction_ref: str, reason: str):
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{SQUAD_BASE_URL}/dispute/transaction/raise-dispute",
                json={"transaction_ref": transaction_ref, "reason": reason},
                headers={
                    "Authorization": f"Bearer {SQUAD_SECRET_KEY}",
                    "Content-Type": "application/json",
                },
            )

    async def _trigger_step_up_auth(self, transaction_ref: str, body: dict):
        """
        Amber Zone: Hold settlement and notify the user to re-authenticate.
        In production, this calls an OTP or Face ID middleware endpoint.
        """
        print(f"🔶 AMBER: Step-up auth triggered for {transaction_ref}")
        # TODO: integrate with OTP/Face ID service

    async def decide(
        self,
        transaction_ref: str,
        scores: EnsembleScores,
        body: dict,
        request,
    ) -> DecisionResult:
        zone = self._determine_zone(scores.unified_score)
        top_signals = self._identify_top_signals(scores)

        if zone == FraudZone.RED:
            await self._call_squad_dispute_api(
                transaction_ref,
                reason=f"TrustGuard RED ZONE: unified score {scores.unified_score:.3f}. "
                       f"Signals: {', '.join(top_signals)}",
            )
            action = "DISPUTE_RAISED — funds frozen before settlement"

        elif zone == FraudZone.AMBER:
            await self._trigger_step_up_auth(transaction_ref, body)
            action = "STEP_UP_AUTH — settlement held pending re-verification"

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