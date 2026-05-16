import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

SQUAD_BASE_URL = os.getenv("SQUAD_BASE_URL", "https://sandbox-api-d.squadco.com")
SQUAD_SECRET_KEY = os.getenv("SQUAD_SECRET_KEY", "")

HEADERS = {
    "Authorization": f"Bearer {SQUAD_SECRET_KEY}",
    "Content-Type": "application/json",
}


# ── Request models ────────────────────────────────────────────────────────────

class RefundRequest(BaseModel):
    gateway_transaction_ref: str   # from webhook Body.gateway_ref — required by Squad
    transaction_ref: str
    refund_type: str = "Full"      # "Full" or "Partial"
    reason_for_refund: str = "Fraud detected by Hybrid_Temporal_Graph_Transformer AI"
    refund_amount: Optional[str] = None   # kobo string, only for Partial refunds


class ResolveDisputeRequest(BaseModel):
    action: str        # "accepted" or "rejected"
    file_name: str = ""


# ── HTTP helpers ──────────────────────────────────────────────────────────────

async def _post(endpoint: str, body: dict) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{SQUAD_BASE_URL}{endpoint}",
            json=body,
            headers=HEADERS,
        )
    if response.status_code not in (200, 201):
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Squad API error: {response.text}",
        )
    return response.json()


async def _get(endpoint: str, params: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{SQUAD_BASE_URL}{endpoint}",
            headers=HEADERS,
            params=params or {},
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Squad API error: {response.text}",
        )
    return response.json()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/refund")
async def initiate_refund(payload: RefundRequest):
    """
    Initiate a full or partial refund on a completed transaction.
    This is the correct Squad API for reversing a transaction post-settlement.

    gateway_transaction_ref comes from the webhook Body.gateway_ref field —
    the decision engine stores it in the audit log for this purpose.

    NOTE: Squad does not expose a merchant-side freeze/hold API.
    This refund call is the only programmatic remedy after settlement completes.
    """
    body = {
        "gateway_transaction_ref": payload.gateway_transaction_ref,
        "transaction_ref": payload.transaction_ref,
        "refund_type": payload.refund_type,
        "reason_for_refund": payload.reason_for_refund,
    }
    if payload.refund_type == "Partial" and payload.refund_amount:
        body["refund_amount"] = payload.refund_amount

    result = await _post("/transaction/refund", body)
    return {
        "status": "refund_initiated",
        "transaction_ref": payload.transaction_ref,
        "squad_response": result,
    }


@router.get("/disputes")
async def get_all_disputes():
    """
    Retrieve all disputes raised by customers on your transactions.
    Squad's dispute system is customer/chargeback initiated — merchants
    respond to disputes; they do not raise them.
    """
    result = await _get("/dispute")
    return result


@router.post("/disputes/{ticket_id}/resolve")
async def resolve_dispute(ticket_id: str, payload: ResolveDisputeRequest):
    """
    Accept or reject a customer chargeback by ticket_id.
    action must be "accepted" or "rejected".
    If rejecting, supply file_name of previously uploaded evidence.
    """
    if payload.action not in ("accepted", "rejected"):
        raise HTTPException(status_code=400, detail="action must be 'accepted' or 'rejected'.")

    result = await _post(
        f"/dispute/{ticket_id}/resolve",
        {"action": payload.action, "file_name": payload.file_name},
    )
    return {"status": "dispute_resolved", "ticket_id": ticket_id, "squad_response": result}


@router.get("/disputes/upload-url/{ticket_id}/{file_name}")
async def get_dispute_upload_url(ticket_id: str, file_name: str):
    """
    Get a pre-signed URL to upload evidence for rejecting a dispute.
    Call this before resolve_dispute when action is 'rejected'.
    """
    result = await _get(f"/dispute/upload-url/{ticket_id}/{file_name}")
    return result