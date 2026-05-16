import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

NEOBANK_BASE_URL = os.getenv("NEOBANK_BASE_URL", "https://api.neobank.example")
NEOBANK_SECRET_KEY = os.getenv("NEOBANK_SECRET_KEY", "")

HEADERS = {
    "Authorization": f"Bearer {NEOBANK_SECRET_KEY}",
    "Content-Type": "application/json",
}


class RefundRequest(BaseModel):
    transaction_ref: str
    reason: str = "Fraud detected by Hybrid_Temporal_Graph_Transformer AI"
    refund_type: str = "Full"
    refund_amount: Optional[str] = None
    gateway_transaction_ref: Optional[str] = None


class ResolveDisputeRequest(BaseModel):
    action: str
    file_name: str = ""


async def _post(endpoint: str, body: dict) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{NEOBANK_BASE_URL}{endpoint}",
            json=body,
            headers=HEADERS,
        )
    if response.status_code not in (200, 201):
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Backend API error: {response.text}",
        )
    return response.json()


async def _get(endpoint: str, params: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{NEOBANK_BASE_URL}{endpoint}",
            headers=HEADERS,
            params=params or {},
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Backend API error: {response.text}",
        )
    return response.json()


@router.post("/refund")
async def initiate_refund(payload: RefundRequest):
    body = {
        "gateway_transaction_ref": payload.gateway_transaction_ref or payload.transaction_ref,
        "transaction_ref": payload.transaction_ref,
        "refund_type": payload.refund_type,
        "reason_for_refund": payload.reason,
    }
    if payload.refund_type == "Partial" and payload.refund_amount:
        body["refund_amount"] = payload.refund_amount

    result = await _post("/transaction/refund", body)
    return {
        "status": "refund_initiated",
        "transaction_ref": payload.transaction_ref,
        "backend_response": result,
    }


@router.get("/disputes")
async def get_all_disputes():
    result = await _get("/dispute")
    return result


@router.post("/disputes/{ticket_id}/resolve")
async def resolve_dispute(ticket_id: str, payload: ResolveDisputeRequest):
    if payload.action not in ("accepted", "rejected"):
        raise HTTPException(status_code=400, detail="action must be 'accepted' or 'rejected'.")

    result = await _post(
        f"/dispute/{ticket_id}/resolve",
        {"action": payload.action, "file_name": payload.file_name},
    )
    return {"status": "dispute_resolved", "ticket_id": ticket_id, "backend_response": result}


@router.get("/disputes/upload-url/{ticket_id}/{file_name}")
async def get_dispute_upload_url(ticket_id: str, file_name: str):
    result = await _get(f"/dispute/upload-url/{ticket_id}/{file_name}")
    return result
