import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

SQUAD_BASE_URL = os.getenv("SQUAD_BASE_URL", "https://sandbox-api-d.squadco.com")
SQUAD_SECRET_KEY = os.getenv("SQUAD_SECRET_KEY", "")

HEADERS = {
    "Authorization": f"Bearer {SQUAD_SECRET_KEY}",
    "Content-Type": "application/json",
}


class DisputeRequest(BaseModel):
    transaction_ref: str
    reason: str = "Fraud detected by Hybrid_Temporal_Graph_Transformer AI"


class ReverseRequest(BaseModel):
    transaction_ref: str


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


@router.post("/dispute")
async def raise_dispute(payload: DisputeRequest):
    """
    Raise a dispute on a transaction flagged as Red Zone fraud.
    Freezes funds in the virtual account before merchant settlement.
    """
    result = await _post(
        "/dispute/transaction/raise-dispute",
        {
            "transaction_ref": payload.transaction_ref,
            "reason": payload.reason,
        },
    )
    return {
        "status": "dispute_raised",
        "transaction_ref": payload.transaction_ref,
        "squad_response": result,
    }


@router.post("/reverse")
async def reverse_transaction(payload: ReverseRequest):
    """
    Reverse a transaction that has already settled but was flagged post-hoc.
    """
    result = await _post(
        "/transaction/reverse",
        {"transaction_ref": payload.transaction_ref},
    )
    return {
        "status": "reversed",
        "transaction_ref": payload.transaction_ref,
        "squad_response": result,
    }


@router.get("/dispute/status/{transaction_ref}")
async def get_dispute_status(transaction_ref: str):
    """
    Fetch the current status of a raised dispute.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{SQUAD_BASE_URL}/dispute/transaction/{transaction_ref}",
            headers=HEADERS,
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Squad API error: {response.text}",
        )
    return response.json()