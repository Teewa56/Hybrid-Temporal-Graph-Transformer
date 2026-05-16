import os
import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()

SQUAD_BASE_URL = os.getenv("SQUAD_BASE_URL", "https://sandbox-api-d.squadco.com")
SQUAD_SECRET_KEY = os.getenv("SQUAD_SECRET_KEY", "")

HEADERS = {
    "Authorization": f"Bearer {SQUAD_SECRET_KEY}",
    "Content-Type": "application/json",
}


async def fetch_transaction(transaction_ref: str) -> dict:
    """
    Fetch full transaction details from Squad by reference.
    Used by feature engineering services to enrich webhook payloads.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{SQUAD_BASE_URL}/transaction/verify/{transaction_ref}",
            headers=HEADERS,
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Squad API error: {response.text}",
        )
    return response.json().get("data", {})


async def fetch_customer_transactions(customer_identifier: str, limit: int = 50) -> list[dict]:
    """
    Fetch transactions for a customer by their customer_identifier.
    Uses the virtual account customer transactions endpoint.

    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{SQUAD_BASE_URL}/virtual-account/customer/transactions/{customer_identifier}",
            headers=HEADERS,
        )
    if response.status_code != 200:
        return []

    # 'data' is a direct list — not a nested dict with a 'transactions' key
    transactions = response.json().get("data", [])
    if not isinstance(transactions, list):
        return []

    # Most recent first; Squad returns chronological by default
    return transactions[-limit:]


@router.get("/transaction/{transaction_ref}")
async def get_transaction(transaction_ref: str):
    data = await fetch_transaction(transaction_ref)
    return {"transaction_ref": transaction_ref, "data": data}


@router.get("/transactions/customer/{customer_identifier}")
async def get_customer_transactions(customer_identifier: str, limit: int = 50):
    transactions = await fetch_customer_transactions(customer_identifier, limit=limit)
    return {
        "customer_identifier": customer_identifier,
        "count": len(transactions),
        "transactions": transactions,
    }