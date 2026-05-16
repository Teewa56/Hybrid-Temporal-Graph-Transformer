import os
import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()

NEOBANK_BASE_URL = os.getenv("NEOBANK_BASE_URL", "https://api.neobank.example")
NEOBANK_SECRET_KEY = os.getenv("NEOBANK_SECRET_KEY", "")

HEADERS = {
    "Authorization": f"Bearer {NEOBANK_SECRET_KEY}",
    "Content-Type": "application/json",
}


async def fetch_transaction(transaction_ref: str) -> dict:
    """
    Fetch full transaction details from the integrated payment backend by reference.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{NEOBANK_BASE_URL}/transaction/verify/{transaction_ref}",
            headers=HEADERS,
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Backend API error: {response.text}",
        )
    return response.json().get("data", {})


async def fetch_customer_transactions(customer_identifier: str, limit: int = 50) -> list[dict]:
    """
    Fetch transactions for a customer from the integrated payment backend.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{NEOBANK_BASE_URL}/virtual-account/customer/transactions/{customer_identifier}",
            headers=HEADERS,
        )
    if response.status_code != 200:
        return []

    transactions = response.json().get("data", [])
    if not isinstance(transactions, list):
        return []

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
