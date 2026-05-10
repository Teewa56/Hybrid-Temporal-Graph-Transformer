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


async def fetch_customer_transactions(email: str, limit: int = 50) -> list[dict]:
    """
    Fetch the last N transactions for a customer by email.
    Used by SequentialService to build the behavioral sequence vector.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{SQUAD_BASE_URL}/transaction/filter",
            headers=HEADERS,
            params={"email": email, "limit": limit, "sort": "desc"},
        )
    if response.status_code != 200:
        return []
    return response.json().get("data", {}).get("transactions", [])


@router.get("/transaction/{transaction_ref}")
async def get_transaction(transaction_ref: str):
    data = await fetch_transaction(transaction_ref)
    return {"transaction_ref": transaction_ref, "data": data}


@router.get("/transactions/customer/{email}")
async def get_customer_transactions(email: str, limit: int = 50):
    transactions = await fetch_customer_transactions(email, limit)
    return {"email": email, "count": len(transactions), "transactions": transactions}