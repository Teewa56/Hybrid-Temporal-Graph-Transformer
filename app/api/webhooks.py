import os
import json
import hmac
import hashlib
from typing import Any

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.cache_service import CacheService
from app.services.sequential_service import SequentialService
from app.services.graph_service import GraphService
from app.services.decision_engine import DecisionEngine
from app.explainability.audit_trail import AuditTrail

router = APIRouter()

SQUAD_SECRET = os.getenv("SQUAD_WEBHOOK_SECRET", "")

# ── Actual Squad event name ──────────────────────────────────────────────────
HANDLED_EVENTS = {"charge_successful"}


class SquadWebhookPayload(BaseModel):
    Event: str
    TransactionRef: str
    Body: dict[str, Any] = Field(default_factory=dict)


# ── Signature verification ───────────────────────────────────────────────────

def _verify_card_signature(raw_body: bytes, signature: str) -> bool:
    """
    Card/payment webhook: HMAC-SHA512 of the entire raw body.
    Squad sends the hash UPPERCASED in x-squad-encrypted-body.
    """
    expected = hmac.new(
        SQUAD_SECRET.encode(), raw_body, hashlib.sha512
    ).hexdigest().upper()                          # ← must be uppercase
    return hmac.compare_digest(expected, (signature or "").upper())


def _verify_va_signature(payload: dict, signature: str) -> bool:
    """
    Virtual account webhook: HMAC-SHA512 of six pipe-separated fields.
    Squad sends the hash in x-squad-signature (also uppercase).
    """
    sig_string = "|".join([
        payload.get("transaction_reference", ""),
        payload.get("virtual_account_number", ""),
        payload.get("currency", ""),
        payload.get("principal_amount", ""),
        payload.get("settled_amount", ""),
        payload.get("customer_identifier", ""),
    ])
    expected = hmac.new(
        SQUAD_SECRET.encode(), sig_string.encode(), hashlib.sha512
    ).hexdigest().upper()
    return hmac.compare_digest(expected, (signature or "").upper())


# ── Payload normalisation ─────────────────────────────────────────────────────

def _normalise_card_payload(payload: dict) -> dict:
    """
    Card/bank/USSD/transfer webhooks already have the {Event, TransactionRef, Body}
    envelope, but field names differ from the old assumed schema.
    Normalises to a consistent internal format expected by the pipeline.
    """
    body = payload.get("Body", {})
    return {
        "TransactionRef": payload.get("TransactionRef", ""),
        "amount": body.get("amount", 0),
        # Squad sends 'email', not 'customer_email'
        "customer_email": body.get("email", ""),
        "currency": body.get("currency", "NGN"),
        # Squad transaction_type values: "Card", "Transfer", "Bank", "Ussd", "MerchantUssd"
        "transaction_type": body.get("transaction_type", ""),
        "transaction_status": body.get("transaction_status", ""),
        "merchant_id": body.get("merchant_id", ""),
        "created_at": body.get("created_at", ""),
        # gateway_ref is needed for the Refund API in Red Zone
        "gateway_ref": body.get("gateway_ref", ""),
        "channel": body.get("transaction_type", "").lower(),
        # Fields not sent by Squad — default to safe values
        "ip_address": "",
        "device_id": "",
        "merchant_category": "unknown",
        "is_new_device": False,
        "is_new_recipient": False,
        "meta": body.get("meta", {}),
    }


def _normalise_va_payload(payload: dict) -> dict:
    """
    Virtual account webhooks have a completely different flat structure.
    Normalises to the same internal format as card payloads.
    principal_amount is a naira string — convert to kobo int to match card format.
    """
    principal_naira = float(payload.get("principal_amount", 0) or 0)
    return {
        "TransactionRef": payload.get("transaction_reference", ""),
        "amount": int(principal_naira * 100),       # kobo
        "customer_email": "",                        # not in VA webhook
        "customer_identifier": payload.get("customer_identifier", ""),
        "currency": payload.get("currency", "NGN"),
        "transaction_type": "VirtualAccount",
        "transaction_status": "Success",
        "created_at": payload.get("transaction_date", ""),
        "gateway_ref": payload.get("transaction_reference", ""),
        "channel": "virtual-account",
        "sender_name": payload.get("sender_name", ""),
        "virtual_account_number": payload.get("virtual_account_number", ""),
        # Not present in VA webhook
        "ip_address": "",
        "device_id": "",
        "merchant_category": "unknown",
        "is_new_device": False,
        "is_new_recipient": False,
        "meta": payload.get("meta", {}),
    }


# ── Pipeline ─────────────────────────────────────────────────────────────────

async def _run_fraud_pipeline(normalised_body: dict, transaction_ref: str, request: Request):
    """
    Core async pipeline. Accepts a normalised body dict regardless of
    whether the original webhook was a card payment or virtual account credit.
    """
    cache: CacheService = request.app.state.cache
    model_server = request.app.state.model_server
    drift_detector = request.app.state.drift_detector

    # 1. Cache normalised payload
    await cache.set(f"txn:{transaction_ref}", json.dumps(normalised_body), ttl=300)

    # 2. Build sequential feature vector (last 50 transactions)
    seq_service = SequentialService(cache)
    sequence_vector = await seq_service.build(normalised_body)

    # 3. Build/update graph
    graph_service = GraphService()
    graph_snapshot = await graph_service.update_and_fetch(normalised_body)

    # 4. Run all 5 models in parallel
    scores = await model_server.run_ensemble(
        body=normalised_body,
        sequence_vector=sequence_vector,
        graph_snapshot=graph_snapshot,
    )

    # 5. Decision engine
    engine = DecisionEngine()
    decision = await engine.decide(
        transaction_ref=transaction_ref,
        scores=scores,
        body=normalised_body,
        request=request,
    )

    # 6. Feed to drift detector
    drift_detector.observe(normalised_body, scores)

    # 7. Audit trail
    audit = AuditTrail()
    await audit.log(
        transaction_ref=transaction_ref,
        scores=scores,
        decision=decision,
        body=normalised_body,
    )

    return decision


# ── Webhook endpoint ─────────────────────────────────────────────────────────

@router.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    raw_body = await request.body()

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    # ── Detect webhook type ──────────────────────────────────────────────────
    # Card/transfer webhooks have a top-level "Event" key.
    # Virtual account webhooks have "transaction_reference" at the top level.

    if "Event" in payload:
        # Card / bank / transfer / USSD payment webhook
        if SQUAD_SECRET:
            sig = request.headers.get("x-squad-encrypted-body", "")
            if not _verify_card_signature(raw_body, sig):
                raise HTTPException(status_code=401, detail="Invalid webhook signature.")

        event = payload.get("Event", "")
        if event not in HANDLED_EVENTS:
            return {"status": "ignored", "event": event}

        transaction_ref = payload.get("TransactionRef", "")
        normalised_body = _normalise_card_payload(payload)

    elif "transaction_reference" in payload:
        # Virtual account webhook
        if SQUAD_SECRET:
            sig = request.headers.get("x-squad-signature", "")
            if not _verify_va_signature(payload, sig):
                raise HTTPException(status_code=401, detail="Invalid VA webhook signature.")

        transaction_ref = payload.get("transaction_reference", "")
        normalised_body = _normalise_va_payload(payload)

    else:
        return {"status": "ignored", "reason": "unrecognised payload structure"}

    if not transaction_ref:
        raise HTTPException(status_code=400, detail="Missing transaction reference.")

    background_tasks.add_task(_run_fraud_pipeline, normalised_body, transaction_ref, request)
    return {"status": "received", "transaction_ref": transaction_ref}