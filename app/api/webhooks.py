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

WEBHOOK_SECRET = os.getenv("NEOBANK_WEBHOOK_SECRET", "")

HANDLED_EVENTS = {"charge_successful", "transaction.success"}


class WebhookPayload(BaseModel):
    Event: str
    TransactionRef: str
    Body: dict[str, Any] = Field(default_factory=dict)


def _verify_card_signature(raw_body: bytes, signature: str) -> bool:
    """
    Card/payment webhook: HMAC-SHA512 of the entire raw body.
    The upstream payment backend sends the hash UPPERCASED in x-webhook-signature.
    """
    expected = hmac.new(
        WEBHOOK_SECRET.encode(), raw_body, hashlib.sha512
    ).hexdigest().upper()
    return hmac.compare_digest(expected, (signature or "").upper())


def _verify_va_signature(payload: dict, signature: str) -> bool:
    """
    Virtual account webhook: HMAC-SHA512 of six pipe-separated fields.
    The upstream payment backend sends the hash UPPERCASED in x-webhook-signature.
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
        WEBHOOK_SECRET.encode(), sig_string.encode(), hashlib.sha512
    ).hexdigest().upper()
    return hmac.compare_digest(expected, (signature or "").upper())


def _normalise_card_payload(payload: dict) -> dict:
    body = payload.get("Body", {})
    return {
        "TransactionRef": payload.get("TransactionRef", ""),
        "amount": body.get("amount", 0),
        "customer_email": body.get("email", ""),
        "currency": body.get("currency", "NGN"),
        "transaction_type": body.get("transaction_type", ""),
        "transaction_status": body.get("transaction_status", ""),
        "merchant_id": body.get("merchant_id", ""),
        "created_at": body.get("created_at", ""),
        "gateway_ref": body.get("gateway_ref", ""),
        "channel": body.get("transaction_type", "").lower(),
        "ip_address": body.get("ip_address", ""),
        "device_id": body.get("device_id", ""),
        "merchant_category": body.get("merchant_category", "unknown"),
        "is_new_device": body.get("is_new_device", False),
        "is_new_recipient": body.get("is_new_recipient", False),
        "meta": body.get("meta", {}),
    }


def _normalise_va_payload(payload: dict) -> dict:
    principal_naira = float(payload.get("principal_amount", 0) or 0)
    return {
        "TransactionRef": payload.get("transaction_reference", ""),
        "amount": int(principal_naira * 100),
        "customer_email": payload.get("customer_email", ""),
        "customer_identifier": payload.get("customer_identifier", ""),
        "currency": payload.get("currency", "NGN"),
        "transaction_type": "VirtualAccount",
        "transaction_status": "Success",
        "created_at": payload.get("transaction_date", ""),
        "gateway_ref": payload.get("transaction_reference", ""),
        "channel": "virtual-account",
        "sender_name": payload.get("sender_name", ""),
        "virtual_account_number": payload.get("virtual_account_number", ""),
        "ip_address": payload.get("ip_address", ""),
        "device_id": payload.get("device_id", ""),
        "merchant_category": payload.get("merchant_category", "unknown"),
        "is_new_device": payload.get("is_new_device", False),
        "is_new_recipient": payload.get("is_new_recipient", False),
        "meta": payload.get("meta", {}),
    }


async def _run_fraud_pipeline(normalised_body: dict, transaction_ref: str, request: Request):
    cache: CacheService = request.app.state.cache
    model_server = request.app.state.model_server
    drift_detector = request.app.state.drift_detector

    await cache.set(f"txn:{transaction_ref}", json.dumps(normalised_body), ttl=300)

    seq_service = SequentialService(cache)
    sequence_vector = await seq_service.build(normalised_body)

    graph_service = GraphService()
    graph_snapshot = await graph_service.update_and_fetch(normalised_body)

    scores = await model_server.run_ensemble(
        body=normalised_body,
        sequence_vector=sequence_vector,
        graph_snapshot=graph_snapshot,
    )

    engine = DecisionEngine()
    decision = await engine.decide(
        transaction_ref=transaction_ref,
        scores=scores,
        body=normalised_body,
        request=request,
    )

    drift_detector.observe(normalised_body, scores)

    audit = AuditTrail()
    await audit.log(
        transaction_ref=transaction_ref,
        scores=scores,
        decision=decision,
        body=normalised_body,
    )

    return decision


@router.post("/")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    raw_body = await request.body()

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    if "Event" in payload:
        if WEBHOOK_SECRET:
            sig = request.headers.get("x-webhook-signature", "")
            if not _verify_card_signature(raw_body, sig):
                raise HTTPException(status_code=401, detail="Invalid webhook signature.")

        event = payload.get("Event", "")
        if event not in HANDLED_EVENTS:
            return {"status": "ignored", "event": event}

        transaction_ref = payload.get("TransactionRef", "")
        normalised_body = _normalise_card_payload(payload)

    elif "transaction_reference" in payload:
        if WEBHOOK_SECRET:
            sig = request.headers.get("x-webhook-signature", "")
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
