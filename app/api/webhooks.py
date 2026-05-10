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


class SquadWebhookPayload(BaseModel):
    Event: str
    TransactionRef: str
    Body: dict[str, Any] = Field(default_factory=dict)


def _verify_signature(raw_body: bytes, signature: str) -> bool:
    expected = hmac.new(
        SQUAD_SECRET.encode(), raw_body, hashlib.sha512
    ).hexdigest()
    return hmac.compare_digest(expected, signature or "")


async def _run_fraud_pipeline(payload: dict, request: Request):
    """
    Core async pipeline triggered on every Squad webhook event.
    Runs feature extraction → model ensemble → decision engine.
    """
    cache: CacheService = request.app.state.cache
    model_server = request.app.state.model_server
    drift_detector = request.app.state.drift_detector

    transaction_ref = payload.get("TransactionRef")
    body = payload.get("Body", {})

    # 1. Cache raw payload for sub-ms retrieval during inference
    await cache.set(f"txn:{transaction_ref}", json.dumps(body), ttl=300)

    # 2. Build sequential feature vector (last 50 transactions)
    seq_service = SequentialService(cache)
    sequence_vector = await seq_service.build(body)

    # 3. Build/update graph with this transaction
    graph_service = GraphService()
    graph_snapshot = await graph_service.update_and_fetch(body)

    # 4. Run all 5 models in parallel
    scores = await model_server.run_ensemble(
        body=body,
        sequence_vector=sequence_vector,
        graph_snapshot=graph_snapshot,
    )

    # 5. Decision engine routes based on unified fraud score
    engine = DecisionEngine()
    decision = await engine.decide(
        transaction_ref=transaction_ref,
        scores=scores,
        body=body,
        request=request,
    )

    # 6. Feed features to drift detector for continual learning
    drift_detector.observe(body, scores)

    # 7. Log to audit trail
    audit = AuditTrail()
    await audit.log(
        transaction_ref=transaction_ref,
        scores=scores,
        decision=decision,
        body=body,
    )

    return decision


@router.post("/webhook")
async def receive_webhook(
    request: Request, background_tasks: BackgroundTasks
):
    raw_body = await request.body()
    signature = request.headers.get("x-squad-encrypted-body", "")

    if SQUAD_SECRET and not _verify_signature(raw_body, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature.")

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    event = payload.get("Event", "")
    if event not in ("transaction.success", "transfer.initiated", "payment_link.paid"):
        # Acknowledge but don't process irrelevant events
        return {"status": "ignored", "event": event}

    # Run pipeline in background so Squad gets a fast 200 ACK
    background_tasks.add_task(_run_fraud_pipeline, payload, request)

    return {"status": "received", "transaction_ref": payload.get("TransactionRef")}