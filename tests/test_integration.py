import pytest
import json
import hmac
import hashlib
from unittest.mock import AsyncMock, patch, MagicMock
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

WEBHOOK_TEST_SECRET = os.getenv("WEBHOOK_SECRET", "testsecret")

SAMPLE_WEBHOOK_PAYLOAD = {
    "Event": "transaction.success",
    "TransactionRef": "TESTREF000001",
    "Body": {
        "amount": 5000000,
        "currency": "NGN",
        "customer_email": "amaka@gmail.com",
        "customer_name": "Amaka Okonkwo",
        "ip_address": "102.1.2.3",
        "device_id": "device-xyz-123",
        "channel": "app",
        "merchant_category": "transfer",
        "transaction_type": "debit",
        "created_at": "2025-05-01T14:30:00Z",
        "is_new_device": False,
        "is_new_recipient": True,
    }
}


def _sign_payload(payload: dict, secret: str) -> str:
    raw = json.dumps(payload).encode()
    return hmac.new(secret.encode(), raw, hashlib.sha512).hexdigest()


def _mock_app_state():
    """
    Build mock objects for app.state and inject them directly.
    Called inside each test that hits the background pipeline.
    """
    mock_cache = AsyncMock()
    mock_cache.set        = AsyncMock()
    mock_cache.get        = AsyncMock(return_value=None)
    mock_cache.get_list   = AsyncMock(return_value=[])
    mock_cache.push_to_list = AsyncMock()

    mock_scores = MagicMock()
    mock_scores.unified_score       = 0.30
    mock_scores.transformer_score   = 0.25
    mock_scores.graphsage_score     = 0.30
    mock_scores.cnn_gnn_score       = 0.20
    mock_scores.tssgc_score         = 0.28
    mock_scores.gan_autoencoder_score = 0.15
    mock_scores.to_dict = lambda: {"unified": 0.30}

    mock_model_server = AsyncMock()
    mock_model_server.run_ensemble = AsyncMock(return_value=mock_scores)

    mock_drift = MagicMock()
    mock_drift.observe = MagicMock()

    app.state.cache          = mock_cache
    app.state.model_server   = mock_model_server
    app.state.drift_detector = mock_drift

    return mock_cache, mock_model_server, mock_drift


# ── Webhook Endpoint Tests ────────────────────────────────────────────────────

class TestWebhookEndpoint:

    def test_health_check(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ignores_unknown_events(self):
        _mock_app_state()
        with patch("app.api.webhooks.WEBHOOK_SECRET", ""):
            client = TestClient(app)
            payload = {**SAMPLE_WEBHOOK_PAYLOAD, "Event": "refund.completed"}
            response = client.post(
                "/webhook",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_webhook_returns_200_fast(self):
        """Webhook must return 200 immediately — pipeline runs in background."""
        _mock_app_state()

        with patch("app.api.webhooks.WEBHOOK_SECRET", ""), \
             patch("app.api.webhooks._run_fraud_pipeline", new_callable=AsyncMock):
            client = TestClient(app)
            response = client.post(
                "/webhook",
                json=SAMPLE_WEBHOOK_PAYLOAD,
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 200
        assert "transaction_ref" in response.json()

    def test_rejects_invalid_signature(self):
        _mock_app_state()
        with patch("app.api.webhooks.WEBHOOK_SECRET", WEBHOOK_TEST_SECRET):
            client = TestClient(app)
            response = client.post(
                "/webhook",
                json=SAMPLE_WEBHOOK_PAYLOAD,
                headers={
                    "Content-Type": "application/json",
                    "x-webhook-signature": "invalid_signature",
                },
            )
        assert response.status_code == 401

    def test_accepts_valid_signature(self):
        _mock_app_state()
        payload_bytes = json.dumps(SAMPLE_WEBHOOK_PAYLOAD).encode()
        signature = hmac.new(
            WEBHOOK_TEST_SECRET.encode(), payload_bytes, hashlib.sha512
        ).hexdigest()

        with patch("app.api.webhooks.WEBHOOK_SECRET", WEBHOOK_TEST_SECRET), \
             patch("app.api.webhooks._run_fraud_pipeline", new_callable=AsyncMock):
            client = TestClient(app)
            response = client.post(
                "/webhook",
                content=payload_bytes,
                headers={
                    "Content-Type": "application/json",
                    "x-webhook-signature": signature,
                },
            )
        assert response.status_code == 200

    def test_rejects_malformed_json(self):
        _mock_app_state()
        with patch("app.api.webhooks.WEBHOOK_SECRET", ""):
            client = TestClient(app)
            response = client.post(
                "/webhook",
                content=b"not valid json {{{",
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400


# ── Dispute Endpoint Tests ────────────────────────────────────────────────────

class TestDisputeEndpoint:

    @pytest.mark.asyncio
    async def test_raise_dispute_calls_backend(self):
        _mock_app_state()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "message": "Dispute raised"}

        with patch("app.api.disputes.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            # Use ASGITransport — correct way to test ASGI apps with httpx
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as ac:
                response = await ac.post(
                    "/dispute/refund",
                    json={
                        "transaction_ref": "TESTREF000001",
                        "reason": "Fraud detected",
                    },
                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "refund_initiated"
        assert data["transaction_ref"] == "TESTREF000001"

    @pytest.mark.asyncio
    async def test_refund_transaction(self):
        _mock_app_state()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}

        with patch("app.api.disputes.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as ac:
                response = await ac.post(
                    "/dispute/refund",
                    json={"transaction_ref": "TESTREF000001"},
                )

        assert response.status_code == 200
        assert response.json()["status"] == "refund_initiated"


# ── Synthetic Data Pipeline Integration ───────────────────────────────────────

class TestSyntheticDataPipeline:

    def test_behavioral_pipeline_runs(self):
        from synthetic_data_generator.behavioral import (
            UserProfileGenerator, TransactionSequenceGenerator, AnomalyInjector
        )
        profiles = UserProfileGenerator(seed=42).generate_batch(
            n_users=10, n_fraud_users=2
        )
        assert len(profiles) == 10

        seq_gen   = TransactionSequenceGenerator(seed=42)
        sequences = [seq_gen.generate_for_user(p, n_transactions=20) for p in profiles]
        assert all(len(s) == 20 for s in sequences)

        injector = AnomalyInjector(seed=42)
        pairs    = list(zip(sequences, profiles))
        injected = injector.inject_batch(pairs, fraud_rate=0.5)
        assert len(injected) == 10

    def test_payload_pipeline_runs(self):
        from synthetic_data_generator.payload import (
            LegitimatePayloadGenerator, PayloadAnomalyInjector, PaymentPayloadSchema
        )
        schema   = PaymentPayloadSchema()
        gen      = LegitimatePayloadGenerator(seed=42)
        payloads = gen.generate_batch(n=20)
        assert len(payloads) == 20

        for p in payloads:
            is_valid, errors = schema.validate(p)
            assert is_valid, f"Invalid payload: {errors}"

        injector  = PayloadAnomalyInjector(seed=42)
        anomalous = injector.inject_batch(payloads, n_anomalous=10)
        assert len(anomalous) == 10
        assert all(p["label"] == 1 for p in anomalous)

    def test_kyc_pipeline_runs(self):
        from synthetic_data_generator.kyc import DocumentMetadataGenerator, ForgerySimulator
        gen    = DocumentMetadataGenerator(seed=42)
        docs   = gen.generate_batch(n=20)
        assert len(docs) == 20
        assert all(d.label == 0 for d in docs)

        sim    = ForgerySimulator(seed=42)
        forged = sim.generate_forged_batch(docs, n=10)
        assert len(forged) == 10
        assert all(d.label == 1 for d in forged)

    def test_sim_swap_pipeline_runs(self):
        from synthetic_data_generator.sim_swap import (
            DeviceProfileGenerator, HandoverEventSimulator
        )
        gen        = DeviceProfileGenerator(seed=42)
        histories  = gen.generate_batch(n_users=20)
        assert len(histories) == 20

        sim        = HandoverEventSimulator(seed=42)
        with_swaps = sim.simulate_batch(histories)
        assert len(with_swaps) == 20