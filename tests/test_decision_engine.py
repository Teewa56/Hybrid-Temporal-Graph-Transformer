import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.ensemble import EnsembleScores, DEFAULT_WEIGHTS
from app.services.decision_engine import DecisionEngine, FraudZone


def make_scores(unified: float, **kwargs) -> EnsembleScores:
    defaults = dict(
        transformer_score=unified,
        graphsage_score=unified,
        cnn_gnn_score=unified,
        tssgc_score=unified,
        gan_autoencoder_score=unified,
        unified_score=unified,
        weights=DEFAULT_WEIGHTS,
    )
    defaults.update(kwargs)
    return EnsembleScores(**defaults)


@pytest.fixture
def engine():
    return DecisionEngine()


@pytest.fixture
def sample_body():
    return {
        "customer_email": "test@gmail.com",
        "amount": 500000,
        "channel": "app",
        "ip_address": "102.1.1.1",
        "device_id": "device-abc",
        "currency": "NGN",
    }


class TestDecisionEngineZones:

    def test_green_zone(self, engine):
        assert engine._determine_zone(0.30) == FraudZone.GREEN
        assert engine._determine_zone(0.64) == FraudZone.GREEN

    def test_amber_zone(self, engine):
        assert engine._determine_zone(0.65) == FraudZone.AMBER
        assert engine._determine_zone(0.80) == FraudZone.AMBER
        assert engine._determine_zone(0.89) == FraudZone.AMBER

    def test_red_zone(self, engine):
        assert engine._determine_zone(0.90) == FraudZone.RED
        assert engine._determine_zone(0.99) == FraudZone.RED
        assert engine._determine_zone(1.00) == FraudZone.RED

    def test_boundary_green_amber(self, engine):
        assert engine._determine_zone(0.649) == FraudZone.GREEN
        assert engine._determine_zone(0.650) == FraudZone.AMBER

    def test_boundary_amber_red(self, engine):
        assert engine._determine_zone(0.899) == FraudZone.AMBER
        assert engine._determine_zone(0.900) == FraudZone.RED


class TestTopSignals:

    def test_returns_high_scoring_models(self, engine):
        scores = make_scores(
            0.5,
            transformer_score=0.95,
            graphsage_score=0.88,
            cnn_gnn_score=0.30,
            tssgc_score=0.20,
            gan_autoencoder_score=0.10,
        )
        signals = engine._identify_top_signals(scores)
        assert any("Transformer" in s or "Behavioral" in s for s in signals)

    def test_empty_when_all_low(self, engine):
        scores = make_scores(0.1)
        signals = engine._identify_top_signals(scores)
        assert len(signals) == 0

    def test_sorted_by_score_descending(self, engine):
        scores = make_scores(
            0.8,
            transformer_score=0.75,
            graphsage_score=0.95,
            cnn_gnn_score=0.72,
            tssgc_score=0.80,
            gan_autoencoder_score=0.10,
        )
        signals = engine._identify_top_signals(scores)
        assert len(signals) >= 1


class TestDecisionEngineDecide:

    @pytest.mark.asyncio
    async def test_green_zone_no_api_call(self, engine, sample_body):
        scores = make_scores(0.30)
        with patch.object(engine, "_initiate_refund", new_callable=AsyncMock) as mock_refund:
            with patch.object(engine, "_trigger_step_up_auth", new_callable=AsyncMock) as mock_auth:
                result = await engine.decide("REF001", scores, sample_body, request=None)
                mock_refund.assert_not_called()
                mock_auth.assert_not_called()
        assert result.zone == FraudZone.GREEN

    @pytest.mark.asyncio
    async def test_amber_zone_triggers_step_up(self, engine, sample_body):
        scores = make_scores(0.75)
        with patch.object(engine, "_trigger_step_up_auth", new_callable=AsyncMock) as mock_auth:
            with patch.object(engine, "_initiate_refund", new_callable=AsyncMock) as mock_refund:
                result = await engine.decide("REF002", scores, sample_body, request=None)
                mock_auth.assert_called_once()
                mock_refund.assert_not_called()
        assert result.zone == FraudZone.AMBER

    @pytest.mark.asyncio
    async def test_red_zone_calls_dispute_api(self, engine, sample_body):
        scores = make_scores(0.95)
        with patch.object(engine, "_initiate_refund", new_callable=AsyncMock) as mock_refund:
            with patch.object(engine, "_trigger_step_up_auth", new_callable=AsyncMock):
                result = await engine.decide("REF003", scores, sample_body, request=None)
                mock_refund.assert_called_once()
        assert result.zone == FraudZone.RED

    @pytest.mark.asyncio
    async def test_result_has_correct_fields(self, engine, sample_body):
        scores = make_scores(0.30)
        with patch.object(engine, "_initiate_refund", new_callable=AsyncMock):
            with patch.object(engine, "_trigger_step_up_auth", new_callable=AsyncMock):
                result = await engine.decide("REF004", scores, sample_body, request=None)
        assert result.transaction_ref == "REF004"
        assert result.unified_score == 0.30
        assert isinstance(result.scores_breakdown, dict)
        assert isinstance(result.top_signals, list)