import asyncio
import numpy as np
from typing import Any

from app.models.transformer import BehavioralTransformer
from app.models.graphsage import GraphSAGEFraudDetector
from app.models.cnn_gnn import CNNGNNHybrid
from app.models.tssgc import SIMSwapDetector
from app.models.gan_autoencoder import GANAutoencoderKYC
from app.models.ensemble import ModelEnsemble, EnsembleScores


class ModelServer:
    """
    Loads and serves all five fraud detection models.
    All models run in parallel via asyncio for sub-200ms inference.
    In production, models are quantized with ONNX/TensorRT.
    """

    def __init__(self):
        self.transformer: BehavioralTransformer = None
        self.graphsage: GraphSAGEFraudDetector = None
        self.cnn_gnn: CNNGNNHybrid = None
        self.tssgc: SIMSwapDetector = None
        self.gan_autoencoder: GANAutoencoderKYC = None
        self.ensemble = ModelEnsemble()
        self._loaded = False

    async def load_all(self):
        """Initialize all models. Load from checkpoint if available."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_models_sync)
        self._loaded = True
        print(" All 5 models loaded.")

    def _load_models_sync(self):
        self.transformer = BehavioralTransformer()
        self.graphsage = GraphSAGEFraudDetector()
        self.cnn_gnn = CNNGNNHybrid()
        self.tssgc = SIMSwapDetector()
        self.gan_autoencoder = GANAutoencoderKYC()

        # TODO: Load pre-trained weights from checkpoint paths in config
        # self.transformer.load_state_dict(torch.load("checkpoints/transformer.pt"))

    async def run_ensemble(
        self,
        body: dict,
        sequence_vector: np.ndarray,
        graph_snapshot: dict,
    ) -> EnsembleScores:
        """
        Execute all 5 models concurrently and return EnsembleScores.
        Uses asyncio.gather to run inference in parallel.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_all() first.")

        self.ensemble.adjust_weights_for_context(body)

        loop = asyncio.get_event_loop()

        # Extract feature inputs per model
        node_features = graph_snapshot.get("node_features", np.zeros((10, 64)))
        edge_index = graph_snapshot.get("edge_index", np.zeros((2, 10), dtype=int))
        target_node_idx = graph_snapshot.get("target_node_idx", 0)
        payload_features = graph_snapshot.get("payload_features", np.zeros(64))
        graph_embedding = graph_snapshot.get("graph_embedding", np.zeros(64))
        device_sequence = graph_snapshot.get("device_sequence", np.zeros((10, 32)))
        account_history = graph_snapshot.get("account_history", np.zeros(32))
        kyc_features = graph_snapshot.get("kyc_features", np.zeros(128))

        # Run all models concurrently in executor (they're CPU/GPU bound)
        results = await asyncio.gather(
            loop.run_in_executor(None, self.transformer.predict, sequence_vector),
            loop.run_in_executor(None, self.graphsage.predict_node, node_features, edge_index, target_node_idx),
            loop.run_in_executor(None, self.cnn_gnn.predict, payload_features, graph_embedding),
            loop.run_in_executor(None, self.tssgc.predict, device_sequence, account_history),
            loop.run_in_executor(None, self.gan_autoencoder.predict, kyc_features),
        )

        transformer_score, graphsage_score, cnn_gnn_score, tssgc_score, gan_score = results

        return self.ensemble.compute_unified_score(
            transformer_score=transformer_score,
            graphsage_score=graphsage_score,
            cnn_gnn_score=cnn_gnn_score,
            tssgc_score=tssgc_score,
            gan_autoencoder_score=gan_score,
        )