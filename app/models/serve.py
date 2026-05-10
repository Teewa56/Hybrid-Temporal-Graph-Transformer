import asyncio
import numpy as np
import torch
from pathlib import Path
from typing import Any

from app.models.transformer import BehavioralTransformer, TransformerConfig
from app.models.graphsage import GraphSAGEFraudDetector
from app.models.cnn_gnn import CNNGNNHybrid
from app.models.tssgc import SIMSwapDetector
from app.models.gan_autoencoder import GANAutoencoderKYC
from app.models.ensemble import ModelEnsemble, EnsembleScores

import os

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))

# Maps model name → (class, constructor kwargs, checkpoint filename)
MODEL_REGISTRY = {
    "transformer":     (BehavioralTransformer,    {"config": TransformerConfig()},  "transformer.pt"),
    "graphsage":       (GraphSAGEFraudDetector,   {},                               "graphsage.pt"),
    "cnn_gnn":         (CNNGNNHybrid,             {},                               "cnn_gnn.pt"),
    "tssgc":           (SIMSwapDetector,           {},                               "tssgc.pt"),
    "gan_autoencoder": (GANAutoencoderKYC,         {},                               "gan_autoencoder.pt"),
}


def _load_model(name: str):
    """
    Instantiate a model and load its checkpoint if one exists.
    Falls back to random weights with a clear warning if no checkpoint is found.
    """
    model_cls, kwargs, ckpt_filename = MODEL_REGISTRY[name]
    model = model_cls(**kwargs)
    model.eval()

    ckpt_path = CHECKPOINT_DIR / ckpt_filename
    if ckpt_path.exists():
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"  ✅ {name:<20} loaded from {ckpt_path}")
        except Exception as e:
            print(f"  ⚠️  {name:<20} checkpoint found but failed to load: {e}")
            print(f"       Running with random weights.")
    else:
        print(f"  ⚠️  {name:<20} no checkpoint at {ckpt_path} — random weights.")

    return model


class ModelServer:
    """
    Loads and serves all five fraud detection models.
    All models run in parallel via asyncio for sub-200ms inference.
    Checkpoints are loaded from CHECKPOINT_DIR at startup.
    Falls back gracefully to random weights if checkpoints are missing.
    """

    def __init__(self):
        self.transformer: BehavioralTransformer    = None
        self.graphsage:   GraphSAGEFraudDetector   = None
        self.cnn_gnn:     CNNGNNHybrid             = None
        self.tssgc:       SIMSwapDetector           = None
        self.gan_autoencoder: GANAutoencoderKYC    = None
        self.ensemble  = ModelEnsemble()
        self._loaded   = False
        self._ckpt_status: dict[str, bool] = {}

    async def load_all(self):
        """Load all five models. Runs synchronously in an executor to avoid blocking."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_models_sync)
        self._loaded = True

    def _load_models_sync(self):
        print(f"\n🔄 Loading models from {CHECKPOINT_DIR}/")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        self.transformer     = _load_model("transformer")
        self.graphsage       = _load_model("graphsage")
        self.cnn_gnn         = _load_model("cnn_gnn")
        self.tssgc           = _load_model("tssgc")
        self.gan_autoencoder = _load_model("gan_autoencoder")

        # Record which models have real checkpoints vs. random weights
        self._ckpt_status = {
            name: (CHECKPOINT_DIR / fname).exists()
            for name, (_, _, fname) in MODEL_REGISTRY.items()
        }

        trained   = sum(self._ckpt_status.values())
        untrained = len(self._ckpt_status) - trained
        print(f"\n  {trained}/5 models loaded from checkpoints.")
        if untrained:
            print(f"  ⚠️  {untrained} model(s) running on random weights — "
                  f"run model_training.ipynb to fix this.")

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

        # Extract inputs for each model from the graph snapshot
        node_features    = graph_snapshot.get("node_features",    np.zeros((10, 64),  dtype=np.float32))
        edge_index       = graph_snapshot.get("edge_index",       np.zeros((2, 10),   dtype=np.int64))
        target_node_idx  = graph_snapshot.get("target_node_idx",  0)
        payload_features = graph_snapshot.get("payload_features", np.zeros(64,        dtype=np.float32))
        graph_embedding  = graph_snapshot.get("graph_embedding",  np.zeros(64,        dtype=np.float32))
        device_sequence  = graph_snapshot.get("device_sequence",  np.zeros((10, 32),  dtype=np.float32))
        account_history  = graph_snapshot.get("account_history",  np.zeros(32,        dtype=np.float32))
        kyc_features     = graph_snapshot.get("kyc_features",     np.zeros(128,       dtype=np.float32))

        loop = asyncio.get_event_loop()

        results = await asyncio.gather(
            loop.run_in_executor(
                None, self.transformer.predict, sequence_vector
            ),
            loop.run_in_executor(
                None, self.graphsage.predict_node, node_features, edge_index, target_node_idx
            ),
            loop.run_in_executor(
                None, self.cnn_gnn.predict, payload_features, graph_embedding
            ),
            loop.run_in_executor(
                None, self.tssgc.predict, device_sequence, account_history
            ),
            loop.run_in_executor(
                None, self.gan_autoencoder.predict, kyc_features
            ),
        )

        transformer_score, graphsage_score, cnn_gnn_score, tssgc_score, gan_score = results

        return self.ensemble.compute_unified_score(
            transformer_score    = transformer_score,
            graphsage_score      = graphsage_score,
            cnn_gnn_score        = cnn_gnn_score,
            tssgc_score          = tssgc_score,
            gan_autoencoder_score= gan_score,
        )

    def checkpoint_status(self) -> dict:
        """Returns which models are running on trained vs. random weights."""
        return {
            name: "✅ trained" if loaded else "⚠️  random weights"
            for name, loaded in self._ckpt_status.items()
        }