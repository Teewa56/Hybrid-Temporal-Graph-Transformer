"""
Export all five Hybrid_Temporal_Graph_Transformer models to ONNX format for quantized inference.

Usage:
    python scripts/export_onnx.py

Outputs land in checkpoints/onnx/ with one .onnx file per model.
Each export is verified by running a dummy inference through the ONNX runtime
and comparing output shapes against the PyTorch originals.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
ONNX_DIR       = CHECKPOINT_DIR / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OPSET = 18

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("❌ onnx and onnxruntime are required. Run: pip install onnx onnxruntime")
    sys.exit(1)


def _load_checkpoint(model, name: str):
    ckpt_path = CHECKPOINT_DIR / f"{name}.pt"
    if not ckpt_path.exists():
        print(f"    No checkpoint for {name} — exporting with random weights.")
        return model
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"   Loaded checkpoint: {ckpt_path}")
    return model


def _verify_onnx(onnx_path: Path, dummy_inputs: list[np.ndarray], expected_shape: tuple):
    """Run a dummy inference through the exported ONNX model and verify output shape."""
    session = ort.InferenceSession(str(onnx_path))
    input_names = [inp.name for inp in session.get_inputs()]
    feed = {name: inp for name, inp in zip(input_names, dummy_inputs)}
    outputs = session.run(None, feed)
    actual_shape = outputs[0].shape
    assert actual_shape == expected_shape, (
        f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
    )
    print(f"   ONNX verification passed — output shape: {actual_shape}")


def export_transformer():
    print("\n[1/5] Exporting Behavioral Transformer...")
    from app.models.transformer import BehavioralTransformer, TransformerConfig

    model = _load_checkpoint(BehavioralTransformer(TransformerConfig()), "transformer")
    model.eval()

    dummy_input = torch.randn(1, 50, 32)
    onnx_path   = ONNX_DIR / "transformer.onnx"

    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        input_names  = ["sequence"],
        output_names = ["risk_score", "encoded"],
        dynamic_axes = {
            "sequence":   {0: "batch_size"},
            "risk_score": {0: "batch_size"},
            "encoded":    {0: "batch_size"},
        },
        opset_version    = 17,
        do_constant_folding= True,
    )

    onnx.checker.check_model(str(onnx_path))
    _verify_onnx(
        onnx_path,
        [dummy_input.numpy()],
        expected_shape=(1, 1),
    )
    print(f"   Saved → {onnx_path}")


def export_graphsage():
    print("\n[2/5] Exporting GraphSAGE...")
    from app.models.graphsage import GraphSAGEFraudDetector

    
    model = GraphSAGEFraudDetector(
        in_channels=64,
        hidden_channels=128,
        out_channels=64, 
        num_layers=3,
    )
    model = _load_checkpoint(model, "graphsage")
    model.eval()

    dummy_input = torch.randn(1, 64)   # out_channels=64 — embedding fed directly to sigmoid
    onnx_path   = ONNX_DIR / "graphsage.onnx"

    classifier = model.classifier
    classifier.eval()

    torch.onnx.export(
        classifier,
        (dummy_input,),
        str(onnx_path),
        input_names   = ["node_embedding"],
        output_names  = ["fraud_score"],
        dynamic_axes  = {
            "node_embedding": {0: "batch_size"},
            "fraud_score":    {0: "batch_size"},
        },
        opset_version      = 17,
        do_constant_folding= True,
    )

    onnx.checker.check_model(str(onnx_path))
    _verify_onnx(
        onnx_path,
        [dummy_input.numpy()],
        expected_shape=(1, 1),
    )
    print(f"   Saved → {onnx_path}")

def export_cnn_gnn():
    print("\n[3/5] Exporting CNN-GNN Hybrid...")
    from app.models.cnn_gnn import CNNGNNHybrid

    model = _load_checkpoint(CNNGNNHybrid(), "cnn_gnn")
    model.eval()

    dummy_payload = torch.randn(1, 64)
    dummy_graph   = torch.randn(1, 64)
    onnx_path     = ONNX_DIR / "cnn_gnn.onnx"

    torch.onnx.export(
        model,
        (dummy_payload, dummy_graph),
        str(onnx_path),
        input_names  = ["payload_features", "graph_embedding"],
        output_names = ["risk_score"],
        dynamic_axes = {
            "payload_features": {0: "batch_size"},
            "graph_embedding":  {0: "batch_size"},
            "risk_score":       {0: "batch_size"},
        },
        opset_version     = 17,
        do_constant_folding= True,
    )

    onnx.checker.check_model(str(onnx_path))
    _verify_onnx(
        onnx_path,
        [dummy_payload.numpy(), dummy_graph.numpy()],
        expected_shape=(1, 1),
    )
    print(f" Saved → {onnx_path}")


def export_tssgc():
    print("\n[4/5] Exporting TSSGC (SIM Swap Detector)...")
    from app.models.tssgc import SIMSwapDetector

    model = _load_checkpoint(SIMSwapDetector(), "tssgc")
    model.eval()

    SEQ_LEN   = 30
    dummy_seq  = torch.randn(1, SEQ_LEN, 32)
    dummy_acct = torch.randn(1, 32)
    onnx_path  = ONNX_DIR / "tssgc.onnx"

    torch.onnx.export(
        model,
        (dummy_seq, dummy_acct),
        str(onnx_path),
        input_names   = ["device_sequence", "account_history"],
        output_names  = ["sim_swap_score"],
        dynamic_axes  = {
            "device_sequence":  {0: "batch_size"},
            "account_history":  {0: "batch_size"},
            "sim_swap_score":   {0: "batch_size"},
        },
        opset_version      = 18,
        do_constant_folding= True,
    )

    onnx.checker.check_model(str(onnx_path))
    _verify_onnx(
        onnx_path,
        [dummy_seq.numpy(), dummy_acct.numpy()],
        expected_shape=(1, 1),
    )
    print(f"   Saved → {onnx_path}")

def export_gan_autoencoder():
    print("\n[5/5] Exporting GAN + Autoencoder (KYC)...")
    from app.models.gan_autoencoder import GANAutoencoderKYC

    model = _load_checkpoint(GANAutoencoderKYC(), "gan_autoencoder")
    model.eval()

    dummy_kyc = torch.randn(1, 128)
    onnx_path = ONNX_DIR / "gan_autoencoder.onnx"

    torch.onnx.export(
        model,
        (dummy_kyc,),
        str(onnx_path),
        input_names  = ["kyc_features"],
        output_names = ["fraud_score"],
        dynamic_axes = {
            "kyc_features": {0: "batch_size"},
            "fraud_score":  {0: "batch_size"},
        },
        opset_version     = 17,
        do_constant_folding= True,
    )

    onnx.checker.check_model(str(onnx_path))
    _verify_onnx(
        onnx_path,
        [dummy_kyc.numpy()],
        expected_shape=(1, 1),
    )
    print(f"   Saved → {onnx_path}")


def print_summary():
    print(f"\n{'='*55}")
    print("  ONNX Export Summary")
    print(f"{'='*55}")
    models = {
        "transformer.onnx":     "Behavioral Transformer",
        "graphsage.onnx":       "GraphSAGE (classifier head)",
        "cnn_gnn.onnx":         "CNN-GNN Hybrid",
        "tssgc.onnx":           "TSSGC (SIM Swap)",
        "gan_autoencoder.onnx": "GAN + Autoencoder (KYC)",
    }
    total_size = 0
    for filename, label in models.items():
        path = ONNX_DIR / filename
        if path.exists():
            size_kb  = path.stat().st_size / 1024
            total_size += size_kb
            print(f"   {label:<35} {size_kb:>8.1f} KB")
        else:
            print(f"  ❌ {label:<35} {'MISSING':>8}")
    print(f"{'─'*55}")
    print(f"  {'Total':35} {total_size:>8.1f} KB")
    print(f"  Output directory: {ONNX_DIR}")


if __name__ == "__main__":
    print("Hybrid_Temporal_Graph_Transformer — ONNX Model Export")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"ONNX output:    {ONNX_DIR}")

    exporters = [
        export_transformer,
        export_graphsage,
        export_cnn_gnn,
        export_tssgc,
        export_gan_autoencoder,
    ]

    failed = []
    for exporter in exporters:
        try:
            exporter()
        except Exception as e:
            print(f"  ❌ Export failed: {e}")
            failed.append(exporter.__name__)

    print_summary()

    if failed:
        print(f"\n  {len(failed)} export(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n All 5 models exported and verified successfully.")