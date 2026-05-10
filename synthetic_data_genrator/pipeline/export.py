import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from synthetic_data_generator.config import CONFIG


OUTPUT_DIR = CONFIG.output_dir


def _numpy_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} not serializable")


def _save_json(data: list[dict], path: Path):
    with open(path, "w") as f:
        json.dump(data, f, default=_numpy_serializer, indent=2)
    print(f"    💾 Saved {len(data):,} records → {path.name}")


def _save_numpy(array: np.ndarray, path: Path):
    np.save(path, array)
    print(f"    💾 Saved array {array.shape} → {path.name}")


def export_behavioral(data: dict):
    folder = OUTPUT_DIR / "behavioral"
    folder.mkdir(parents=True, exist_ok=True)
    _save_json(data.get("data", []), folder / "transactions.json")


def export_graph(data: dict):
    folder = OUTPUT_DIR / "graph"
    folder.mkdir(parents=True, exist_ok=True)
    graph_data = data.get("data", {})

    if "node_features" in graph_data:
        _save_numpy(graph_data["node_features"], folder / "node_features.npy")
    if "edge_index" in graph_data:
        _save_numpy(graph_data["edge_index"], folder / "edge_index.npy")
    if "labels" in graph_data:
        _save_numpy(graph_data["labels"], folder / "labels.npy")

    meta = {k: v for k, v in graph_data.items()
            if k not in ("node_features", "edge_index", "labels")}
    with open(folder / "graph_meta.json", "w") as f:
        json.dump(meta, f, default=_numpy_serializer, indent=2)
    print(f"    💾 Saved graph metadata → graph_meta.json")


def export_payload(data: dict):
    folder = OUTPUT_DIR / "payload"
    folder.mkdir(parents=True, exist_ok=True)
    _save_json(data.get("data", []), folder / "payloads.json")


def export_sim_swap(data: dict):
    folder = OUTPUT_DIR / "sim_swap"
    folder.mkdir(parents=True, exist_ok=True)

    histories = data.get("data", [])
    serializable = []
    for h in histories:
        serializable.append({
            "user_id": h.user_id,
            "account_id": h.account_id,
            "has_sim_swap": h.has_sim_swap,
            "sim_swap_index": h.sim_swap_index,
            "events": [asdict(ev) for ev in h.events],
        })
    _save_json(serializable, folder / "device_histories.json")


def export_kyc(data: dict):
    folder = OUTPUT_DIR / "kyc"
    folder.mkdir(parents=True, exist_ok=True)

    docs = data.get("data", [])
    _save_json([asdict(d) for d in docs], folder / "documents.json")


def export_manifest(results: dict):
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": {
            "behavioral": {
                "n_users": CONFIG.behavioral.n_users,
                "fraud_injection_rate": CONFIG.behavioral.fraud_injection_rate,
            },
            "graph": {
                "n_legit_accounts": CONFIG.graph.n_legit_accounts,
                "n_fraud_rings": CONFIG.graph.n_fraud_rings,
            },
            "payload": {
                "n_legit_payloads": CONFIG.payload.n_legit_payloads,
                "n_anomalous_payloads": CONFIG.payload.n_anomalous_payloads,
            },
            "sim_swap": {
                "n_users": CONFIG.sim_swap.n_users,
                "n_sim_swap_events": CONFIG.sim_swap.n_sim_swap_events,
            },
            "kyc": {
                "n_legitimate_docs": CONFIG.kyc.n_legitimate_docs,
                "n_forged_docs": CONFIG.kyc.n_forged_docs,
            },
        },
        "summary": {
            module: {k: v for k, v in stats.items() if k != "data"}
            for module, stats in results.items()
            if isinstance(stats, dict) and "error" not in stats
        },
        "output_files": {
            "behavioral":  "behavioral/transactions.json",
            "graph":       "graph/node_features.npy + edge_index.npy + labels.npy",
            "payload":     "payload/payloads.json",
            "sim_swap":    "sim_swap/device_histories.json",
            "kyc":         "kyc/documents.json",
        },
    }

    with open(OUTPUT_DIR / "data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    📋 Manifest written → data_manifest.json")


def export_all(results: dict):
    print(f"\n  Output directory: {OUTPUT_DIR}")

    exporters = {
        "behavioral": export_behavioral,
        "graph":      export_graph,
        "payload":    export_payload,
        "sim_swap":   export_sim_swap,
        "kyc":        export_kyc,
    }

    for module, exporter in exporters.items():
        if module in results and "error" not in results[module]:
            print(f"\n  [{module.upper()}]")
            try:
                exporter(results[module])
            except Exception as e:
                print(f"    ❌ Export failed: {e}")

    export_manifest(results)
    print(f"\n  ✅ All exports complete → {OUTPUT_DIR}")