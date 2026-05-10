import random
import numpy as np

from synthetic_data_generator.config import CONFIG
from synthetic_data_generator.graph.graph_builder import GraphBuilder
from synthetic_data_generator.graph.mule_network_simulator import MuleNetworkSimulator


class FraudRingInjector:
    """
    Takes a clean transaction graph and injects coordinated fraud ring
    subgraphs at a configurable prevalence rate.
    Labels injected nodes and edges as fraudulent.
    Output is a single merged graph with fraud annotations.
    """

    def __init__(self, seed: int = None):
        seed = seed or CONFIG.graph.random_seed
        random.seed(seed)
        np.random.seed(seed)

        self.graph_builder = GraphBuilder(seed=seed)
        self.mule_simulator = MuleNetworkSimulator(seed=seed)

    def build_injected_graph(self) -> dict:
        # 1. Build clean base graph
        base = self.graph_builder.build(
            n_legit=CONFIG.graph.n_legit_accounts,
            n_mule=0,
        )

        # 2. Simulate mule networks
        networks = self.mule_simulator.simulate_batch(
            n_networks=CONFIG.graph.n_fraud_rings
        )
        mule_accounts, mule_edges = self.mule_simulator.to_edge_list(networks)

        # 3. Merge into base graph
        n_base = base["n_nodes"]
        mule_id_to_idx = {acc: n_base + i for i, acc in enumerate(mule_accounts)}

        # Extend node features with mule node features
        n_mule = len(mule_accounts)
        mule_features = np.random.randn(n_mule, base["node_features"].shape[1]).astype(np.float32)
        # Mule nodes: young accounts, high dispute counts
        mule_features[:, 0] = np.random.uniform(0.001, 0.01, n_mule)  # age_days very low
        mule_features[:, 6] = np.random.uniform(0.3, 0.8, n_mule)     # international_transfers high

        combined_features = np.vstack([base["node_features"], mule_features])
        combined_labels = np.concatenate([base["labels"], np.ones(n_mule, dtype=np.int64)])

        # Inject some edges from legit accounts into mule hub accounts
        # (simulates victims sending to mule hubs)
        injected_edges = list(base["edge_index"].T.tolist()) if base["edge_index"].size > 0 else []

        for net in networks:
            hub_idx = mule_id_to_idx.get(net.hub_account)
            if hub_idx is None:
                continue
            # Connect 2-4 random legit accounts to the hub (victim transactions)
            victim_indices = random.sample(range(n_base), min(4, n_base))
            for v in victim_indices:
                injected_edges.append([v, hub_idx])

        # Add mule-to-mule edges
        for src_id, dst_id in mule_edges:
            src_idx = mule_id_to_idx.get(src_id)
            dst_idx = mule_id_to_idx.get(dst_id)
            if src_idx is not None and dst_idx is not None:
                injected_edges.append([src_idx, dst_idx])

        edge_index = np.array(injected_edges, dtype=np.int64).T if injected_edges else np.zeros((2, 0), dtype=np.int64)

        fraud_rate = combined_labels.mean()
        print(f"📊 Injected graph: {len(combined_labels)} nodes | "
              f"{edge_index.shape[1]} edges | "
              f"Fraud rate: {fraud_rate:.2%}")

        return {
            "node_features": combined_features,
            "edge_index": edge_index,
            "labels": combined_labels,
            "n_nodes": len(combined_labels),
            "n_edges": edge_index.shape[1],
            "n_legit": n_base,
            "n_mule": n_mule,
            "fraud_rate": float(fraud_rate),
            "mule_networks": [
                {
                    "network_id": net.network_id,
                    "size": len(net.accounts),
                    "hop_depth": net.hop_depth,
                }
                for net in networks
            ],
        }