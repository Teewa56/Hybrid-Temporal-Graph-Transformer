import uuid
import random
import numpy as np
from dataclasses import dataclass

from synthetic_data_generator.config import CONFIG, NIGERIAN_STATES, NIGERIAN_CARRIERS

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False


@dataclass
class AccountNode:
    account_id: str
    email: str
    age_days: int
    transaction_count: int
    total_volume: float
    is_verified: bool
    has_bvn: bool
    dispute_count: int
    international_transfers: int
    unique_devices: int
    state: str
    is_fraud: bool = False
    is_mule: bool = False


def _make_account(is_fraud: bool = False, is_mule: bool = False) -> AccountNode:
    uid = str(uuid.uuid4())[:8]
    age = random.randint(1, 5) if is_mule else random.randint(30, 1825)
    return AccountNode(
        account_id=uid,
        email=f"user_{uid}@gmail.com",
        age_days=age,
        transaction_count=random.randint(1, 20) if is_mule else random.randint(10, 500),
        total_volume=random.uniform(1000, 500000),
        is_verified=False if is_mule else random.random() > 0.2,
        has_bvn=False if is_mule else random.random() > 0.3,
        dispute_count=random.randint(0, 3) if is_fraud else 0,
        international_transfers=random.randint(2, 15) if is_fraud else random.randint(0, 2),
        unique_devices=random.randint(3, 8) if is_fraud else random.randint(1, 2),
        state=random.choice(NIGERIAN_STATES),
        is_fraud=is_fraud,
        is_mule=is_mule,
    )


def _account_to_feature_vector(account: AccountNode, dim: int = 64) -> np.ndarray:
    features = [
        account.age_days / 1825.0,
        account.transaction_count / 500.0,
        account.total_volume / 5_000_000.0,
        float(account.is_verified),
        float(account.has_bvn),
        account.dispute_count / 10.0,
        account.international_transfers / 20.0,
        account.unique_devices / 10.0,
        float(account.is_fraud),
        float(account.is_mule),
    ]
    features += [0.0] * (dim - len(features))
    return np.array(features[:dim], dtype=np.float32)


class GraphBuilder:
    """
    Constructs a heterogeneous transaction graph from synthetic account data.
    Nodes: accounts, devices, IPs, merchants.
    Edges: transactions, shared device usage, shared IP clusters.
    Exports as NetworkX graph and PyTorch Geometric-compatible arrays.
    """

    def __init__(self, seed: int = None):
        seed = seed or CONFIG.graph.random_seed
        random.seed(seed)
        np.random.seed(seed)
        self.graph = nx.DiGraph() if NX_AVAILABLE else None

    def build(
        self,
        n_legit: int = None,
        n_mule: int = None,
    ) -> dict:
        n_legit = n_legit or CONFIG.graph.n_legit_accounts
        n_mule = n_mule or CONFIG.graph.n_mule_accounts

        accounts = (
            [_make_account(is_fraud=False, is_mule=False) for _ in range(n_legit)]
            + [_make_account(is_fraud=True, is_mule=True) for _ in range(n_mule)]
        )

        account_ids = [a.account_id for a in accounts]
        labels = np.array([int(a.is_fraud) for a in accounts], dtype=np.int64)
        node_features = np.stack(
            [_account_to_feature_vector(a) for a in accounts], axis=0
        )

        # Build edges: random transactions between accounts
        edges = []
        n = len(accounts)
        for i in range(n):
            n_edges = random.randint(1, 5)
            for _ in range(n_edges):
                j = random.randint(0, n - 1)
                if i != j:
                    edges.append((i, j))

        # Additional edges: mule accounts share devices and IPs
        mule_indices = [i for i, a in enumerate(accounts) if a.is_mule]
        for i in range(0, len(mule_indices) - 1, 2):
            edges.append((mule_indices[i], mule_indices[i + 1]))
            edges.append((mule_indices[i + 1], mule_indices[i]))

        edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)

        if NX_AVAILABLE:
            for i, acc in enumerate(accounts):
                self.graph.add_node(i, **{
                    "account_id": acc.account_id,
                    "is_fraud": acc.is_fraud,
                    "is_mule": acc.is_mule,
                })
            for src, dst in edges:
                self.graph.add_edge(src, dst)

        return {
            "accounts": accounts,
            "account_ids": account_ids,
            "node_features": node_features,      # (n_nodes, 64)
            "edge_index": edge_index,             # (2, n_edges)
            "labels": labels,                     # (n_nodes,)
            "n_nodes": n,
            "n_edges": len(edges),
        }

    def get_networkx_graph(self):
        return self.graph