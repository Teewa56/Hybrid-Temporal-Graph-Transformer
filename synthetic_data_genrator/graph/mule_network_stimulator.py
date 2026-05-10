import uuid
import random
import numpy as np
from dataclasses import dataclass

from synthetic_data_generator.config import CONFIG

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False


@dataclass
class MuleNetwork:
    network_id: str
    accounts: list[str]         # Account IDs in the ring
    edges: list[tuple]          # (src, dst) transaction edges
    hub_account: str            # Primary receiving account
    exit_account: str           # Final disbursement account
    hop_depth: int
    labels: dict[str, int]      # account_id → 1 (fraud)


class MuleNetworkSimulator:
    """
    Simulates realistic mule account networks using
    Barabási–Albert preferential attachment model.

    Structure: victim funds → hub account → chain of mule accounts
    → exit account (international transfer or crypto).

    Even new accounts (age: hours) are correctly identified as mules
    by their connection topology — this is what GraphSAGE learns.
    """

    def __init__(self, seed: int = None):
        seed = seed or CONFIG.graph.random_seed
        random.seed(seed)
        np.random.seed(seed)

    def _generate_ba_graph(self, n: int, m: int = 2):
        """Barabási-Albert graph: new nodes preferentially attach to high-degree nodes."""
        if NX_AVAILABLE:
            return nx.barabasi_albert_graph(n, m, seed=random.randint(0, 9999))
        # Fallback: simple random graph
        edges = []
        for i in range(1, n):
            targets = random.sample(range(i), min(m, i))
            for t in targets:
                edges.append((i, t))
        return edges

    def simulate_one(
        self,
        ring_size: int = None,
        hop_depth: int = None,
    ) -> MuleNetwork:
        ring_size = ring_size or random.randint(
            CONFIG.graph.min_ring_size, CONFIG.graph.max_ring_size
        )
        hop_depth = hop_depth or CONFIG.graph.mule_hop_depth

        account_ids = [str(uuid.uuid4())[:8] for _ in range(ring_size)]
        hub = account_ids[0]
        exit_acc = account_ids[-1]

        # Build layered hop structure
        edges = []
        layer_size = max(1, ring_size // hop_depth)

        for hop in range(hop_depth - 1):
            src_layer_start = hop * layer_size
            dst_layer_start = (hop + 1) * layer_size
            src_layer = account_ids[src_layer_start: src_layer_start + layer_size]
            dst_layer = account_ids[dst_layer_start: dst_layer_start + layer_size]

            for src in src_layer:
                n_targets = random.randint(1, min(3, len(dst_layer)))
                targets = random.sample(dst_layer, n_targets)
                for dst in targets:
                    edges.append((src, dst))

        # Add BA-style cross-connections within the network
        if NX_AVAILABLE:
            ba = self._generate_ba_graph(ring_size, m=2)
            for u, v in ba.edges():
                edges.append((account_ids[u], account_ids[v]))

        return MuleNetwork(
            network_id=str(uuid.uuid4())[:8],
            accounts=account_ids,
            edges=edges,
            hub_account=hub,
            exit_account=exit_acc,
            hop_depth=hop_depth,
            labels={acc_id: 1 for acc_id in account_ids},
        )

    def simulate_batch(self, n_networks: int = None) -> list[MuleNetwork]:
        n_networks = n_networks or CONFIG.graph.n_fraud_rings
        return [self.simulate_one() for _ in range(n_networks)]

    def to_edge_list(self, networks: list[MuleNetwork]) -> tuple[list[str], list[tuple]]:
        """Flatten all networks into a global account list and edge list."""
        all_accounts = []
        all_edges = []
        for net in networks:
            all_accounts.extend(net.accounts)
            all_edges.extend(net.edges)
        return list(set(all_accounts)), all_edges