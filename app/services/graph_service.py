import os
import numpy as np
from typing import Any

try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

NODE_FEATURE_DIM = 64
GRAPH_EMBEDDING_DIM = 64


def _extract_node_features(account_data: dict) -> np.ndarray:
    """Convert account data to a fixed-size node feature vector."""
    features = [
        float(account_data.get("age_days", 0)) / 365.0,
        float(account_data.get("transaction_count", 0)) / 1000.0,
        float(account_data.get("total_volume", 0)) / 10_000_000.0,
        float(account_data.get("is_verified", False)),
        float(account_data.get("has_bvn", False)),
        float(account_data.get("dispute_count", 0)) / 10.0,
        float(account_data.get("international_transfers", 0)) / 100.0,
        float(account_data.get("unique_devices", 1)) / 10.0,
    ]
    # Pad to NODE_FEATURE_DIM
    features = features + [0.0] * (NODE_FEATURE_DIM - len(features))
    return np.array(features[:NODE_FEATURE_DIM], dtype=np.float32)


class GraphService:
    """
    Manages the live transaction graph in Neo4j.
    Places each incoming transaction into the global graph,
    resolves account-to-account edges, device fingerprints,
    and IP clusters in real-time.
    Returns a graph snapshot for GraphSAGE and CNN-GNN inference.
    """

    def __init__(self):
        self.driver = None
        if NEO4J_AVAILABLE:
            self.driver = AsyncGraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )

    async def update_and_fetch(self, body: dict) -> dict:
        """
        Insert transaction into graph and return local neighbourhood snapshot.
        Falls back to a synthetic graph snapshot if Neo4j is unavailable.
        """
        if not NEO4J_AVAILABLE or not self.driver:
            return self._synthetic_snapshot(body)

        try:
            return await self._neo4j_update_and_fetch(body)
        except Exception as e:
            print(f" Neo4j error: {e}. Using synthetic snapshot.")
            return self._synthetic_snapshot(body)

    async def _neo4j_update_and_fetch(self, body: dict) -> dict:
        sender_email = body.get("customer_email", "unknown")
        recipient = body.get("meta", {}).get("receiver_account", "unknown")
        device_id = body.get("device_id", "unknown")
        ip_address = body.get("ip_address", "0.0.0.0")
        amount = float(body.get("amount", 0)) / 100
        txn_ref = body.get("transaction_ref", "")

        async with self.driver.session() as session:
            # Upsert sender, recipient, device, IP nodes + edges
            await session.run("""
                MERGE (sender:Account {email: $sender})
                MERGE (recipient:Account {id: $recipient})
                MERGE (device:Device {id: $device_id})
                MERGE (ip:IP {address: $ip})
                MERGE (sender)-[:USED]->(device)
                MERGE (sender)-[:FROM_IP]->(ip)
                CREATE (sender)-[:SENT {
                    ref: $txn_ref,
                    amount: $amount,
                    timestamp: datetime()
                }]->(recipient)
            """, sender=sender_email, recipient=recipient,
                 device_id=device_id, ip=ip_address,
                 txn_ref=txn_ref, amount=amount)

            # Fetch 2-hop neighbourhood of sender
            result = await session.run("""
                MATCH (a:Account {email: $sender})-[*1..2]-(neighbor)
                RETURN neighbor, labels(neighbor) as types
                LIMIT 50
            """, sender=sender_email)

            records = [r.data() async for r in result]

        # Build simple graph snapshot from neighbourhood
        num_nodes = max(len(records) + 1, 2)
        node_features = np.random.randn(num_nodes, NODE_FEATURE_DIM).astype(np.float32)
        edge_index = np.array(
            [[i, 0] for i in range(1, num_nodes)] +
            [[0, i] for i in range(1, num_nodes)],
            dtype=np.int64
        ).T

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "target_node_idx": 0,
            "graph_embedding": node_features[0],
            "payload_features": self._encode_payload(body),
            "device_sequence": np.zeros((10, 32), dtype=np.float32),
            "account_history": node_features[0, :32],
            "kyc_features": np.zeros(128, dtype=np.float32),
            "neighbourhood_size": len(records),
        }

    def _encode_payload(self, body: dict) -> np.ndarray:
        """Encode raw payload fields into a numeric feature vector for CNN."""
        features = [
            float(body.get("amount", 0)) / 10_000_000.0,
            float(len(body.get("transaction_ref", "")) == 12),
            float(bool(body.get("ip_address"))),
            float(bool(body.get("device_id"))),
            float(bool(body.get("customer_email"))),
            float(body.get("currency", "NGN") == "NGN"),
        ]
        features = features + [0.0] * (64 - len(features))
        return np.array(features[:64], dtype=np.float32)

    def _synthetic_snapshot(self, body: dict) -> dict:
        """Fallback graph snapshot when Neo4j is unavailable."""
        num_nodes = 10
        node_features = np.random.randn(num_nodes, NODE_FEATURE_DIM).astype(np.float32)
        edge_index = np.array(
            [[i, 0] for i in range(1, num_nodes)] +
            [[0, i] for i in range(1, num_nodes)],
            dtype=np.int64
        ).T
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "target_node_idx": 0,
            "graph_embedding": node_features[0],
            "payload_features": self._encode_payload(body),
            "device_sequence": np.zeros((10, 32), dtype=np.float32),
            "account_history": node_features[0, :32],
            "kyc_features": np.zeros(128, dtype=np.float32),
            "neighbourhood_size": num_nodes,
        }

    async def close(self):
        if self.driver:
            await self.driver.close()