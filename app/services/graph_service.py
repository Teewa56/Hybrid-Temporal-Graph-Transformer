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
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

NODE_FEATURE_DIM = 64
GRAPH_EMBEDDING_DIM = 64


def _extract_node_features(account_data: dict) -> np.ndarray:
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
    features = features + [0.0] * (NODE_FEATURE_DIM - len(features))
    return np.array(features[:NODE_FEATURE_DIM], dtype=np.float32)


def _resolve_email(body: dict) -> str:
    """
    Card webhooks: 'customer_email' (normalised from upstream payment webhook email).
    VA webhooks: no email — use customer_identifier as graph node key instead.
    """
    return (
        body.get("customer_email")
        or body.get("customer_identifier")
        or "unknown"
    )


class GraphService:
    """
    Manages the live transaction graph in Neo4j.
    Compatible with both card/transfer and virtual account webhook payloads
    via the normalised body format produced in webhooks.py.
    """

    def __init__(self):
        self.driver = None
        if NEO4J_AVAILABLE:
            self.driver = AsyncGraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )

    async def update_and_fetch(self, body: dict) -> dict:
        if not NEO4J_AVAILABLE or not self.driver:
            return self._synthetic_snapshot(body)
        try:
            return await self._neo4j_update_and_fetch(body)
        except Exception as e:
            print(f"⚠️  Neo4j error: {e}. Using synthetic snapshot.")
            return self._synthetic_snapshot(body)

    async def _neo4j_update_and_fetch(self, body: dict) -> dict:
        sender_key = _resolve_email(body)

        # ip_address and device_id are optional from the upstream webhook — default gracefully
        device_id = body.get("device_id") or "unknown-device"
        ip_address = body.get("ip_address") or "0.0.0.0"

        recipient = body.get("meta", {}).get("receiver_account", "unknown")
        amount = float(body.get("amount", 0)) / 100
        txn_ref = body.get("TransactionRef") or body.get("transaction_ref", "")

        async with self.driver.session() as session:
            await session.run("""
                MERGE (sender:Account {key: $sender_key})
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
            """, sender_key=sender_key, recipient=recipient,
                 device_id=device_id, ip=ip_address,
                 txn_ref=txn_ref, amount=amount)

            result = await session.run("""
                MATCH (a:Account {key: $sender_key})-[*1..2]-(neighbor)
                RETURN neighbor, labels(neighbor) as types
                LIMIT 50
            """, sender_key=sender_key)

            records = [r.data() async for r in result]

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
        txn_ref = body.get("TransactionRef") or body.get("transaction_ref", "")
        features = [
            float(body.get("amount", 0)) / 10_000_000.0,
            float(len(txn_ref) >= 10),
            # ip_address may be missing in webhook payloads — encode as 0
            float(bool(body.get("ip_address", ""))),
            float(bool(body.get("device_id", ""))),
            float(bool(body.get("customer_email", "") or body.get("customer_identifier", ""))),
            float(body.get("currency", "NGN") == "NGN"),
        ]
        features = features + [0.0] * (64 - len(features))
        return np.array(features[:64], dtype=np.float32)

    def _synthetic_snapshot(self, body: dict) -> dict:
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