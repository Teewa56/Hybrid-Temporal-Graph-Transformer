import json
import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from app.models.ensemble import EnsembleScores
from app.services.decision_engine import DecisionResult


AUDIT_LOG_PATH = Path("logs/audit_trail.jsonl")


@dataclass
class AuditRecord:
    transaction_ref: str
    timestamp: str
    unified_score: float
    zone: str
    action_taken: str
    scores_breakdown: dict
    top_signals: list[str]
    customer_email: str
    amount: float
    channel: str
    ip_address: str
    device_id: str
    shap_explanation: Optional[dict] = None
    reviewer_label: Optional[int] = None   # Set post-review
    reviewer_notes: str = ""
    model_version: str = "1.0.0"
    environment: str = field(default_factory=lambda: __import__('os').getenv("ENV", "sandbox"))


class AuditTrail:
    """
    CBN-compliant audit trail for all fraud decisions.

    Every Red and Amber Zone decision is logged with:
    - Full score breakdown per model
    - Top contributing SHAP features
    - Action taken and backend response
    - Immutable timestamp and model version

    Logs are written as newline-delimited JSON (JSONL)
    for easy ingestion into any SIEM or compliance dashboard.
    """

    def __init__(self, log_path: Path = AUDIT_LOG_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def log(
        self,
        transaction_ref: str,
        scores: EnsembleScores,
        decision: DecisionResult,
        body: dict,
        shap_explanation: Optional[dict] = None,
    ):
        """Write an audit record for every Amber and Red Zone decision."""
        if decision.zone == "GREEN":
            # Only log green zone decisions at reduced verbosity
            return

        record = AuditRecord(
            transaction_ref=transaction_ref,
            timestamp=datetime.utcnow().isoformat() + "Z",
            unified_score=decision.unified_score,
            zone=decision.zone,
            action_taken=decision.action_taken,
            scores_breakdown=decision.scores_breakdown,
            top_signals=decision.top_signals,
            customer_email=body.get("customer_email", ""),
            amount=float(body.get("amount", 0)) / 100,
            channel=body.get("channel", "unknown"),
            ip_address=body.get("ip_address", ""),
            device_id=body.get("device_id", ""),
            shap_explanation=shap_explanation,
        )

        async with self._lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")

        print(f"📋 Audit logged: {transaction_ref} | {decision.zone} | {decision.unified_score:.3f}")

    async def get_recent(self, limit: int = 50) -> list[dict]:
        """Fetch the most recent audit records for the compliance dashboard."""
        if not self.log_path.exists():
            return []

        async with self._lock:
            with open(self.log_path, "r") as f:
                lines = f.readlines()

        records = []
        for line in reversed(lines[-limit:]):
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        return records

    async def annotate_review(
        self,
        transaction_ref: str,
        label: int,
        notes: str = "",
    ) -> bool:
        """
        Post-review annotation: attach human reviewer label to an existing record.
        Used for closing the active learning loop.
        """
        if not self.log_path.exists():
            return False

        async with self._lock:
            with open(self.log_path, "r") as f:
                lines = f.readlines()

            updated = False
            new_lines = []
            for line in lines:
                try:
                    record = json.loads(line.strip())
                    if record.get("transaction_ref") == transaction_ref:
                        record["reviewer_label"] = label
                        record["reviewer_notes"] = notes
                        updated = True
                    new_lines.append(json.dumps(record) + "\n")
                except json.JSONDecodeError:
                    new_lines.append(line)

            with open(self.log_path, "w") as f:
                f.writelines(new_lines)

        return updated