import json
import asyncio
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import deque
from typing import Optional

from app.models.ensemble import EnsembleScores


@dataclass
class UncertainSample:
    transaction_ref: str
    body: dict
    scores: dict
    unified_score: float
    queued_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    label: Optional[int] = None        # 1=fraud, 0=legit — set by human reviewer
    reviewed: bool = False
    review_notes: str = ""


class ActiveLearningQueue:
    """
    Collects uncertain predictions for priority human review.
    Confirmed labels are fed back to the EWC fine-tuning pipeline.
    Maximizes the value of every labeled example added to the training set.

    Uncertainty criterion: unified score in band 0.45–0.75.
    """

    def __init__(self, max_size: int = 500):
        self.queue: deque[UncertainSample] = deque(maxlen=max_size)
        self._reviewed_buffer: list[UncertainSample] = []
        self._lock = asyncio.Lock()

    async def maybe_enqueue(
        self,
        transaction_ref: str,
        body: dict,
        scores: EnsembleScores,
    ) -> bool:
        """
        Enqueue a transaction if it falls in the uncertainty band.
        Returns True if enqueued.
        """
        if not (0.45 <= scores.unified_score <= 0.75):
            return False

        sample = UncertainSample(
            transaction_ref=transaction_ref,
            body=body,
            scores=scores.to_dict(),
            unified_score=scores.unified_score,
        )

        async with self._lock:
            self.queue.append(sample)

        print(f"📥 Active Learning Queue: {transaction_ref} enqueued "
              f"(score: {scores.unified_score:.3f})")
        return True

    async def submit_label(
        self,
        transaction_ref: str,
        label: int,
        notes: str = "",
    ) -> bool:
        """
        Human reviewer submits a label (1=fraud, 0=legit) for a queued sample.
        Labeled samples are moved to the reviewed buffer for EWC fine-tuning.
        """
        async with self._lock:
            for sample in self.queue:
                if sample.transaction_ref == transaction_ref:
                    sample.label = label
                    sample.reviewed = True
                    sample.review_notes = notes
                    self._reviewed_buffer.append(sample)
                    return True
        return False

    async def get_pending_reviews(self, limit: int = 20) -> list[dict]:
        """Return the highest-uncertainty unreviewed samples for the review dashboard."""
        async with self._lock:
            unreviewed = [s for s in self.queue if not s.reviewed]
            # Sort by uncertainty — closest to 0.60 is most uncertain
            unreviewed.sort(key=lambda s: abs(s.unified_score - 0.60))
            return [asdict(s) for s in unreviewed[:limit]]

    def drain_reviewed(self) -> list[UncertainSample]:
        """
        Called by EWC fine-tuning pipeline to retrieve confirmed labels.
        Clears the reviewed buffer after draining.
        """
        samples = self._reviewed_buffer.copy()
        self._reviewed_buffer.clear()
        return samples

    def queue_size(self) -> int:
        return len(self.queue)

    def reviewed_count(self) -> int:
        return len(self._reviewed_buffer)