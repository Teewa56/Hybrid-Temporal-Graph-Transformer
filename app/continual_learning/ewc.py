import torch
import torch.nn as nn
from copy import deepcopy
from typing import Any


class EWC:
    """
    Elastic Weight Consolidation (EWC) for continual learning.

    Allows incremental fine-tuning of any sub-model on new fraud patterns
    while mathematically penalizing changes to weights that are critical
    for detecting already-learned fraud types — preventing catastrophic forgetting.

    Usage:
        ewc = EWC(model, dataloader_of_old_tasks)
        loss = task_loss + ewc.penalty(model)
    """

    def __init__(self, model: nn.Module, dataloader, importance: float = 1000.0):
        self.model = model
        self.importance = importance
        self._means: dict[str, torch.Tensor] = {}
        self._fisher: dict[str, torch.Tensor] = {}

        self._compute_fisher(dataloader)
        self._store_means()

    def _compute_fisher(self, dataloader):
        """
        Compute the diagonal of the Fisher Information Matrix
        by averaging squared gradients over old task samples.
        The Fisher approximates how important each weight is
        for the tasks the model already knows.
        """
        self.model.eval()
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        for batch in dataloader:
            self.model.zero_grad()

            # Handle variable input formats across models
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            if isinstance(inputs, torch.Tensor):
                output = self.model(inputs)
            else:
                continue

            # Use log-likelihood proxy: log of the output probability
            if isinstance(output, tuple):
                output = output[0]

            log_likelihood = output.sum()
            log_likelihood.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2)

        n_batches = max(len(dataloader), 1)
        self._fisher = {n: f / n_batches for n, f in fisher.items()}

    def _store_means(self):
        """Store current parameter values as the reference point (θ*)."""
        self._means = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty term: λ/2 * Σ F_i * (θ_i - θ*_i)²
        Add this to your task loss during fine-tuning.
        """
        loss = torch.tensor(0.0, requires_grad=True)
        for n, p in model.named_parameters():
            if n in self._means and n in self._fisher:
                loss = loss + (
                    self._fisher[n] * (p - self._means[n]).pow(2)
                ).sum()
        return (self.importance / 2) * loss

    def update_reference(self, model: nn.Module, dataloader):
        """
        After fine-tuning on a new fraud type, update the reference parameters
        and Fisher matrix to include the new task's knowledge.
        """
        self.model = model
        self._compute_fisher(dataloader)
        self._store_means()
        print("✅ EWC reference updated with new fraud pattern weights.")