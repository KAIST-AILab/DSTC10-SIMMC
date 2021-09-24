from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class DisambiguateAccuracy(Metric):
    def __init__(
        self,
        compute_on_step: bool=True,
        dist_sync_on_step: bool=False,
        process_group: Optional[Any]=None,
        dist_sync_fn: Optional[Callable]=None
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state(
            "n_correct", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_total", torch.tensor(0, dtype=torch.float),
            dist_reduce_fx="sum"
        )

    def update(
        self,
        predicted: torch.Tensor,
        labels: torch.Tensor
    ):
        # Get predictions
        _, pred = predicted.max(-1)
        # Mask out turns without labels
        mask = (labels != -100)
        # Compute correct instances only on predictions with labels
        crt = ((pred == labels) * mask).sum()
        self.n_correct += crt
        self.n_total += mask.sum()

    def compute(self):
        acc = self.n_correct / self.n_total if self.n_total.item() != 0 else 0
        return acc
    
    @property
    def is_differentiable(self) -> bool:
        return False