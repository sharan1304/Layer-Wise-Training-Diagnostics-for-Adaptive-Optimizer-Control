"""Update fusion and warm-up/blend scheduling."""

from __future__ import annotations

from typing import List

import torch

from candidates import CandidateUpdates


class UpdateFusionEngine:
    def fuse(self, candidates: CandidateUpdates, weights: List[float]) -> torch.Tensor:
        if len(weights) != 4:
            raise ValueError("Need exactly four fusion weights")
        total = sum(weights)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Fusion weights must sum to 1.0, got {total}")
        w1, w2, w3, w4 = weights
        return (
            w1 * candidates.adamw
            + w2 * candidates.rmsprop
            + w3 * candidates.sgd
            + w4 * candidates.normalized
        )

    def blend(
        self,
        delta_ai: torch.Tensor,
        delta_adamw: torch.Tensor,
        step: int,
        warmup: int = 50,
        blend_steps: int = 50,
    ) -> torch.Tensor:
        if step < warmup:
            return delta_adamw
        if step < warmup + blend_steps:
            blend = max(0.0, min(1.0, (step - warmup) / max(blend_steps, 1)))
            return (1.0 - blend) * delta_adamw + blend * delta_ai
        return delta_ai
