"""Fixed-ratio visual-token pruning baseline."""

from __future__ import annotations

import torch

from .base import CompressionMethod, CompressionResult


class FixedRatioPruning(CompressionMethod):
    """Keep evenly spaced visual tokens according to a fixed retention ratio."""

    name = "fixed"

    def compress_visual_tokens(self, tokens: torch.Tensor) -> CompressionResult:
        squeeze_batch = False
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
            squeeze_batch = True
        if tokens.ndim != 3:
            raise ValueError("Expected tokens with shape [B, N, D] or [N, D].")

        batch, num_tokens, _dim = tokens.shape
        keep = max(1, int(round(num_tokens * self.retention_ratio)))
        if keep >= num_tokens:
            output = tokens.squeeze(0) if squeeze_batch else tokens
            return CompressionResult(tokens=output, kept_indices=None)

        indices = torch.linspace(
            0,
            num_tokens - 1,
            steps=keep,
            device=tokens.device,
        ).round().long()
        pruned = tokens.index_select(dim=1, index=indices)
        if squeeze_batch:
            pruned = pruned.squeeze(0)
        return CompressionResult(
            tokens=pruned,
            kept_indices=indices,
            metadata={"original_tokens": num_tokens, "kept_tokens": keep, "batch": batch},
        )
