"""Importance-based visual-token pruning baseline."""

from __future__ import annotations

import torch

from .base import CompressionMethod, CompressionResult


class ImportanceBasedPruning(CompressionMethod):
    """Keep top-k visual tokens by L2 norm.

    Token norm is simple, stable, training-free, and does not require attention maps.
    Future extensions can replace _score_tokens with attention rollout or gradient-free
    saliency scores.
    """

    name = "importance"

    def _score_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(tokens.float(), ord=2, dim=-1)

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

        scores = self._score_tokens(tokens)
        topk = torch.topk(scores, k=keep, dim=1, largest=True).indices
        topk_sorted = torch.sort(topk, dim=1).values
        gather_idx = topk_sorted.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        pruned = torch.gather(tokens, dim=1, index=gather_idx)
        if squeeze_batch:
            pruned = pruned.squeeze(0)
            kept_indices = topk_sorted.squeeze(0)
        else:
            kept_indices = topk_sorted
        return CompressionResult(
            tokens=pruned,
            kept_indices=kept_indices,
            metadata={"original_tokens": num_tokens, "kept_tokens": keep, "batch": batch},
        )
