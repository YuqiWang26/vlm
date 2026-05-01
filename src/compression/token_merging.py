"""Token merging baseline with cosine-similarity anchor clustering."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import CompressionMethod, CompressionResult


def _merge_single_sequence(tokens: torch.Tensor, keep: int) -> torch.Tensor:
    """Merge [N, D] tokens into keep anchors by cosine assignment."""

    num_tokens = tokens.shape[0]
    if keep >= num_tokens:
        return tokens

    anchor_indices = torch.linspace(
        0,
        num_tokens - 1,
        steps=keep,
        device=tokens.device,
    ).round().long()
    anchors = tokens.index_select(dim=0, index=anchor_indices)
    norm_tokens = F.normalize(tokens.float(), dim=-1)
    norm_anchors = F.normalize(anchors.float(), dim=-1)
    assignments = torch.argmax(norm_tokens @ norm_anchors.T, dim=-1)

    merged = []
    for cluster_id in range(keep):
        mask = assignments == cluster_id
        if torch.any(mask):
            merged.append(tokens[mask].mean(dim=0))
        else:
            merged.append(anchors[cluster_id])
    return torch.stack(merged, dim=0).to(dtype=tokens.dtype)


def compress_visual_tokens(tokens: torch.Tensor, ratio: float) -> torch.Tensor:
    """Standalone token merging function for experiments outside the VLM forward pass."""

    method = TokenMerging(retention_ratio=ratio, apply_proxy_image_budget=False)
    return method.compress_visual_tokens(tokens).tokens


class TokenMerging(CompressionMethod):
    """Merge similar visual tokens using cosine assignment to evenly spaced anchors."""

    name = "merging"

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

        merged = torch.stack([_merge_single_sequence(tokens[i], keep) for i in range(batch)], dim=0)
        if squeeze_batch:
            merged = merged.squeeze(0)
        return CompressionResult(
            tokens=merged,
            kept_indices=None,
            metadata={"original_tokens": num_tokens, "kept_tokens": keep, "batch": batch},
        )
