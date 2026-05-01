"""Qwen2.5-VL visual-token pruning adapter.

This module implements the real insertion point for image-token pruning:

    pixel_values -> Qwen visual encoder -> image embeddings -> pruning -> LLM prefill

Qwen2.5-VL normally calls `get_image_features(...)`, concatenates all image
embeddings, and then `masked_scatter`s them into the language-model embedding
sequence. If we simply shorten the image embeddings inside a generic forward
hook, the image placeholder tokens, attention mask, position ids, and KV cache
length still describe the uncompressed sequence.

The adapter below avoids that mismatch by preparing a compressed prefill batch:
it runs the visual encoder once, prunes fixed-ratio image embeddings, removes the
matching unused image placeholder tokens from `input_ids`, rebuilds
`inputs_embeds`, prunes `attention_mask` and `position_ids`, and then calls
`generate()` with a genuinely shorter prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch


@dataclass
class QwenFixedPruningOutput:
    inputs: Dict[str, torch.Tensor]
    stats: Dict[str, Any]


def _get_from_batch(batch: Any, key: str) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(key)
    return getattr(batch, key, None)


def _fixed_keep_indices(num_tokens: int, ratio: float, device: torch.device) -> torch.Tensor:
    keep = max(1, int(round(num_tokens * ratio)))
    keep = min(keep, num_tokens)
    if keep >= num_tokens:
        return torch.arange(num_tokens, device=device, dtype=torch.long)
    return torch.linspace(0, num_tokens - 1, steps=keep, device=device).round().long()


def _contiguous_runs(positions: torch.Tensor) -> List[torch.Tensor]:
    if positions.numel() == 0:
        return []
    if positions.numel() == 1:
        return [positions]
    breaks = torch.where(positions[1:] != positions[:-1] + 1)[0] + 1
    split_points = [int(item) for item in breaks.detach().cpu().tolist()]
    return list(torch.tensor_split(positions, split_points))


def _pad_token_id(model: torch.nn.Module) -> int:
    generation_config = getattr(model, "generation_config", None)
    for owner in (generation_config, getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if owner is None:
            continue
        value = getattr(owner, "pad_token_id", None)
        if value is not None:
            return int(value)
    eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        return int(eos[0])
    if eos is not None:
        return int(eos)
    return 0


def _prune_position_ids(
    position_ids: Optional[torch.Tensor],
    keep_masks: List[torch.Tensor],
    max_len: int,
) -> Optional[torch.Tensor]:
    if position_ids is None:
        return None

    batch_size = len(keep_masks)
    if position_ids.ndim == 3:
        pruned = position_ids.new_zeros((position_ids.shape[0], batch_size, max_len))
        for batch_idx, keep_mask in enumerate(keep_masks):
            selected = position_ids[:, batch_idx, keep_mask]
            pruned[:, batch_idx, : selected.shape[-1]] = selected
        return pruned

    if position_ids.ndim == 2:
        pruned = position_ids.new_zeros((batch_size, max_len))
        for batch_idx, keep_mask in enumerate(keep_masks):
            selected = position_ids[batch_idx, keep_mask]
            pruned[batch_idx, : selected.shape[-1]] = selected
        return pruned

    raise ValueError(f"Unsupported position_ids shape for pruning: {tuple(position_ids.shape)}")


def _update_rope_deltas(
    inner_model: torch.nn.Module,
    position_ids: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
) -> None:
    if position_ids is None:
        inner_model.rope_deltas = None
        return

    rotary_position_ids = position_ids[1:] if position_ids.ndim == 3 and position_ids.shape[0] == 4 else position_ids
    if rotary_position_ids.ndim == 2:
        rotary_position_ids = rotary_position_ids.unsqueeze(0)

    deltas = []
    for batch_idx in range(attention_mask.shape[0]):
        valid = attention_mask[batch_idx].bool()
        seq_len = int(valid.sum().item())
        if seq_len == 0:
            deltas.append(torch.zeros((), dtype=torch.long, device=attention_mask.device))
            continue
        max_pos = rotary_position_ids[:, batch_idx, valid].max()
        deltas.append(max_pos + 1 - seq_len)
    inner_model.rope_deltas = torch.stack(deltas).reshape(-1, 1).to(attention_mask.device)


class Qwen2_5_VLFixedPruningAdapter:
    """Prepare compressed Qwen2.5-VL generation inputs after visual encoding."""

    def __init__(self, model: torch.nn.Module, retention_ratio: float) -> None:
        if not 0.0 < retention_ratio <= 1.0:
            raise ValueError("retention_ratio must be in (0, 1].")
        self.model = model
        self.inner_model = getattr(model, "model", model)
        self.retention_ratio = float(retention_ratio)

    @staticmethod
    def supports(model: torch.nn.Module) -> bool:
        class_name = model.__class__.__name__.lower()
        config_name = getattr(getattr(model, "config", None), "model_type", "")
        looks_like_qwen25 = "qwen2_5" in class_name or "qwen2.5" in class_name or config_name == "qwen2_5_vl"
        return (
            looks_like_qwen25
            and hasattr(model, "get_image_features")
            and hasattr(model, "get_input_embeddings")
            and hasattr(getattr(model, "config", None), "image_token_id")
            and hasattr(getattr(model, "model", None), "compute_3d_position_ids")
        )

    def _compute_full_position_ids(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_grid_thw: torch.Tensor,
        batch: Any,
    ) -> Optional[torch.Tensor]:
        existing = _get_from_batch(batch, "position_ids")
        if existing is not None:
            return existing

        return self.inner_model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=_get_from_batch(batch, "video_grid_thw"),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            second_per_grid_ts=_get_from_batch(batch, "second_per_grid_ts"),
            mm_token_type_ids=_get_from_batch(batch, "mm_token_type_ids"),
        )

    def prepare_inputs_for_generate(self, batch: Any) -> QwenFixedPruningOutput:
        input_ids = _get_from_batch(batch, "input_ids")
        attention_mask = _get_from_batch(batch, "attention_mask")
        pixel_values = _get_from_batch(batch, "pixel_values")
        image_grid_thw = _get_from_batch(batch, "image_grid_thw")

        if input_ids is None:
            raise ValueError("Qwen2.5-VL token pruning requires input_ids.")
        if pixel_values is None or image_grid_thw is None:
            return QwenFixedPruningOutput(
                inputs=dict(batch),
                stats={
                    "compression_applied_internal": False,
                    "internal_compression_reason": "No image pixel_values/image_grid_thw found.",
                },
            )

        embedding = self.model.get_input_embeddings()
        embed_device = embedding.weight.device
        inputs_embeds = embedding(input_ids.to(embed_device)).to(dtype=embedding.weight.dtype)
        inputs_embeds = inputs_embeds.to(device=embed_device)

        full_position_ids = self._compute_full_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            batch=batch,
        )

        vision_outputs = self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        image_feature_chunks = list(vision_outputs.pooler_output)

        image_token_id = int(self.model.config.image_token_id)
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        keep_masks: List[torch.Tensor] = []
        pruned_image_features: List[torch.Tensor] = []
        per_image_original_tokens: List[int] = []
        per_image_kept_tokens: List[int] = []
        image_cursor = 0

        for batch_idx in range(batch_size):
            active = attention_mask[batch_idx].bool()
            is_image = (input_ids[batch_idx] == image_token_id) & active
            row_keep = active & ~is_image

            image_positions = torch.nonzero(is_image, as_tuple=False).flatten()
            for run in _contiguous_runs(image_positions):
                if image_cursor >= len(image_feature_chunks):
                    raise ValueError("More image placeholder runs than visual feature chunks.")

                features = image_feature_chunks[image_cursor]
                if run.numel() != features.shape[0]:
                    raise ValueError(
                        "Image placeholder count does not match visual features for image "
                        f"{image_cursor}: placeholders={run.numel()}, features={features.shape[0]}."
                    )

                local_keep = _fixed_keep_indices(features.shape[0], self.retention_ratio, run.device)
                row_keep[run[local_keep]] = True
                pruned_image_features.append(features.index_select(0, local_keep.to(features.device)))
                per_image_original_tokens.append(int(features.shape[0]))
                per_image_kept_tokens.append(int(local_keep.numel()))
                image_cursor += 1

            keep_masks.append(row_keep)

        if image_cursor != len(image_feature_chunks):
            raise ValueError(
                f"Unused visual feature chunks: consumed={image_cursor}, total={len(image_feature_chunks)}."
            )

        row_lengths = [int(mask.sum().item()) for mask in keep_masks]
        max_len = max(row_lengths)
        pad_id = _pad_token_id(self.model)
        compressed_input_ids = input_ids.new_full((batch_size, max_len), pad_id)
        compressed_attention_mask = attention_mask.new_zeros((batch_size, max_len))

        for batch_idx, keep_mask in enumerate(keep_masks):
            selected_ids = input_ids[batch_idx, keep_mask]
            compressed_input_ids[batch_idx, : selected_ids.numel()] = selected_ids
            compressed_attention_mask[batch_idx, : selected_ids.numel()] = 1

        compressed_input_ids = compressed_input_ids.to(embed_device)
        compressed_attention_mask = compressed_attention_mask.to(embed_device)
        compressed_inputs_embeds = embedding(compressed_input_ids)

        if pruned_image_features:
            flat_image_features = torch.cat(pruned_image_features, dim=0).to(
                device=compressed_inputs_embeds.device,
                dtype=compressed_inputs_embeds.dtype,
            )
            image_mask = compressed_input_ids == image_token_id
            if int(image_mask.sum().item()) != flat_image_features.shape[0]:
                raise ValueError(
                    "Compressed image placeholder count does not match pruned visual features: "
                    f"placeholders={int(image_mask.sum().item())}, features={flat_image_features.shape[0]}."
                )
            expanded_mask = image_mask.unsqueeze(-1).expand_as(compressed_inputs_embeds)
            compressed_inputs_embeds = compressed_inputs_embeds.masked_scatter(expanded_mask, flat_image_features)

        compressed_position_ids = _prune_position_ids(full_position_ids, keep_masks, max_len)
        if compressed_position_ids is not None:
            compressed_position_ids = compressed_position_ids.to(embed_device)
        _update_rope_deltas(self.inner_model, compressed_position_ids, compressed_attention_mask)

        output_inputs: Dict[str, torch.Tensor] = {
            "input_ids": compressed_input_ids,
            "inputs_embeds": compressed_inputs_embeds,
            "attention_mask": compressed_attention_mask,
        }
        if compressed_position_ids is not None:
            output_inputs["position_ids"] = compressed_position_ids

        # Videos are not compressed here, but keeping these keys allows mixed inputs to keep working.
        for key in ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts"):
            value = _get_from_batch(batch, key)
            if value is not None:
                output_inputs[key] = value

        stats = {
            "compression_applied_internal": True,
            "internal_compression_method": "qwen2_5_vl_fixed_ratio_pruning",
            "original_seq_len": int(seq_len),
            "compressed_seq_len": int(max_len),
            "original_visual_tokens": int(sum(per_image_original_tokens)),
            "kept_visual_tokens": int(sum(per_image_kept_tokens)),
            "per_image_original_tokens": per_image_original_tokens,
            "per_image_kept_tokens": per_image_kept_tokens,
            "actual_retention_ratio": (
                float(sum(per_image_kept_tokens) / max(1, sum(per_image_original_tokens)))
            ),
        }
        return QwenFixedPruningOutput(inputs=output_inputs, stats=stats)
