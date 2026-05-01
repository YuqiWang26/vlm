"""Base classes and hook scaffolding for visual token compression."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image

from src.utils import resize_image_to_pixel_budget, visual_tokens_to_pixels


@dataclass
class CompressionResult:
    tokens: torch.Tensor
    kept_indices: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class CompressionMethod:
    """Base class for inference-time visual-token compressors."""

    name = "base"

    def __init__(
        self,
        retention_ratio: float = 1.0,
        apply_proxy_image_budget: bool = True,
    ) -> None:
        if not 0.0 < retention_ratio <= 1.0:
            raise ValueError("retention_ratio must be in (0, 1].")
        self.retention_ratio = float(retention_ratio)
        self.apply_proxy_image_budget = apply_proxy_image_budget

    def compress_visual_tokens(self, tokens: torch.Tensor) -> CompressionResult:
        """Compress [B, N, D] or [N, D] visual tokens.

        This method is fully runnable and unit-testable. For Qwen2.5-VL, wiring it
        into the actual forward pass requires adjusting image pad tokens, image_grid_thw,
        attention masks, and MRoPE position ids. See VisionTokenHookAdapter below.
        """

        return CompressionResult(tokens=tokens)

    def compress_images(
        self,
        images: List[Image.Image],
        base_visual_tokens: int,
    ) -> List[Image.Image]:
        """Runnable proxy: reduce image pixel budget before processor tokenization."""

        if not self.apply_proxy_image_budget:
            return images
        target_tokens = max(4, int(round(base_visual_tokens * self.retention_ratio)))
        target_pixels = visual_tokens_to_pixels(target_tokens)
        return [resize_image_to_pixel_budget(image, target_pixels) for image in images]

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "retention_ratio": self.retention_ratio,
            "apply_proxy_image_budget": self.apply_proxy_image_budget,
        }


class NoCompression(CompressionMethod):
    name = "none"

    def __init__(self, retention_ratio: float = 1.0, apply_proxy_image_budget: bool = True) -> None:
        super().__init__(retention_ratio=1.0, apply_proxy_image_budget=apply_proxy_image_budget)

    def compress_images(self, images: List[Image.Image], base_visual_tokens: int) -> List[Image.Image]:
        target_pixels = visual_tokens_to_pixels(base_visual_tokens)
        return [resize_image_to_pixel_budget(image, target_pixels) for image in images]


class VisionTokenHookAdapter:
    """Experimental low-level adapter for future internal VLM token hooks.

    Qwen2.5-VL fixed-ratio pruning is implemented in qwen2_5_vl_fixed.py with a
    compressed prefill adapter. Keep this generic hook disabled for pruning that
    changes sequence length: a blind PyTorch hook cannot update input_ids,
    attention_mask, position_ids, and generation KV-cache bookkeeping together.

    The register_forward_hook path below is intentionally disabled by default. It is
    useful for experiments where the compressor preserves token count, but pruning to
    a shorter sequence needs a model-aware adapter.
    """

    candidate_module_names = (
        "visual",
        "vision_tower",
        "model.visual",
        "model.vision_tower",
    )

    def __init__(self, method: CompressionMethod) -> None:
        self.method = method
        self.handle: Optional[Any] = None

    def locate_visual_module(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        modules = dict(model.named_modules())
        for name in self.candidate_module_names:
            if name in modules:
                return modules[name]
        return None

    def register(self, model: torch.nn.Module) -> Optional[Any]:
        target = self.locate_visual_module(model)
        if target is None:
            return None

        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
            if torch.is_tensor(output):
                return self.method.compress_visual_tokens(output).tokens
            if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                updated = list(output)
                updated[0] = self.method.compress_visual_tokens(output[0]).tokens
                return tuple(updated)
            return output

        self.handle = target.register_forward_hook(hook)
        return self.handle

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def clone_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return copy.deepcopy(messages)


def create_compression_method(
    method_name: str,
    retention_ratio: float = 1.0,
    apply_proxy_image_budget: bool = True,
) -> CompressionMethod:
    normalized = method_name.lower()
    if normalized in {"none", "no_compression", "baseline"}:
        return NoCompression(apply_proxy_image_budget=apply_proxy_image_budget)
    if normalized in {"fixed", "fixed_ratio", "fixed_ratio_pruning"}:
        from .fixed_ratio_pruning import FixedRatioPruning

        return FixedRatioPruning(retention_ratio, apply_proxy_image_budget)
    if normalized in {"importance", "importance_based", "importance_based_pruning"}:
        from .importance_pruning import ImportanceBasedPruning

        return ImportanceBasedPruning(retention_ratio, apply_proxy_image_budget)
    if normalized in {"merging", "token_merging", "merge"}:
        from .token_merging import TokenMerging

        return TokenMerging(retention_ratio, apply_proxy_image_budget)
    raise ValueError(f"Unknown compression method: {method_name}")
