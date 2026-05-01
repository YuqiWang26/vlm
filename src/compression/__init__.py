"""Compression methods and adapters."""

from .base import CompressionMethod, NoCompression, create_compression_method
from .fixed_ratio_pruning import FixedRatioPruning
from .importance_pruning import ImportanceBasedPruning
from .qwen2_5_vl_fixed import Qwen2_5_VLFixedPruningAdapter
from .token_merging import TokenMerging, compress_visual_tokens

__all__ = [
    "CompressionMethod",
    "NoCompression",
    "FixedRatioPruning",
    "ImportanceBasedPruning",
    "Qwen2_5_VLFixedPruningAdapter",
    "TokenMerging",
    "compress_visual_tokens",
    "create_compression_method",
]
