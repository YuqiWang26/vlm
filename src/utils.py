"""General utilities for Colab-friendly VLM benchmarking."""

from __future__ import annotations

import io
import math
import os
import random
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import yaml
from PIL import Image


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str_to_torch_dtype(dtype_name: str | None) -> torch.dtype | str:
    if dtype_name is None:
        return torch.bfloat16
    normalized = str(dtype_name).lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "auto": "auto",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[normalized]


def get_first_model_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    return batch


def reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_peak_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def get_current_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024**2)


@contextmanager
def cuda_timer() -> Iterable[Dict[str, float]]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result: Dict[str, float] = {}
    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["latency_ms"] = (time.perf_counter() - start) * 1000.0


def print_environment_info() -> None:
    print("Python executable:", os.sys.executable)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / (1024**3)
            print(f"GPU {idx}: {props.name} | {total_gb:.1f} GB")
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"NVML GPU 0: {name} | used {mem.used / (1024**2):.1f} MB")
        except Exception as exc:  # pragma: no cover - optional diagnostics
            print(f"NVML unavailable: {exc}")
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi not found.")


def load_image(image: Image.Image | str | os.PathLike[str]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    image_str = str(image)
    if image_str.startswith("http://") or image_str.startswith("https://"):
        with urllib.request.urlopen(image_str, timeout=20) as response:
            data = response.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(image_str).convert("RGB")


def ensure_image_list(images: Image.Image | str | Sequence[Image.Image | str]) -> List[Image.Image]:
    if isinstance(images, (Image.Image, str, os.PathLike)):
        raw_images: Sequence[Image.Image | str] = [images]
    else:
        raw_images = images
    return [load_image(image) for image in raw_images]


def round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def visual_tokens_to_pixels(num_visual_tokens: int, patch_area: int = 28 * 28) -> int:
    return max(4, int(num_visual_tokens)) * patch_area


def resize_image_to_pixel_budget(
    image: Image.Image,
    max_pixels: int,
    multiple: int = 28,
    min_side: int = 56,
) -> Image.Image:
    """Resize while preserving aspect ratio and roughly matching Qwen's token budget."""

    image = image.convert("RGB")
    width, height = image.size
    current_pixels = width * height
    if current_pixels <= max_pixels and width % multiple == 0 and height % multiple == 0:
        return image

    scale = math.sqrt(max_pixels / max(1, current_pixels))
    new_width = max(min_side, round_to_multiple(int(width * scale), multiple))
    new_height = max(min_side, round_to_multiple(int(height * scale), multiple))
    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)


def estimate_qwen_visual_tokens_from_inputs(inputs: Any, image_token_id: int | None = None) -> int | None:
    """Estimate visual token count from image pad tokens or image_grid_thw."""

    if image_token_id is not None and hasattr(inputs, "input_ids"):
        input_ids = inputs.input_ids
        return int((input_ids == image_token_id).sum().item())

    if hasattr(inputs, "image_grid_thw"):
        grid = inputs.image_grid_thw
        if grid is not None and torch.is_tensor(grid):
            return int(torch.prod(grid, dim=-1).sum().item())

    if isinstance(inputs, dict):
        input_ids = inputs.get("input_ids")
        if image_token_id is not None and torch.is_tensor(input_ids):
            return int((input_ids == image_token_id).sum().item())
        grid = inputs.get("image_grid_thw")
        if torch.is_tensor(grid):
            return int(torch.prod(grid, dim=-1).sum().item())

    return None


def count_new_tokens(input_ids: torch.Tensor, output_ids: torch.Tensor) -> int:
    if output_ids.ndim == 1:
        output_ids = output_ids.unsqueeze(0)
    return int(max(0, output_ids.shape[-1] - input_ids.shape[-1]))


def maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
