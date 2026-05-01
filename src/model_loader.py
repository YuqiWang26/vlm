"""Model loading and unified VLM inference interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor

from src.compression import (
    CompressionMethod,
    NoCompression,
    Qwen2_5_VLFixedPruningAdapter,
    create_compression_method,
)
from src.utils import (
    count_new_tokens,
    cuda_timer,
    ensure_image_list,
    estimate_qwen_visual_tokens_from_inputs,
    get_first_model_device,
    move_batch_to_device,
    str_to_torch_dtype,
)


@dataclass
class PreparedBatch:
    inputs: Any
    messages: List[Dict[str, Any]]
    images: List[Image.Image]
    number_of_visual_tokens: Optional[int]
    compression_method: CompressionMethod


def _import_model_class(model_id: str):
    """Select a stable HF model class for Qwen/LLaVA-like VLMs."""

    normalized = model_id.lower()
    if "qwen2.5-vl" in normalized or "qwen2_5" in normalized:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        except ImportError:
            pass

    if "qwen2-vl" in normalized or "qwen2_vl" in normalized:
        try:
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration
        except ImportError:
            pass

    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq


def _from_pretrained_with_retries(model_cls: Any, model_id: str, kwargs: Dict[str, Any]) -> torch.nn.Module:
    """Try common Transformers argument variants across versions."""

    attempts = [kwargs]
    if "torch_dtype" in kwargs:
        dtype_kwargs = dict(kwargs)
        dtype_kwargs["dtype"] = dtype_kwargs.pop("torch_dtype")
        attempts.append(dtype_kwargs)
    if "attn_implementation" in kwargs:
        no_attn_kwargs = dict(kwargs)
        no_attn_kwargs.pop("attn_implementation", None)
        attempts.append(no_attn_kwargs)

    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return model_cls.from_pretrained(model_id, **attempt)
        except TypeError as exc:
            last_error = exc
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"Could not load {model_id}: {last_error}") from last_error


class VLMEngine:
    """Thin wrapper around HF VLMs with compression-aware input preparation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_config = config.get("model", {})
        self.generation_config = config.get("generation", {})
        self.compression_config = config.get("compression", {})
        self.resolution_tokens = config.get(
            "image_resolution_tokens",
            {"low": 256, "medium": 512, "high": 1024},
        )
        self.model_id = ""
        self.model: Optional[torch.nn.Module] = None
        self.processor: Optional[Any] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load()

    def _candidate_model_ids(self) -> List[str]:
        primary = self.model_config.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        fallbacks = self.model_config.get("fallback_model_ids", [])
        return [primary] + [item for item in fallbacks if item != primary]

    def load(self) -> None:
        dtype = str_to_torch_dtype(self.model_config.get("dtype", "bf16"))
        device_map = self.model_config.get("device_map", "auto")
        trust_remote_code = bool(self.model_config.get("trust_remote_code", False))
        low_cpu_mem_usage = bool(self.model_config.get("low_cpu_mem_usage", True))
        attn_implementation = self.model_config.get("attn_implementation", "sdpa")

        last_error: Optional[Exception] = None
        for model_id in self._candidate_model_ids():
            model_cls = _import_model_class(model_id)
            kwargs: Dict[str, Any] = {
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
                "low_cpu_mem_usage": low_cpu_mem_usage,
            }
            if dtype is not None:
                kwargs["torch_dtype"] = dtype
            if attn_implementation:
                kwargs["attn_implementation"] = attn_implementation

            try:
                print(f"Loading model: {model_id}")
                self.model = _from_pretrained_with_retries(model_cls, model_id, kwargs)
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote_code,
                )
                self.model.eval()
                self.model_id = model_id
                self.device = get_first_model_device(self.model)
                print(f"Loaded {model_id} on first device: {self.device}")
                return
            except Exception as exc:
                last_error = exc
                print(f"Failed to load {model_id}: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        raise RuntimeError(f"All model loading attempts failed. Last error: {last_error}")

    def _base_visual_tokens(self, image_resolution: str) -> int:
        if image_resolution not in self.resolution_tokens:
            valid = ", ".join(sorted(self.resolution_tokens))
            raise ValueError(f"Unknown image_resolution={image_resolution}. Valid: {valid}")
        return int(self.resolution_tokens[image_resolution])

    def _build_messages(self, num_images: int, question: str) -> List[Dict[str, Any]]:
        content = [{"type": "image"} for _ in range(num_images)]
        content.append({"type": "text", "text": question})
        return [{"role": "user", "content": content}]

    def _resolve_compression_method(
        self,
        compression_method: CompressionMethod | str | None,
        retention_ratio: float,
    ) -> CompressionMethod:
        apply_proxy = bool(self.compression_config.get("apply_proxy_image_budget", True))
        if compression_method is None:
            return NoCompression(apply_proxy_image_budget=apply_proxy)
        if isinstance(compression_method, str):
            return create_compression_method(
                compression_method,
                retention_ratio=retention_ratio,
                apply_proxy_image_budget=apply_proxy,
            )
        return compression_method

    def _should_use_internal_fixed_pruning(self, method: CompressionMethod) -> bool:
        if self.model is None:
            return False
        return (
            bool(self.compression_config.get("enable_internal_hooks", False))
            and method.name == "fixed"
            and method.retention_ratio < 1.0
            and Qwen2_5_VLFixedPruningAdapter.supports(self.model)
        )

    def prepare_inputs(
        self,
        image: Image.Image | str | Sequence[Image.Image | str],
        question: str,
        compression_method: CompressionMethod | str | None = None,
        retention_ratio: float = 1.0,
        image_resolution: str = "medium",
    ) -> PreparedBatch:
        if self.processor is None or self.model is None:
            raise RuntimeError("Model is not loaded.")

        method = self._resolve_compression_method(compression_method, retention_ratio)

        images = ensure_image_list(image)
        base_tokens = self._base_visual_tokens(image_resolution)
        if self._should_use_internal_fixed_pruning(method):
            # Keep the requested image-resolution budget, then prune real visual
            # embeddings after Qwen's visual encoder. Do not also apply proxy
            # ratio-based resizing, or the ratio would be counted twice.
            prepared_images = NoCompression().compress_images(images, base_tokens)
        else:
            prepared_images = method.compress_images(images, base_tokens)
        messages = self._build_messages(len(prepared_images), question)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=prepared_images,
            padding=True,
            return_tensors="pt",
        )
        inputs = move_batch_to_device(inputs, self.device)

        image_token_id = getattr(getattr(self.model, "config", None), "image_token_id", None)
        num_visual_tokens = estimate_qwen_visual_tokens_from_inputs(inputs, image_token_id)
        return PreparedBatch(
            inputs=inputs,
            messages=messages,
            images=prepared_images,
            number_of_visual_tokens=num_visual_tokens,
            compression_method=method,
        )

    def _prepare_generation_inputs_with_optional_internal_pruning(
        self,
        batch: PreparedBatch,
    ) -> tuple[Any, Dict[str, Any]]:
        method = batch.compression_method
        if not self._should_use_internal_fixed_pruning(method):
            return batch.inputs, {
                "compression_applied_internal": False,
                "original_visual_tokens": batch.number_of_visual_tokens,
                "kept_visual_tokens": batch.number_of_visual_tokens,
            }

        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        adapter = Qwen2_5_VLFixedPruningAdapter(self.model, retention_ratio=method.retention_ratio)
        output = adapter.prepare_inputs_for_generate(batch.inputs)
        return output.inputs, output.stats

    def generate_answer(
        self,
        image: Image.Image | str | Sequence[Image.Image | str],
        question: str,
        compression_method: CompressionMethod | str | None = None,
        retention_ratio: float = 1.0,
        image_resolution: str = "medium",
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded.")

        batch = self.prepare_inputs(
            image=image,
            question=question,
            compression_method=compression_method,
            retention_ratio=retention_ratio,
            image_resolution=image_resolution,
        )
        max_tokens = int(max_new_tokens or self.generation_config.get("max_new_tokens", 64))
        do_sample = bool(self.generation_config.get("do_sample", False))
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(self.generation_config.get("temperature", 0.7))

        compression_stats: Dict[str, Any] = {}
        with torch.inference_mode(), cuda_timer() as timer:
            generation_inputs, compression_stats = self._prepare_generation_inputs_with_optional_internal_pruning(batch)
            output_ids = self.model.generate(**generation_inputs, **generation_kwargs)

        input_ids = generation_inputs["input_ids"] if isinstance(generation_inputs, dict) else generation_inputs.input_ids
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)
        ]
        answers = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generated_tokens = count_new_tokens(input_ids, output_ids)
        latency_ms = timer["latency_ms"]
        throughput = generated_tokens / max(latency_ms / 1000.0, 1e-9)

        return {
            "generated_answer": answers[0].strip() if answers else "",
            "latency_ms": latency_ms,
            "generated_tokens": generated_tokens,
            "throughput_tokens_per_second": throughput,
            "number_of_visual_tokens": compression_stats.get(
                "kept_visual_tokens",
                batch.number_of_visual_tokens,
            ),
            "original_visual_tokens": compression_stats.get(
                "original_visual_tokens",
                batch.number_of_visual_tokens,
            ),
            "kept_visual_tokens": compression_stats.get(
                "kept_visual_tokens",
                batch.number_of_visual_tokens,
            ),
            "compression_applied_internal": compression_stats.get("compression_applied_internal", False),
            "compressed_seq_len": compression_stats.get("compressed_seq_len"),
            "original_seq_len": compression_stats.get("original_seq_len"),
            "prepared_image_sizes": [img.size for img in batch.images],
            "model_id": self.model_id,
        }


def load_model(config: Dict[str, Any]) -> VLMEngine:
    return VLMEngine(config)


def prepare_inputs(
    engine: VLMEngine,
    image: Image.Image | str | Sequence[Image.Image | str],
    question: str,
    compression_method: CompressionMethod | str | None = None,
    retention_ratio: float = 1.0,
    image_resolution: str = "medium",
) -> PreparedBatch:
    return engine.prepare_inputs(
        image=image,
        question=question,
        compression_method=compression_method,
        retention_ratio=retention_ratio,
        image_resolution=image_resolution,
    )


def generate_answer(
    engine: VLMEngine,
    image: Image.Image | str | Sequence[Image.Image | str],
    question: str,
    compression_method: CompressionMethod | str | None = None,
    retention_ratio: float = 1.0,
    image_resolution: str = "medium",
    max_new_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    return engine.generate_answer(
        image=image,
        question=question,
        compression_method=compression_method,
        retention_ratio=retention_ratio,
        image_resolution=image_resolution,
        max_new_tokens=max_new_tokens,
    )
