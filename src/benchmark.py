"""Benchmark runner for visual-token compression experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from src.compression import create_compression_method
from src.metrics import compute_quality_score, summarize_results
from src.model_loader import VLMEngine
from src.utils import ensure_dir, get_peak_gpu_memory_mb, reset_peak_gpu_memory


@dataclass
class ToySample:
    sample_id: str
    image: Image.Image
    question: str
    reference_answer: str
    keywords: List[str]


def _draw_centered_text(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int]) -> None:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text(xy, text, fill=(20, 20, 20), font=font)


def create_toy_dataset(image_size: int = 896) -> List[ToySample]:
    """Create synthetic images so Colab smoke tests do not need dataset downloads."""

    samples: List[ToySample] = []

    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([180, 180, 720, 720], fill=(220, 40, 40))
    samples.append(
        ToySample("red_square", img, "What color and shape is shown?", "red square", ["red", "square"])
    )

    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([180, 180, 720, 720], fill=(40, 90, 230))
    samples.append(
        ToySample("blue_circle", img, "What color and shape is shown?", "blue circle", ["blue", "circle"])
    )

    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)
    draw.polygon([(448, 120), (120, 760), (776, 760)], fill=(245, 205, 35))
    samples.append(
        ToySample("yellow_triangle", img, "What color and shape is shown?", "yellow triangle", ["yellow", "triangle"])
    )

    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([130, 520, 260, 760], fill=(55, 170, 90))
    draw.rectangle([360, 380, 490, 760], fill=(55, 170, 90))
    draw.rectangle([590, 230, 720, 760], fill=(55, 170, 90))
    draw.line([100, 760, 780, 760], fill=(0, 0, 0), width=5)
    samples.append(
        ToySample("green_bars", img, "What simple chart is shown?", "green bar chart", ["green", "bar", "chart"])
    )

    img = Image.new("RGB", (image_size, image_size), (235, 245, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([180, 260, 720, 630], outline=(30, 30, 30), width=8)
    _draw_centered_text(draw, "CAT", (390, 430))
    samples.append(
        ToySample("cat_text", img, "What word is written in the image?", "CAT", ["cat"])
    )

    return samples


def build_multi_image_case(
    dataset: Sequence[ToySample],
    start_index: int,
    num_images: int,
) -> Dict[str, Any]:
    selected = [dataset[(start_index + offset) % len(dataset)] for offset in range(num_images)]
    images = [sample.image for sample in selected]
    if num_images == 1:
        question = selected[0].question + " Answer briefly."
    else:
        question = "List the main colors, shapes, or words visible across these images. Answer briefly."
    keywords: List[str] = []
    for sample in selected:
        keywords.extend(sample.keywords)
    reference_answer = " ".join(keywords)
    return {
        "sample_id": "+".join(sample.sample_id for sample in selected),
        "images": images,
        "question": question,
        "reference_answer": reference_answer,
        "keywords": keywords,
    }


class BenchmarkRunner:
    def __init__(self, engine: VLMEngine, config: Dict[str, Any]) -> None:
        self.engine = engine
        self.config = config
        self.benchmark_config = config.get("benchmark", {})
        self.compression_config = config.get("compression", {})
        self.quality_config = config.get("quality", {})

    def _total_runs(
        self,
        methods: Sequence[str],
        ratios: Sequence[float],
        resolutions: Sequence[str],
        num_images_values: Sequence[int],
        max_samples: int,
    ) -> int:
        method_ratio_count = 0
        for method in methods:
            method_ratio_count += 1 if method == "none" else len(ratios)
        return method_ratio_count * len(resolutions) * len(num_images_values) * max_samples

    def run(
        self,
        methods: Sequence[str] | None = None,
        ratios: Sequence[float] | None = None,
        resolutions: Sequence[str] | None = None,
        num_images_values: Sequence[int] | None = None,
        max_samples: int | None = None,
        output_csv: str | Path | None = None,
        save_every: int = 1,
    ) -> pd.DataFrame:
        methods = list(methods or self.benchmark_config.get("methods", ["none", "fixed", "importance", "merging"]))
        ratios = [float(x) for x in (ratios or self.benchmark_config.get("retention_ratios", [1.0, 0.5, 0.25]))]
        resolutions = list(resolutions or self.benchmark_config.get("image_resolutions", ["medium"]))
        num_images_values = [int(x) for x in (num_images_values or self.benchmark_config.get("num_images", [1]))]
        max_samples = int(max_samples or self.benchmark_config.get("max_samples", 3))
        output_csv = Path(output_csv or self.benchmark_config.get("output_csv", "results/benchmark_results.csv"))
        ensure_dir(output_csv.parent)

        dataset = create_toy_dataset()
        max_samples = min(max_samples, len(dataset))
        records: List[Dict[str, Any]] = []
        total = self._total_runs(methods, ratios, resolutions, num_images_values, max_samples)
        metric = self.quality_config.get("metric", "keyword_match")
        apply_proxy = bool(self.compression_config.get("apply_proxy_image_budget", True))
        max_new_tokens = int(self.config.get("generation", {}).get("max_new_tokens", 64))

        progress = tqdm(total=total, desc="Benchmark")
        for method_name in methods:
            method_ratios = [1.0] if method_name == "none" else ratios
            for ratio in method_ratios:
                compression_method = create_compression_method(
                    method_name,
                    retention_ratio=ratio,
                    apply_proxy_image_budget=apply_proxy,
                )
                for image_resolution in resolutions:
                    for num_images in num_images_values:
                        for sample_idx in range(max_samples):
                            case = build_multi_image_case(dataset, sample_idx, num_images)
                            record: Dict[str, Any] = {
                                "model_id": self.engine.model_id,
                                "compression_method": compression_method.name,
                                "retention_ratio": compression_method.retention_ratio,
                                "input_resolution": image_resolution,
                                "num_images": num_images,
                                "max_new_tokens": max_new_tokens,
                                "sample_id": case["sample_id"],
                                "question": case["question"],
                                "reference_answer": case["reference_answer"],
                                "keywords": ",".join(case["keywords"]),
                                "success": False,
                                "oom": False,
                                "error": "",
                                "proxy_image_budget": apply_proxy,
                            }
                            try:
                                reset_peak_gpu_memory()
                                result = self.engine.generate_answer(
                                    image=case["images"],
                                    question=case["question"],
                                    compression_method=compression_method,
                                    retention_ratio=ratio,
                                    image_resolution=image_resolution,
                                    max_new_tokens=max_new_tokens,
                                )
                                peak_memory = get_peak_gpu_memory_mb()
                                quality = compute_quality_score(
                                    result["generated_answer"],
                                    case["reference_answer"],
                                    keywords=case["keywords"],
                                    metric=metric,
                                )
                                record.update(result)
                                record.update(
                                    {
                                        "peak_gpu_memory_mb": peak_memory,
                                        "quality_score": quality,
                                        "success": True,
                                    }
                                )
                            except torch.cuda.OutOfMemoryError as exc:
                                record.update({"oom": True, "error": str(exc)})
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except RuntimeError as exc:
                                message = str(exc)
                                record.update({"oom": "out of memory" in message.lower(), "error": message})
                                if record["oom"] and torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception as exc:
                                record.update({"error": repr(exc)})

                            records.append(record)
                            if save_every > 0 and len(records) % save_every == 0:
                                pd.DataFrame(records).to_csv(output_csv, index=False)
                            progress.update(1)

        progress.close()
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)

        summary_csv = self.benchmark_config.get("summary_csv")
        if summary_csv:
            summarize_results(df, output_csv=summary_csv)
        return df


def run_benchmark(engine: VLMEngine, config: Dict[str, Any], **kwargs: Any) -> pd.DataFrame:
    return BenchmarkRunner(engine, config).run(**kwargs)
