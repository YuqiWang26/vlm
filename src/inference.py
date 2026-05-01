"""Single-example inference entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.model_loader import load_model
from src.utils import load_config, print_environment_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one VLM inference example.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", required=True, nargs="+", help="One or more image paths or URLs.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--method", default="none", choices=["none", "fixed", "importance", "merging"])
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--resolution", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    return parser.parse_args()


def run_single_inference(
    config_path: str,
    images: List[str],
    question: str,
    method: str = "none",
    ratio: float = 1.0,
    resolution: str = "medium",
    max_new_tokens: int | None = None,
    model_id: str | None = None,
) -> dict:
    config = load_config(config_path)
    if model_id:
        config.setdefault("model", {})["model_id"] = model_id
    engine = load_model(config)
    return engine.generate_answer(
        image=images,
        question=question,
        compression_method=method,
        retention_ratio=ratio,
        image_resolution=resolution,
        max_new_tokens=max_new_tokens,
    )


def main() -> None:
    args = parse_args()
    print_environment_info()
    result = run_single_inference(
        config_path=args.config,
        images=args.image,
        question=args.question,
        method=args.method,
        ratio=args.ratio,
        resolution=args.resolution,
        max_new_tokens=args.max_new_tokens,
        model_id=args.model_id,
    )
    print("Answer:", result["generated_answer"])
    print("Latency ms:", f"{result['latency_ms']:.2f}")
    print("Visual tokens:", result["number_of_visual_tokens"])


if __name__ == "__main__":
    main()
