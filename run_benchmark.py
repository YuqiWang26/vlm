"""CLI for running the VLM visual-token compression benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Sequence, TypeVar

from src.benchmark import run_benchmark
from src.metrics import print_summary_table, summarize_results
from src.model_loader import load_model
from src.utils import load_config, print_environment_info, set_seed

T = TypeVar("T")


def _parse_csv_list(value: str | None, cast: Callable[[str], T]) -> List[T] | None:
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VLM inference with visual token compression.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-id", default=None, help="Override model.model_id.")
    parser.add_argument("--dtype", default=None, help="Override model dtype: bf16, fp16, float32, auto.")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
        help="Override attention backend. Use eager if Colab/PyTorch SDPA kernels fail.",
    )
    parser.add_argument("--methods", default=None, help="Comma list: none,fixed,importance,merging")
    parser.add_argument("--ratios", default=None, help="Comma list, e.g. 1.0,0.75,0.5,0.25,0.1")
    parser.add_argument("--resolutions", default=None, help="Comma list: low,medium,high")
    parser.add_argument("--num-images", default=None, help="Comma list: 1,2,4")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--quick", action="store_true", help="Small smoke test.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.model_id:
        config.setdefault("model", {})["model_id"] = args.model_id
    if args.dtype:
        config.setdefault("model", {})["dtype"] = args.dtype
    if args.attn_implementation:
        config.setdefault("model", {})["attn_implementation"] = args.attn_implementation
    if args.max_new_tokens:
        config.setdefault("generation", {})["max_new_tokens"] = args.max_new_tokens
    if args.output:
        config.setdefault("benchmark", {})["output_csv"] = args.output
    if args.summary_output:
        config.setdefault("benchmark", {})["summary_csv"] = args.summary_output

    methods = _parse_csv_list(args.methods, str)
    ratios = _parse_csv_list(args.ratios, float)
    resolutions = _parse_csv_list(args.resolutions, str)
    num_images_values = _parse_csv_list(args.num_images, int)
    max_samples = args.samples

    if args.quick:
        methods = methods or ["none", "fixed", "importance", "merging"]
        ratios = ratios or [1.0, 0.5]
        resolutions = resolutions or ["medium"]
        num_images_values = num_images_values or [1]
        max_samples = max_samples or 1
        config.setdefault("generation", {})["max_new_tokens"] = min(
            int(config.get("generation", {}).get("max_new_tokens", 64)),
            32,
        )

    set_seed(int(config.get("project", {}).get("seed", 42)))
    print_environment_info()
    engine = load_model(config)

    df = run_benchmark(
        engine,
        config,
        methods=methods,
        ratios=ratios,
        resolutions=resolutions,
        num_images_values=num_images_values,
        max_samples=max_samples,
        output_csv=config.get("benchmark", {}).get("output_csv", "results/benchmark_results.csv"),
    )

    output_csv = Path(config.get("benchmark", {}).get("output_csv", "results/benchmark_results.csv"))
    summary_csv = Path(config.get("benchmark", {}).get("summary_csv", "results/summary_results.csv"))
    summary = summarize_results(df, output_csv=summary_csv)
    print("\nSummary:")
    print_summary_table(summary)
    print(f"\nSaved raw results to: {output_csv}")
    print(f"Saved summary to: {summary_csv}")

    if not args.no_plots:
        from src.plot_results import plot_all

        paths = list(plot_all(output_csv, output_csv.parent))
        print("Saved plots:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
