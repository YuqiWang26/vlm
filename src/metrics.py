"""Metrics and result summarization for VLM compression experiments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_answer(text: str) -> str:
    """Normalize and remove spaces for OCR-like codes such as K7P4 or RX-19."""

    return normalize_answer(text).replace(" ", "")


def _contains_keyword(prediction: str, keyword: str) -> bool:
    normalized_prediction = normalize_answer(prediction)
    normalized_keyword = normalize_answer(keyword)
    if not normalized_keyword:
        return False

    prediction_tokens = normalized_prediction.split()
    keyword_tokens = normalized_keyword.split()

    # Short numeric answers should match whole tokens. Otherwise "7" would match
    # "17", which makes counting and table lookup tasks look falsely correct.
    if len(keyword_tokens) == 1 and keyword_tokens[0].isdigit() and len(keyword_tokens[0]) <= 2:
        return keyword_tokens[0] in prediction_tokens

    if len(keyword_tokens) == 1 and keyword_tokens[0] in prediction_tokens:
        return True
    if f" {normalized_keyword} " in f" {normalized_prediction} ":
        return True

    compact_prediction = compact_answer(prediction)
    compact_keyword = compact_answer(keyword)
    return bool(compact_keyword and compact_keyword in compact_prediction)


def exact_match(prediction: str, reference: str) -> float:
    return float(
        normalize_answer(prediction) == normalize_answer(reference)
        or compact_answer(prediction) == compact_answer(reference)
    )


def keyword_match(
    prediction: str,
    reference: str = "",
    keywords: Optional[Iterable[str]] = None,
) -> float:
    if keywords is None:
        keywords = [tok for tok in normalize_answer(reference).split() if len(tok) > 2]
    keyword_list = [normalize_answer(keyword) for keyword in keywords if normalize_answer(keyword)]
    if not keyword_list:
        return 0.0
    hits = sum(1 for keyword in keyword_list if _contains_keyword(prediction, keyword))
    return hits / len(keyword_list)


def all_keywords_match(
    prediction: str,
    reference: str = "",
    keywords: Optional[Iterable[str]] = None,
) -> float:
    """Strict VQA-style score for synthetic stress tests.

    A sample with multiple target values only receives 1.0 if every required
    value appears in the model answer. This makes partial OCR/counting failures
    visible in the aggregate accuracy curve.
    """

    if keywords is None:
        keywords = [tok for tok in normalize_answer(reference).split() if len(tok) > 2]
    keyword_list = [normalize_answer(keyword) for keyword in keywords if normalize_answer(keyword)]
    if not keyword_list:
        return exact_match(prediction, reference)
    return float(all(_contains_keyword(prediction, keyword) for keyword in keyword_list))


def vqa_strict_match(
    prediction: str,
    reference: str,
    keywords: Optional[Iterable[str]] = None,
) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    if normalized_reference and f" {normalized_reference} " in f" {normalized_prediction} ":
        return 1.0
    if compact_answer(reference) and compact_answer(reference) in compact_answer(prediction):
        return 1.0
    return all_keywords_match(prediction, reference, keywords=keywords)


def compute_quality_score(
    prediction: str,
    reference: str,
    keywords: Optional[Iterable[str]] = None,
    metric: str = "keyword_match",
) -> float:
    if metric == "exact_match":
        return exact_match(prediction, reference)
    if metric == "keyword_match":
        return keyword_match(prediction, reference, keywords=keywords)
    if metric in {"all_keywords", "all_keywords_match"}:
        return all_keywords_match(prediction, reference, keywords=keywords)
    if metric in {"vqa_strict", "strict_vqa", "strict"}:
        return vqa_strict_match(prediction, reference, keywords=keywords)
    raise ValueError(f"Unsupported quality metric: {metric}")


def summarize_results(
    results: pd.DataFrame | str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    if not isinstance(results, pd.DataFrame):
        results = pd.read_csv(results)

    df = results.copy()
    if "success" in df.columns:
        df_success = df[df["success"].astype(bool)]
    else:
        df_success = df

    if df_success.empty:
        summary = pd.DataFrame(
            columns=[
                "compression_method",
                "retention_ratio",
                "avg_latency",
                "avg_memory",
                "avg_quality_score",
                "avg_throughput",
                "avg_visual_tokens",
                "num_runs",
                "success_rate",
            ]
        )
    else:
        grouped = df_success.groupby(["compression_method", "retention_ratio"], dropna=False)
        summary = grouped.agg(
            avg_latency=("latency_ms", "mean"),
            avg_memory=("peak_gpu_memory_mb", "mean"),
            avg_quality_score=("quality_score", "mean"),
            avg_throughput=("throughput_tokens_per_second", "mean"),
            avg_visual_tokens=("number_of_visual_tokens", "mean"),
            num_runs=("success", "count"),
        ).reset_index()

        if "success" in df.columns:
            success_rate = (
                df.groupby(["compression_method", "retention_ratio"], dropna=False)["success"]
                .mean()
                .reset_index(name="success_rate")
            )
            summary = summary.merge(success_rate, on=["compression_method", "retention_ratio"], how="left")
        else:
            summary["success_rate"] = 1.0

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_csv, index=False)
    return summary


def print_summary_table(summary: pd.DataFrame, max_rows: int = 50) -> None:
    if summary.empty:
        print("No successful runs to summarize.")
        return
    display_cols: List[str] = [
        "compression_method",
        "retention_ratio",
        "avg_latency",
        "avg_memory",
        "avg_quality_score",
        "avg_throughput",
        "success_rate",
    ]
    existing_cols = [col for col in display_cols if col in summary.columns]
    print(summary[existing_cols].head(max_rows).to_string(index=False))
