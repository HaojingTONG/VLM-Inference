"""Reusable experiment helpers for the Colab analysis notebook."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import load_dataset

from src.compression import CompressedVLM, FixedRatioPruner, ImportanceBasedPruner, TokenMerger
from src.utils.profiler import InferenceProfiler
from .vqa import (
    score_vqa_prediction,
    validate_vqa_schema,
)


PROJECT_HYPOTHESES = [
    {
        "id": "H1",
        "hypothesis": (
            "Higher visual token counts from larger images increase latency and peak "
            "GPU memory in the baseline VLM pipeline."
        ),
        "evidence": "Baseline sweep over low / medium / high image resolutions.",
    },
    {
        "id": "H2",
        "hypothesis": (
            "Reducing visual token retention lowers prefill latency and memory, "
            "with the largest gains expected at high resolution."
        ),
        "evidence": "Matched baseline-vs-compression sweep over method, retention ratio, and resolution.",
    },
    {
        "id": "H3",
        "hypothesis": (
            "Task quality is retained at moderate compression but degrades under "
            "aggressive token reduction."
        ),
        "evidence": "VQA quality metric selected by validated dataset schema.",
    },
    {
        "id": "H4",
        "hypothesis": (
            "Importance-based pruning or token merging should dominate uniform "
            "fixed-ratio pruning at the same retention ratio if they preserve "
            "more informative visual tokens."
        ),
        "evidence": "Quality-latency tradeoff curves comparing methods at matched retention.",
    },
]


COMPRESSOR_CLASSES = {
    "fixed_ratio": FixedRatioPruner,
    "importance": ImportanceBasedPruner,
    "token_merging": TokenMerger,
}


DEFAULT_RESOLUTIONS = {
    "low": (448, 448),
    "medium": (896, 896),
    "high": (1344, 1344),
}


def make_random_image(height: int, width: int, seed: int) -> Image.Image:
    """Create a deterministic synthetic image for performance stress tests."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def get_image_token_id(model, processor) -> int:
    """Resolve Qwen image-pad token id across transformers versions."""
    for attr in ("image_token_id", "image_token_index"):
        value = getattr(model.config, attr, None)
        if value is not None:
            return int(value)
    token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if token_id is None or token_id == processor.tokenizer.unk_token_id:
        raise ValueError("Could not resolve Qwen image token id.")
    return int(token_id)


def build_qwen_inputs(processor, device: str, image: Image.Image, prompt: str):
    """Build chat-template inputs for one image-question pair."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)


def count_visual_tokens(inputs, image_token_id: int) -> int:
    """Count visual placeholder tokens in one processed input."""
    return int((inputs["input_ids"][0] == image_token_id).sum().item())


def build_compressed_wrapper(model, processor, method: str, retention_ratio: float):
    """Create a baseline or compressed model wrapper for one grid point."""
    if method == "none":
        return CompressedVLM(model, processor, None)
    compressor = COMPRESSOR_CLASSES[method](
        {
            "retention_ratio": retention_ratio,
            "importance_signal": "magnitude",
            "similarity_metric": "cosine",
        }
    )
    return CompressedVLM(model, processor, compressor)


@torch.no_grad()
def decode_answer(processor, model_or_wrapper, inputs, max_new_tokens: int) -> str:
    """Generate greedily and return only newly generated text."""
    is_compressed = (
        isinstance(model_or_wrapper, CompressedVLM)
        and model_or_wrapper.compressor is not None
    )
    if isinstance(model_or_wrapper, CompressedVLM):
        output_ids = model_or_wrapper.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    else:
        output_ids = model_or_wrapper.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if not is_compressed:
        output_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def load_vqa_samples(
    dataset_name: str,
    split: str,
    num_samples: int,
    streaming: bool = False,
):
    """Load a bounded VQA sample list from Hugging Face Datasets."""
    if streaming:
        return list(load_dataset(dataset_name, split=split, streaming=True).take(num_samples))
    split_expr = split if "[" in split else f"{split}[:{num_samples}]"
    return list(load_dataset(dataset_name, split=split_expr))


def load_first_available_vqa(
    candidates: list[dict[str, Any]],
    num_samples: int,
):
    """Try dataset candidates in order and return samples plus schema report."""
    errors = []
    for candidate in candidates:
        name = candidate["name"]
        split = candidate.get("split", "validation")
        streaming = bool(candidate.get("streaming", False))
        try:
            samples = load_vqa_samples(name, split, num_samples, streaming=streaming)
            report = validate_vqa_schema(samples, dataset_name=name)
            return samples, report
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load any VQA dataset candidate:\n" + "\n".join(errors))


def run_baseline_resolution_sweep(
    model,
    processor,
    device: str = "cuda",
    resolutions: dict[str, tuple[int, int]] | None = None,
    num_samples_per_resolution: int = 3,
    max_new_tokens: int = 64,
    prompt: str = "Describe this image in detail.",
    num_warmup: int = 1,
    num_runs: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure baseline full-token generation across input resolutions."""
    resolutions = resolutions or DEFAULT_RESOLUTIONS
    image_token_id = get_image_token_id(model, processor)
    profiler = InferenceProfiler(num_warmup=num_warmup, num_runs=num_runs)
    rows = []

    for resolution, (height, width) in resolutions.items():
        for sample_idx in range(num_samples_per_resolution):
            image = make_random_image(height, width, seed + sample_idx)
            inputs = build_qwen_inputs(processor, device, image, prompt)
            visual_tokens = count_visual_tokens(inputs, image_token_id)

            torch.cuda.empty_cache()
            gc.collect()

            def inference_fn():
                with torch.no_grad():
                    return model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            metrics = profiler.profile(inference_fn)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_tokens = int(output_ids.shape[1] - inputs["input_ids"].shape[1])
            latency_s = max(metrics["latency_ms"] / 1000.0, 1e-9)
            rows.append(
                {
                    "resolution": resolution,
                    "height": height,
                    "width": width,
                    "sample": sample_idx,
                    "visual_tokens": visual_tokens,
                    "latency_ms": metrics["latency_ms"],
                    "tokens_per_sec": generated_tokens / latency_s,
                    "peak_memory_mb": metrics["peak_gpu_memory_mb"],
                    "generated_tokens": generated_tokens,
                }
            )

    per_sample = pd.DataFrame(rows)
    summary = (
        per_sample.groupby(["resolution", "height", "width"], as_index=False)
        .agg(
            visual_tokens=("visual_tokens", "mean"),
            latency_ms=("latency_ms", "mean"),
            latency_std_ms=("latency_ms", "std"),
            tokens_per_sec=("tokens_per_sec", "mean"),
            peak_memory_mb=("peak_memory_mb", "max"),
            generated_tokens=("generated_tokens", "mean"),
        )
    )
    return per_sample, summary


def _time_generate(fn: Callable[[], Any], num_warmup: int, num_runs: int) -> dict[str, float]:
    for _ in range(num_warmup):
        fn()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {
        "latency_ms": float(np.mean(times)),
        "latency_std_ms": float(np.std(times)),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def run_compression_performance_sweep(
    model,
    processor,
    device: str = "cuda",
    resolutions: dict[str, tuple[int, int]] | None = None,
    methods: list[str] | None = None,
    retention_ratios: list[float] | None = None,
    num_samples_per_cell: int = 3,
    max_new_tokens: int = 1,
    prompt: str = "Describe this image in detail.",
    num_warmup: int = 1,
    num_runs: int = 2,
    seed: int = 200,
) -> pd.DataFrame:
    """Measure matched baseline/compression prefill performance."""
    resolutions = resolutions or DEFAULT_RESOLUTIONS
    methods = methods or ["none", "fixed_ratio", "importance", "token_merging"]
    retention_ratios = retention_ratios or [1.0, 0.75, 0.5, 0.25, 0.1]
    image_token_id = get_image_token_id(model, processor)
    rows = []

    for resolution, (height, width) in resolutions.items():
        inputs_list = [
            build_qwen_inputs(
                processor,
                device,
                make_random_image(height, width, seed + sample_idx),
                prompt,
            )
            for sample_idx in range(num_samples_per_cell)
        ]
        visual_tokens_full = count_visual_tokens(inputs_list[0], image_token_id)

        for method in methods:
            ratios = [1.0] if method == "none" else [r for r in retention_ratios if r < 1.0]
            for ratio in ratios:
                wrapper = build_compressed_wrapper(model, processor, method, ratio)
                per_sample = []
                failures = []

                for sample_idx, inputs in enumerate(inputs_list):
                    torch.cuda.empty_cache()
                    gc.collect()

                    def fn():
                        return wrapper.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)

                    try:
                        per_sample.append(_time_generate(fn, num_warmup, num_runs))
                    except Exception as exc:
                        failures.append(f"sample={sample_idx}: {type(exc).__name__}: {exc}")

                if not per_sample:
                    rows.append(
                        {
                            "resolution": resolution,
                            "height": height,
                            "width": width,
                            "method": method,
                            "retention_ratio": ratio,
                            "visual_tokens_full": visual_tokens_full,
                            "visual_tokens_after": None,
                            "latency_ms": None,
                            "latency_std_ms": None,
                            "peak_memory_mb": None,
                            "n_samples": 0,
                            "failures": "; ".join(failures),
                        }
                    )
                    continue

                visual_tokens_after = (
                    visual_tokens_full if method == "none" else int(round(visual_tokens_full * ratio))
                )
                rows.append(
                    {
                        "resolution": resolution,
                        "height": height,
                        "width": width,
                        "method": method,
                        "retention_ratio": ratio,
                        "visual_tokens_full": visual_tokens_full,
                        "visual_tokens_after": visual_tokens_after,
                        "latency_ms": float(np.mean([m["latency_ms"] for m in per_sample])),
                        "latency_std_ms": float(np.mean([m["latency_std_ms"] for m in per_sample])),
                        "peak_memory_mb": float(max(m["peak_memory_mb"] for m in per_sample)),
                        "n_samples": len(per_sample),
                        "failures": "; ".join(failures),
                    }
                )
    return pd.DataFrame(rows)


def run_vqa_quality_sweep(
    model,
    processor,
    samples: list[dict[str, Any]],
    schema_report,
    device: str = "cuda",
    methods: list[str] | None = None,
    retention_ratios: list[float] | None = None,
    max_new_tokens: int = 20,
    prompt_template: str = (
        "Answer the question with a single word or short phrase.\n"
        "Question: {question}\nAnswer:"
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run quality evaluation using the metric selected by schema validation."""
    methods = methods or ["none", "fixed_ratio", "importance", "token_merging"]
    retention_ratios = retention_ratios or [1.0, 0.75, 0.5, 0.25, 0.1]
    question_field = schema_report.question_field or "question"
    image_field = schema_report.image_field or "image"
    metric_name = schema_report.selected_metric

    summary_rows = []
    prediction_rows = []
    for method in methods:
        ratios = [1.0] if method == "none" else [r for r in retention_ratios if r < 1.0]
        for ratio in ratios:
            wrapper = build_compressed_wrapper(model, processor, method, ratio)
            scores = []
            start = time.perf_counter()

            for sample_idx, sample in enumerate(samples):
                question = sample[question_field]
                image = sample[image_field]
                prompt = prompt_template.format(question=question)
                inputs = build_qwen_inputs(processor, device, image, prompt)

                try:
                    prediction = decode_answer(processor, wrapper, inputs, max_new_tokens)
                    score, reference = score_vqa_prediction(prediction, sample, metric_name)
                except Exception as exc:
                    prediction_rows.append(
                        {
                            "sample": sample_idx,
                            "method": method,
                            "retention_ratio": ratio,
                            "question": question,
                            "prediction": None,
                            "reference": None,
                            "score": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                if score is not None:
                    scores.append(score)
                prediction_rows.append(
                    {
                        "sample": sample_idx,
                        "method": method,
                        "retention_ratio": ratio,
                        "question": question,
                        "prediction": prediction,
                        "reference": json.dumps(reference) if isinstance(reference, list) else reference,
                        "score": score,
                        "error": None,
                    }
                )

            torch.cuda.empty_cache()
            gc.collect()
            summary_rows.append(
                {
                    "method": method,
                    "retention_ratio": ratio,
                    "metric": metric_name,
                    "score": float(np.mean(scores)) if scores else None,
                    "n_scored": len(scores),
                    "n_total": len(samples),
                    "eval_seconds": time.perf_counter() - start,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def bootstrap_quality_ci(
    df_predictions: pd.DataFrame,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 123,
) -> pd.DataFrame:
    """Bootstrap confidence intervals over per-question quality scores."""
    valid = df_predictions.dropna(subset=["score"]).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "retention_ratio",
                "mean_score",
                "ci_low",
                "ci_high",
                "n_scored",
            ]
        )

    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci
    rows = []
    for (method, ratio), group in valid.groupby(["method", "retention_ratio"]):
        scores = group["score"].to_numpy(dtype=float)
        boot_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(scores, size=len(scores), replace=True)
            boot_means.append(float(sample.mean()))
        rows.append(
            {
                "method": method,
                "retention_ratio": ratio,
                "mean_score": float(scores.mean()),
                "ci_low": float(np.quantile(boot_means, alpha / 2)),
                "ci_high": float(np.quantile(boot_means, 1 - alpha / 2)),
                "n_scored": len(scores),
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "retention_ratio"]).reset_index(drop=True)


def run_max_batch_size_probe(
    model,
    processor,
    device: str = "cuda",
    resolution_name: str = "medium",
    resolution: tuple[int, int] = (896, 896),
    methods: list[str] | None = None,
    retention_ratios: list[float] | None = None,
    max_batch_size: int = 8,
    max_new_tokens: int = 8,
    prompt: str = "Describe this image briefly.",
    seed: int = 900,
) -> pd.DataFrame:
    """Find max feasible batch size before OOM for selected configurations."""
    methods = methods or ["none", "token_merging"]
    retention_ratios = retention_ratios or [1.0, 0.5]
    height, width = resolution
    image = make_random_image(height, width, seed)
    rows = []

    for method in methods:
        ratios = [1.0] if method == "none" else [r for r in retention_ratios if r < 1.0]
        for ratio in ratios:
            wrapper = build_compressed_wrapper(model, processor, method, ratio)

            def batched_inference(batch_size):
                images = [image] * batch_size
                messages_batch = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    for img in images
                ]
                texts = [
                    processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for messages in messages_batch
                ]
                inputs = processor(
                    text=texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                return wrapper.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            try:
                max_bs = InferenceProfiler.find_max_batch_size(
                    lambda batch_size: batched_inference(batch_size),
                    min_bs=1,
                    max_bs=max_batch_size,
                )
                error = ""
            except Exception as exc:
                max_bs = None
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "resolution": resolution_name,
                    "height": height,
                    "width": width,
                    "method": method,
                    "retention_ratio": ratio,
                    "max_batch_size": max_bs,
                    "probe_max_batch_size": max_batch_size,
                    "max_new_tokens": max_new_tokens,
                    "error": error,
                }
            )
            torch.cuda.empty_cache()
            gc.collect()

    return pd.DataFrame(rows)


def save_results(output_dir: str | Path, **frames: pd.DataFrame) -> dict[str, str]:
    """Save DataFrames to CSV and JSON records for reproducible analysis."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, frame in frames.items():
        csv_path = path / f"{name}.csv"
        json_path = path / f"{name}.json"
        frame.to_csv(csv_path, index=False)
        frame.to_json(json_path, orient="records", indent=2)
        saved[f"{name}_csv"] = str(csv_path)
        saved[f"{name}_json"] = str(json_path)
    return saved
