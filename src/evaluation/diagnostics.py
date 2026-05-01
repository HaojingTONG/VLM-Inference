"""Diagnostics for VLM visual-token compression performance.

These helpers are intentionally separate from the presentation notebook path.
They answer debugging questions:
  * Does compression actually shorten tensors entering the LLM?
  * Which stages consume wall-clock time?
  * Do extreme retention ratios affect latency enough to matter?
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import gc
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.compression import CompressedVLM
from src.evaluation.experiments import (
    build_compressed_wrapper,
    make_random_image,
)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def _timed(stage_times: dict[str, float], name: str):
    _sync()
    start = time.perf_counter()
    yield
    _sync()
    stage_times[name] += (time.perf_counter() - start) * 1000.0


def _shape(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [_shape(v) for v in value[:4]]
    if isinstance(value, dict):
        return {k: _shape(v) for k, v in list(value.items())[:8]}
    return None


def _first_tensor_shape(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor_shape(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor_shape(item)
            if found is not None:
                return found
    return None


class ModuleShapeRecorder:
    """Forward-hook recorder for LLM layers and attention modules."""

    def __init__(self, model, max_events_per_module: int = 4):
        self.model = model
        self.max_events_per_module = max_events_per_module
        self.handles = []
        self.events = []
        self._counts = defaultdict(int)

    def __enter__(self):
        for name, module, role in self._target_modules():
            handle = module.register_forward_hook(self._make_hook(name, role))
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, name: str, role: str):
        def hook(module, args, output):
            count = self._counts[name]
            if count >= self.max_events_per_module:
                self._counts[name] += 1
                return
            self._counts[name] += 1
            self.events.append(
                {
                    "module": name,
                    "role": role,
                    "call_index": count,
                    "class": module.__class__.__name__,
                    "input_shapes": _shape(args),
                    "output_shapes": _shape(output),
                    "first_input_tensor_shape": _first_tensor_shape(args),
                    "first_output_tensor_shape": _first_tensor_shape(output),
                }
            )

        return hook

    def _target_modules(self):
        targets = []
        llm_layers = self._find_llm_layers()
        if llm_layers is not None and len(llm_layers) > 0:
            layer_indices = sorted({0, len(llm_layers) // 2, len(llm_layers) - 1})
            for idx in layer_indices:
                layer = llm_layers[idx]
                targets.append((f"llm.layers.{idx}", layer, "llm_layer"))
                for child_name, child in layer.named_modules():
                    lname = child_name.lower()
                    if child_name and (
                        lname.endswith("self_attn")
                        or lname.endswith("attention")
                        or "self_attn" in lname
                    ):
                        targets.append(
                            (f"llm.layers.{idx}.{child_name}", child, "attention")
                        )
                        break
            return targets

        for name, module in self.model.named_modules():
            lname = name.lower()
            if lname.endswith("self_attn") or lname.endswith("attention"):
                targets.append((name, module, "attention"))
                if len(targets) >= 6:
                    break
        return targets

    def _find_llm_layers(self):
        direct_paths = [
            ("model", "layers"),
            ("model", "language_model", "layers"),
            ("language_model", "layers"),
            ("model", "model", "layers"),
        ]
        for path in direct_paths:
            obj = self.model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
                return obj

        candidates = []
        for _, module in self.model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
                first_name = module[0].__class__.__name__.lower()
                if "decoder" in first_name or "layer" in first_name:
                    candidates.append(module)
        if not candidates:
            return None
        return max(candidates, key=len)


def build_qwen_inputs_timed(processor, device: str, image: Image.Image, prompt: str):
    """Build inputs while returning preprocessing/tok timings."""
    times = defaultdict(float)
    with _timed(times, "image_preprocessing_ms"):
        image = image.convert("RGB") if image.mode != "RGB" else image.copy()

    with _timed(times, "chat_template_ms"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    with _timed(times, "processor_tokenization_image_ms"):
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

    with _timed(times, "device_transfer_ms"):
        inputs = inputs.to(device)

    return inputs, dict(times)


def sequence_length_diagnostics(
    model,
    processor,
    image: Image.Image | None = None,
    prompt: str = "Describe this image in detail.",
    device: str = "cuda",
    methods: list[str] | None = None,
    retention_ratios: list[float] | None = None,
    max_new_tokens: int = 1,
    resolution: tuple[int, int] = (896, 896),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one diagnostic generation per method and capture true module shapes."""
    methods = methods or ["none", "fixed_ratio", "importance", "token_merging"]
    retention_ratios = retention_ratios or [1.0, 0.5, 0.25, 0.1]
    image = image or make_random_image(resolution[0], resolution[1], seed=1234)

    rows = []
    event_rows = []
    for method in methods:
        ratios = [1.0] if method == "none" else [r for r in retention_ratios if r < 1.0]
        for ratio in ratios:
            inputs, build_times = build_qwen_inputs_timed(processor, device, image, prompt)
            wrapper = build_compressed_wrapper(model, processor, method, ratio)
            wrapper.enable_debug = True
            with torch.no_grad(), ModuleShapeRecorder(model) as recorder:
                output = wrapper.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                _sync()

            debug = dict(getattr(wrapper, "last_debug", {}))
            debug.update(
                {
                    "method": method,
                    "retention_ratio": ratio,
                    "output_shape": _shape(output),
                    "build_time_ms": sum(build_times.values()),
                    **build_times,
                }
            )

            first_layer_event = next(
                (event for event in recorder.events if event["role"] == "llm_layer"),
                None,
            )
            first_attn_event = next(
                (event for event in recorder.events if event["role"] == "attention"),
                None,
            )
            if first_layer_event:
                debug["first_llm_layer_input_tensor_shape"] = first_layer_event[
                    "first_input_tensor_shape"
                ]
                debug["first_llm_layer_output_tensor_shape"] = first_layer_event[
                    "first_output_tensor_shape"
                ]
            if first_attn_event:
                debug["first_attention_input_tensor_shape"] = first_attn_event[
                    "first_input_tensor_shape"
                ]
                debug["first_attention_output_tensor_shape"] = first_attn_event[
                    "first_output_tensor_shape"
                ]
            rows.append(debug)

            for event in recorder.events:
                event_rows.append(
                    {
                        "method": method,
                        "retention_ratio": ratio,
                        **event,
                    }
                )
            torch.cuda.empty_cache()
            gc.collect()

    return pd.DataFrame(rows), pd.DataFrame(event_rows)


def stage_timing_breakdown(
    model,
    processor,
    image: Image.Image | None = None,
    prompt: str = "Describe this image in detail.",
    device: str = "cuda",
    methods: list[str] | None = None,
    retention_ratios: list[float] | None = None,
    max_new_tokens: int = 16,
    num_warmup: int = 1,
    num_runs: int = 3,
    resolution: tuple[int, int] = (896, 896),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile preprocessing, vision, compression, prefill, decode, postprocess."""
    methods = methods or ["none", "fixed_ratio", "importance", "token_merging"]
    retention_ratios = retention_ratios or [1.0, 0.5, 0.25, 0.1]
    image = image or make_random_image(resolution[0], resolution[1], seed=5678)
    rows = []

    configs = []
    for method in methods:
        ratios = [1.0] if method == "none" else [r for r in retention_ratios if r < 1.0]
        configs.extend((method, ratio) for ratio in ratios)

    for method, ratio in configs:
        wrapper = build_compressed_wrapper(model, processor, method, ratio)

        # Warmup the full path to compile kernels / populate caches.
        for _ in range(num_warmup):
            inputs, _ = build_qwen_inputs_timed(processor, device, image, prompt)
            with torch.no_grad():
                wrapper.generate(inputs, max_new_tokens=1, do_sample=False)
            _sync()

        for run_idx in range(num_runs):
            stage = defaultdict(float)
            inputs = None
            with _timed(stage, "input_build_total_ms"):
                inputs, build_times = build_qwen_inputs_timed(processor, device, image, prompt)
            stage.update(build_times)

            with torch.no_grad():
                if method == "none":
                    with _timed(stage, "prefill_generate_1_token_ms"):
                        prefill_out = wrapper.generate(
                            inputs,
                            max_new_tokens=1,
                            do_sample=False,
                        )
                    full_out = None
                    if max_new_tokens > 1:
                        with _timed(stage, "full_generate_ms"):
                            full_out = wrapper.generate(
                                inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                            )
                    else:
                        full_out = prefill_out
                    with _timed(stage, "postprocessing_ms"):
                        processor.batch_decode(full_out, skip_special_tokens=True)
                    debug = dict(getattr(wrapper, "last_debug", {}))
                else:
                    with _timed(stage, "vision_encoder_forward_ms"):
                        per_image = wrapper._compute_image_embeds(
                            inputs["pixel_values"], inputs["image_grid_thw"]
                        )
                    with _timed(stage, "compression_scoring_merging_ms"):
                        compressed_per_image = [
                            wrapper.compressor.compress(emb.unsqueeze(0)).squeeze(0)
                            for emb in per_image
                        ]
                    target_dtype = model.get_input_embeddings().weight.dtype
                    with _timed(stage, "dtype_device_cast_ms"):
                        compressed_per_image = [
                            emb.to(model.device, dtype=target_dtype)
                            for emb in compressed_per_image
                        ]
                    merge = wrapper.spatial_merge_size
                    old_lens = (
                        inputs["image_grid_thw"].prod(dim=-1) // (merge * merge)
                    ).tolist()
                    new_lens = [emb.shape[0] for emb in compressed_per_image]
                    with _timed(stage, "placeholder_rewrite_ms"):
                        new_input_ids, new_attention_mask = wrapper._rewrite_image_spans(
                            inputs["input_ids"],
                            inputs["attention_mask"],
                            old_lens,
                            new_lens,
                        )
                    with _timed(stage, "grid_synthesis_ms"):
                        new_image_grid_thw = wrapper._synthesize_grid_thw(
                            new_lens,
                            old_lens,
                            inputs["image_grid_thw"],
                        )
                    prepared = {
                        "input_ids": new_input_ids,
                        "attention_mask": new_attention_mask,
                        "pixel_values": inputs["pixel_values"],
                        "image_grid_thw": new_image_grid_thw,
                    }
                    target, restore_state = wrapper._patch_get_image_features(
                        compressed_per_image
                    )
                    try:
                        with _timed(stage, "prefill_generate_1_token_ms"):
                            prefill_out = model.generate(
                                **prepared,
                                max_new_tokens=1,
                                do_sample=False,
                            )
                        full_out = None
                        if max_new_tokens > 1:
                            with _timed(stage, "full_generate_ms"):
                                full_out = model.generate(
                                    **prepared,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=False,
                                )
                        else:
                            full_out = prefill_out
                    finally:
                        wrapper._unpatch_get_image_features(target, restore_state)
                    with _timed(stage, "postprocessing_ms"):
                        processor.batch_decode(full_out, skip_special_tokens=True)

                    debug = {
                        "visual_tokens_before_total": int(sum(old_lens)),
                        "visual_tokens_after_total": int(sum(new_lens)),
                        "input_sequence_length": int(inputs["input_ids"].shape[1]),
                        "prepared_sequence_length": int(prepared["input_ids"].shape[1]),
                    }

            if "full_generate_ms" in stage and "prefill_generate_1_token_ms" in stage:
                stage["decode_generation_proxy_ms"] = max(
                    0.0, stage["full_generate_ms"] - stage["prefill_generate_1_token_ms"]
                )
            subtotal_cols = {"input_build_total_ms", "full_generate_ms"}
            total = sum(
                v
                for k, v in stage.items()
                if k.endswith("_ms") and k not in subtotal_cols
            )
            row = {
                "method": method,
                "retention_ratio": ratio,
                "run": run_idx,
                **stage,
                **debug,
            }
            row["timed_total_ms"] = total
            rows.append(row)
            torch.cuda.empty_cache()
            gc.collect()

    per_run = pd.DataFrame(rows)
    stage_cols = [c for c in per_run.columns if c.endswith("_ms")]
    grouped = per_run.groupby(["method", "retention_ratio"], as_index=False)
    summary = grouped[stage_cols].mean()
    subtotal_cols = {"input_build_total_ms", "full_generate_ms"}
    for col in stage_cols:
        if col == "timed_total_ms" or col in subtotal_cols:
            continue
        summary[col.replace("_ms", "_pct")] = (
            summary[col] / summary["timed_total_ms"] * 100.0
        )
    for col in [
        "visual_tokens_before_total",
        "visual_tokens_after_total",
        "input_sequence_length",
        "prepared_sequence_length",
    ]:
        if col in per_run.columns:
            extra = grouped[col].mean()
            summary = summary.merge(extra, on=["method", "retention_ratio"], how="left")
    return per_run, summary


def extreme_compression_sanity_check(
    model,
    processor,
    image: Image.Image | None = None,
    prompt: str = "Describe this image in detail.",
    device: str = "cuda",
    methods: list[str] | None = None,
    retention_ratios: list[float] | None = None,
    max_new_tokens: int = 1,
    num_warmup: int = 1,
    num_runs: int = 5,
    resolution: tuple[int, int] = (896, 896),
) -> pd.DataFrame:
    """Measure latency for aggressive retention ratios down to 1%."""
    methods = methods or ["none", "fixed_ratio", "importance", "token_merging"]
    retention_ratios = retention_ratios or [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    image = image or make_random_image(resolution[0], resolution[1], seed=999)
    rows = []

    for method in methods:
        ratios = [1.0] if method == "none" else [r for r in retention_ratios if r < 1.0]
        for ratio in ratios:
            wrapper = build_compressed_wrapper(model, processor, method, ratio)
            wrapper.enable_debug = True

            def run_once():
                inputs, _ = build_qwen_inputs_timed(processor, device, image, prompt)
                with torch.no_grad():
                    output = wrapper.generate(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                return output, dict(getattr(wrapper, "last_debug", {}))

            for _ in range(num_warmup):
                run_once()
                _sync()
            times = []
            debug = {}
            for _ in range(num_runs):
                _sync()
                start = time.perf_counter()
                _, debug = run_once()
                _sync()
                times.append((time.perf_counter() - start) * 1000.0)
            rows.append(
                {
                    "method": method,
                    "retention_ratio": ratio,
                    "latency_ms": float(np.mean(times)),
                    "latency_std_ms": float(np.std(times)),
                    "visual_tokens_before": debug.get("visual_tokens_before_total"),
                    "visual_tokens_after": debug.get("visual_tokens_after_total"),
                    "prepared_sequence_length": debug.get("prepared_sequence_length"),
                    "input_sequence_length": debug.get("input_sequence_length"),
                }
            )

    df = pd.DataFrame(rows)
    baseline = df[df["method"] == "none"]["latency_ms"].mean()
    if baseline:
        df["speed_vs_baseline"] = baseline / df["latency_ms"]
    return df


def diagnose_speedup_root_cause(
    sequence_df: pd.DataFrame,
    stage_summary_df: pd.DataFrame,
    extreme_df: pd.DataFrame,
) -> str:
    """Generate a concise diagnosis from diagnostic tables."""
    lines = []
    if not sequence_df.empty:
        compressed = sequence_df[sequence_df["method"] != "none"]
        if not compressed.empty and (
            compressed["prepared_sequence_length"] < compressed["input_sequence_length"]
        ).any():
            lines.append(
                "Working: compressed runs shorten prepared input_ids/attention_mask "
                "before model.generate."
            )
        if "first_llm_layer_input_tensor_shape" in sequence_df:
            lines.append(
                "Check first_llm_layer_input_tensor_shape in the sequence diagnostics "
                "to confirm the shorter sequence reaches decoder layers."
            )

    if not stage_summary_df.empty:
        compressed = stage_summary_df[stage_summary_df["method"] != "none"]
        if not compressed.empty:
            overhead_cols = [
                "vision_encoder_forward_ms",
                "compression_scoring_merging_ms",
                "placeholder_rewrite_ms",
                "grid_synthesis_ms",
                "dtype_device_cast_ms",
            ]
            present = [c for c in overhead_cols if c in compressed.columns]
            if present:
                overhead = compressed[present].sum(axis=1).mean()
                lines.append(
                    f"Likely overhead: compressed path adds about {overhead:.1f} ms "
                    "on average across measured overhead stages."
                )

    if not extreme_df.empty and "speed_vs_baseline" in extreme_df:
        best = extreme_df[extreme_df["method"] != "none"].sort_values(
            "speed_vs_baseline", ascending=False
        )
        if not best.empty:
            row = best.iloc[0]
            if row["speed_vs_baseline"] <= 1.05:
                lines.append(
                    "Extreme compression sanity check did not show material speedup; "
                    "this suggests either the LLM path is not the dominant bottleneck "
                    "or overhead cancels the reduced sequence length."
                )
            else:
                lines.append(
                    f"Extreme compression can speed up at least one setting: "
                    f"{row['method']} r={row['retention_ratio']} reached "
                    f"{row['speed_vs_baseline']:.2f}x baseline."
                )

    lines.append(
        "Highest-impact fixes: (1) avoid recomputing or patching vision features "
        "inside every generate call, (2) integrate compression earlier/in-model so "
        "shorter sequence lengths are used without Python monkey-patch overhead, "
        "(3) use component-level profiling to optimize the largest measured stage "
        "before adding new compression algorithms."
    )
    return "\n".join(f"- {line}" for line in lines)
