# Compression Speedup Diagnostics

This document records the current root-cause debugging plan for missing
visual-token compression speedups.

## Current Code-Path Audit

The current compressed path in `CompressedVLM` does the following:

1. Runs the Qwen vision tower to compute full visual embeddings.
2. Splits visual embeddings per image.
3. Applies pruning or merging to the visual embeddings.
4. Rewrites `input_ids` and `attention_mask` so the number of image
   placeholders matches the compressed visual-token count.
5. Synthesizes a new `image_grid_thw` for M-RoPE compatibility.
6. Monkey-patches `get_image_features` so `model.generate` receives the
   precomputed compressed embeddings.
7. Calls the normal `model.generate` path.

This means compression can only reduce the text-side LLM/prefill cost. It does
not remove vision encoder cost, and it adds compression, rewriting, grid
synthesis, monkey-patching, tensor allocation, and possible copy/cast overhead.

## What the Diagnostics Check

Run:

```bash
python scripts/run_diagnostics.py \
  --config configs/default.yaml \
  --output results/diagnostics \
  --resolution 896,896 \
  --ratios 1.0,0.5,0.25,0.1,0.05,0.01 \
  --num-runs 3
```

Expected outputs:

- `sequence_length_diagnostics.csv`
- `module_shape_events.csv`
- `stage_timing_per_run.csv`
- `stage_timing_summary.csv`
- `extreme_compression_sanity.csv`
- `diagnosis_summary.md`

## Interpretation Checklist

1. **Effective sequence length**
   - Check `visual_tokens_before_total` and `visual_tokens_after_total`.
   - Check `input_sequence_length` and `prepared_sequence_length`.
   - Check `first_llm_layer_input_tensor_shape` in
     `sequence_length_diagnostics.csv`.
   - Check per-module entries in `module_shape_events.csv`.

2. **Overhead vs savings**
   - In `stage_timing_summary.csv`, compare:
     - `vision_encoder_forward_ms`
     - `compression_scoring_merging_ms`
     - `placeholder_rewrite_ms`
     - `grid_synthesis_ms`
     - `prefill_generate_1_token_ms`
     - `decode_generation_proxy_ms`
   - If `prefill_generate_1_token_ms` drops but total time does not, overhead is
     canceling the savings.
   - If `prefill_generate_1_token_ms` does not drop even under extreme
     compression, compression is either not affecting the true compute-heavy
     path or LLM prefill is not the dominant bottleneck at this scale.

3. **Extreme compression sanity**
   - In `extreme_compression_sanity.csv`, compare `speed_vs_baseline`.
   - If 10%, 5%, or 1% retention still barely changes latency, do not claim
     practical speedup. Treat that as evidence that token-count reduction is
     not translating into wall-clock improvement.

## Likely Root Cause Before Running Diagnostics

From the code audit, the most likely explanation is a combination of:

- The vision encoder still runs on the full image tokens.
- Compression is applied after vision encoding, so it only affects the LLM side.
- Python-side wrapper work, token rewriting, grid synthesis, tensor casts, and
  monkey-patching add overhead.
- Prior benchmarks used `max_new_tokens=1` to isolate prefill, which is correct
  for visual-token effects, but the measured path still includes compressed-path
  overhead.

## Highest-Impact Fixes to Try Next

1. Move compression into the model path earlier and avoid per-call monkey
   patching / Python-side rewrite overhead.
2. Cache or precompute all static metadata needed for `image_grid_thw`,
   placeholder spans, and compressed-token routing.
3. Use component timing to optimize the largest measured overhead stage before
   adding new compression algorithms.
