# Efficient VLM Inference via Visual Token Compression

**Authors:** Haojing Tong (ht2667), Zhuoao Wang (zw4721), Yuqi Wang (yw4338)

## Overview

This project benchmarks visual token compression strategies for Vision-Language Models (VLMs). We measure the efficiency-accuracy trade-off of different token reduction methods on Qwen2.5-VL under varying input resolutions, context densities, and compression intensities.

## Project Structure

```
configs/
  default.yaml              # Experiment configuration
data/                        # Dataset directory
src/
  models/
    model_loader.py          # Load Qwen2.5-VL / LLaVA models
  compression/
    base.py                  # Abstract compressor base class
    fixed_ratio.py           # Fixed-ratio uniform token pruning
    importance.py            # Importance-based pruning (attention / magnitude / similarity)
    token_merging.py         # ToMe-style bipartite token merging
  evaluation/
    evaluator.py             # Evaluation pipeline
    metrics.py               # VQA accuracy and token reduction stats
  utils/
    profiler.py              # Latency, throughput, peak GPU memory profiling
    data_loader.py           # VQA-v2 and synthetic dataset loaders
scripts/
  run_benchmark.py           # Full benchmark grid runner
  run_single.py              # Single inference for debugging / demo
  submit_hpc.sh              # NYU HPC Slurm submission script (A100)
notebooks/                   # Analysis and visualization
results/                     # Experiment outputs
```

## Compression Methods

| # | Method | Description |
|---|--------|-------------|
| 1 | Baseline | Full visual tokens, no compression |
| 2 | Fixed-Ratio Pruning | Uniformly sample a fixed percentage (75%/50%/25%/10%) of tokens |
| 3 | Importance-Based Pruning | Score tokens by attention/magnitude/similarity, keep top-k |
| 4 | Token Merging | Merge similar tokens via bipartite matching (ToMe-style) |

## Current Progress

- [x] Project scaffold and directory structure
- [x] Experiment configuration system (`configs/default.yaml`)
- [x] Model loading utility for Qwen2.5-VL (HuggingFace Transformers)
- [x] Three compression strategies implemented (fixed-ratio, importance-based, token merging)
- [x] GPU profiling utility (latency, throughput, peak memory, max batch size search)
- [x] Dataset loaders (VQA-v2 skeleton, synthetic dataset with controllable resolution)
- [x] Evaluation pipeline and quality metrics
- [x] Benchmark runner with method x ratio x resolution grid
- [x] NYU HPC Slurm job script

## TODO

- [ ] Hook compressor into Qwen2.5-VL forward pass (intercept visual tokens between vision encoder and LLM)
- [ ] Implement VQA-v2 data loading with actual annotations
- [ ] Add multi-image input support
- [ ] Run baseline experiments on HPC
- [ ] Visualization notebooks (latency-accuracy curves, memory plots)
- [ ] (Optional) LLaVA-v1.6-7B comparison experiments

## Quick Start

```bash
pip install -r requirements.txt

# Single inference test
python scripts/run_single.py --image path/to/image.jpg --method fixed_ratio --ratio 0.5

# Full benchmark
python scripts/run_benchmark.py --config configs/default.yaml

# Submit to NYU HPC
sbatch scripts/submit_hpc.sh
```

## Hardware

- NYU HPC cluster, NVIDIA A100 GPUs, single-node
- PyTorch 2.x + HuggingFace Transformers + FlashAttention-2
