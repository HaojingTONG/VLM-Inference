# Efficient VLM Inference via Visual Token Compression

**Authors:** Haojing Tong (ht2667), Zhuoao Wang (zw4721), Yuqi Wang (yw4338)

This repository evaluates visual-token compression for high-resolution
vision-language model (VLM) inference, using Qwen2.5-VL as the main target
model. The project studies whether reducing visual tokens can lower inference
cost while preserving visual-question-answering quality.

The main analysis artifact is:

```text
notebooks/colab_run.ipynb
```
Team members from the original project proposal/checkpoint:

Haojing Tong (ht2667)
Yuqi Wang (yw4338)
Primary runtime target: Google Colab A100 GPU.
Primary model: Qwen/Qwen2.5-VL-3B-Instruct.


It is intended to run on Google Colab or a GPU environment such as an NVIDIA
A100. The notebook validates the dataset schema, runs corrected quality
evaluation, benchmarks baseline and compressed inference, saves CSV/JSON
results, and generates presentation-ready plots.

## Executive Summary
High-performance VLMs can generate hundreds or thousands of visual tokens for high-resolution or multi-image inputs. These tokens are passed into the language model during inference, increasing prefill cost, KV-cache size, GPU memory usage, latency, and limiting throughput.

This project implements a reproducible benchmark suite for evaluating visual token compression. It compares:

full visual tokens with no compression,
fixed-ratio visual token pruning,
importance-based visual token pruning,
token merging.
The most important engineering component is a model-aware Qwen2.5-VL fixed-ratio pruning adapter. It intercepts visual embeddings after Qwen2.5-VL's visual encoder and before LLM prefill, prunes image embeddings, and rebuilds the sequence metadata required by Qwen2.5-VL generation. This creates a genuinely shorter LLM input sequence instead of only simulating compression at the image level.

The benchmark records latency, throughput, peak GPU memory, visual-token counts, generated answers, strict VQA-style accuracy, and OOM/success status. A 50-sample synthetic stress VQA/OCR dataset is included so compression-induced accuracy degradation is visible even without downloading external datasets.

## Relation to the Original Proposal
The original proposal aimed to study Efficiency-Accuracy Trade-off for visual token compression under a benchmark grid:

image resolution: low, medium, high,
image density: 1, 2, and 4 images per prompt,
compression intensity: 100% down to aggressive retention ratios,
metrics: latency, throughput, peak GPU memory, OOM behavior, and answer quality.
This repository implements that direction with a Colab-first stack:

PyTorch and Hugging Face Transformers instead of an HPC/Slurm-first setup,
Qwen2.5-VL-3B-Instruct as the main model,
Qwen2-VL-2B-Instruct as a fallback,
real internal fixed-ratio pruning for Qwen2.5-VL,
standalone/proxy baselines for importance pruning and token merging,
CSV logging and notebook plots for efficiency-accuracy trade-off analysis.
The proposal also discussed attention-based pruning, ToMe-style bipartite merging, vLLM/DeepSpeed, and large external VQA benchmarks. Those are kept as future extensions. The current implementation prioritizes a stable, reproducible Colab A100 workflow.

## What This Project Claims

The project makes a careful distinction between three effects:

1. **Visual-token reduction:** the compressor reduces the number of visual
   embeddings.
2. **LLM-side sequence reduction:** the Qwen input sequence and attention mask
   are rewritten so the compressed visual tokens actually enter the LLM path.
3. **End-to-end latency:** wall-clock latency depends on the whole pipeline,
   including preprocessing, the vision encoder, compression overhead, LLM
   prefill, decode, and postprocessing.

Do not interpret every compression setting as an end-to-end speedup. The
current late-compression middleware can reduce LLM-side work and KV-cache proxy
cost, but it still runs the vision encoder on the full image and adds Python
wrapper overhead.

The notebook also avoids misleading metric names:

- It reports **official VQA accuracy** only when the dataset provides the
  VQA-style 10 human answers per question.
- If the dataset does not support official VQA scoring, it falls back to clearly
  named exact-match metrics.

## Repository Layout

```text
configs/
  default.yaml                  Default model, compression, and evaluation config

docs/
  compression_diagnostics.md    Root-cause notes for missing latency speedup

notebooks/
  colab_run.ipynb               Main final analysis notebook
  colab_demo.ipynb              Lightweight demo notebook used for presentation

scripts/
  run_diagnostics.py            Sequence/timing/root-cause diagnostics
  run_single.py                 Legacy quick single-run scaffold
  run_benchmark.py              Legacy benchmark scaffold
  submit_hpc.sh                 NYU HPC Slurm submission helper

src/
  compression/
    base.py                     Compressor base class
    fixed_ratio.py              Fixed-ratio visual-token pruning
    importance.py               Importance-based pruning
    token_merging.py            ToMe-style token merging
    hook.py                     Qwen2.5-VL compression middleware

  evaluation/
    experiments.py              Notebook experiment helpers
    diagnostics.py              Sequence and stage timing diagnostics
    metrics.py                  Token/sequence efficiency proxy metrics
    plots.py                    Plot helpers
    vqa.py                      VQA schema validation and answer scoring
    evaluator.py                Older evaluator wrapper

  models/
    model_loader.py             HuggingFace model/processor loading

  utils/
    data_loader.py              Synthetic/VQA dataset helpers
    profiler.py                 CUDA timing and peak-memory profiling
```

## Environment Setup

### Recommended Hardware

- NVIDIA A100 40GB/80GB preferred for the full notebook.
- Smaller GPUs may work with the 3B model and reduced sample counts, but model
  loading and high-resolution sweeps may be memory-limited.

### Local or Colab Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

For Colab, use an A100 runtime when available, then install dependencies in the
notebook or run:

```bash
pip install -U torch torchvision transformers accelerate qwen-vl-utils datasets \
  pillow pandas matplotlib tqdm psutil pynvml pyyaml
pip install -e .
```

If Hugging Face rate limits model or dataset downloads, set:

```bash
export HF_TOKEN=your_huggingface_token
```

## How to Run the Main Notebook

The main expected workflow is to run:

```text
notebooks/colab_run.ipynb
```

Run the notebook top-to-bottom on a CUDA runtime. The major sections are:

1. **Project Goals and Hypotheses**
   - Summarizes the project hypotheses from the proposal.
2. **Environment Setup / Clone and Install**
   - Installs dependencies and adds the repo to `sys.path`.
3. **Load Qwen2.5-VL**
   - Loads `Qwen/Qwen2.5-VL-3B-Instruct` by default.
4. **VQA Dataset Schema Validation**
   - Checks whether the dataset supports official VQA accuracy.
5. **Corrected Quality Evaluation**
   - Runs baseline and compressed VQA evaluation.
6. **Baseline Full-Token Performance**
   - Measures baseline behavior across image resolutions.
7. **Fair Baseline vs Compression Performance**
   - Compares methods and retention ratios under the same measurement setup.
8. **Plots**
   - Generates quality, latency, memory, and tradeoff plots.
9. **Final Findings and Limitations**
   - Writes an honest summary of what worked and what did not.
10. **Result Files**
   - Lists saved CSV/JSON/PNG outputs.

The notebook saves results under `results/`, including CSV/JSON tables and
plots. Typical outputs include:

```text
results/vqa_quality_summary.csv
results/vqa_predictions.csv
results/compression_performance.csv
results/quality_vs_compression.png
results/quality_ci_vs_compression.png
results/latency_vs_compression.png
results/memory_vs_compression.png
results/quality_latency_tradeoff.png
```

Exact filenames can vary slightly as the notebook evolves, but all final
tables are saved through `src.evaluation.experiments.save_results`.

## Running Root-Cause Diagnostics

Use diagnostics when you want to answer whether compression actually changes
the compute-heavy path and why speedup is or is not visible.

```bash
python scripts/run_diagnostics.py \
  --config configs/default.yaml \
  --output results/diagnostics \
  --resolution 896,896 \
  --methods none,fixed_ratio,importance,token_merging \
  --ratios 1.0,0.5,0.25,0.1,0.05,0.01 \
  --max-new-tokens 16 \
  --num-warmup 1 \
  --num-runs 3
```

This writes:

```text
results/diagnostics/sequence_length_diagnostics.csv
results/diagnostics/module_shape_events.csv
results/diagnostics/stage_timing_per_run.csv
results/diagnostics/stage_timing_summary.csv
results/diagnostics/extreme_compression_sanity.csv
results/diagnostics/early_resize_sanity.csv
results/diagnostics/diagnosis_summary.md
```

Use these files to check:

- visual tokens before and after compression
- rewritten `input_ids` and `attention_mask` lengths
- first LLM-layer and attention-module tensor shapes
- image preprocessing time
- processor/tokenization time
- vision encoder time
- compression/scoring/merging overhead
- LLM prefill timing
- decode/generation timing
- postprocessing timing
- extreme compression sanity checks
- early resize/token-budget sanity checks

## Compressor Architecture

The compressor stack has two layers:

1. **Compression algorithms** in `src/compression/*.py`
2. **Qwen2.5-VL integration middleware** in `src/compression/hook.py`

### High-Level Data Flow

```text
PIL image + prompt
    |
    v
Qwen processor
    |
    |-- input_ids with <image_pad> spans
    |-- attention_mask
    |-- pixel_values
    |-- image_grid_thw
    v
CompressedVLM.generate()
    |
    v
vision tower produces visual embeddings
    |
    v
compressor.compress(visual_embeddings)
    |
    |-- fixed_ratio
    |-- importance
    |-- token_merging
    v
rewrite input_ids / attention_mask / image_grid_thw
    |
    v
patch get_image_features() to return compressed embeddings
    |
    v
normal model.generate() path
    |
    v
decoded answer
```

### Base Interface

All compressors inherit from `BaseCompressor`:

```python
class BaseCompressor:
    def compress(self, visual_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        ...
```

Input shape:

```text
(batch_size, num_visual_tokens, hidden_dim)
```

Output shape:

```text
(batch_size, retained_visual_tokens, hidden_dim)
```

The number of retained tokens is controlled by:

```text
retained_visual_tokens = int(num_visual_tokens * retention_ratio)
```

Examples:

```text
retention_ratio = 1.00 -> keep 100% of visual tokens
retention_ratio = 0.75 -> keep 75%
retention_ratio = 0.50 -> keep 50%
retention_ratio = 0.25 -> keep 25%
retention_ratio = 0.10 -> keep 10%
```

### Fixed-Ratio Pruning

File:

```text
src/compression/fixed_ratio.py
```

Theory:

- Keep evenly spaced visual tokens.
- Does not inspect image content.
- Lowest overhead and useful as a baseline.

Implementation behavior:

```text
N visual tokens -> choose K evenly spaced indices -> output K tokens
```

Strength:

- Simple and cheap.

Weakness:

- Can delete important small details such as OCR text, numbers, object edges, or
  spatial markers.

### Importance-Based Pruning

File:

```text
src/compression/importance.py
```

Theory:

- Score each visual token.
- Keep the top-k highest-scoring tokens.
- Preserve original spatial order after selection.

Supported scoring signals:

```text
attention   Uses attention weights if available; falls back to magnitude.
magnitude   Uses embedding norm.
similarity  Gives higher score to more unique tokens.
```

In the notebook helper path, the default practical setting is magnitude-based
importance because it does not require extracting attention maps:

```text
score(token) = ||token_embedding||
```

Strength:

- More content-aware than fixed pruning.
- Good practical quality-efficiency tradeoff.

Weakness:

- The score is a heuristic and may not always match true task relevance.
- Top-k scoring adds overhead.

### Token Merging / ToMe-Style Compression

File:

```text
src/compression/token_merging.py
```

Theory:

- Many visual tokens are redundant, especially background or repeated texture.
- Instead of dropping tokens, merge similar tokens.
- This preserves some information from removed tokens.

Implementation behavior:

1. Split tokens into even-indexed source tokens and odd-indexed destination
   tokens.
2. Compute cosine similarity from source to destination tokens.
3. Select highly similar source tokens.
4. Average each selected source token into its best destination token.
5. Repeat until the requested retention ratio is reached.

Strength:

- More information-preserving than direct pruning.
- Often more stable than fixed pruning under moderate compression.

Weakness:

- Pairwise similarity and scatter/merge operations add compute overhead.
- It is not always the fastest method despite preserving quality well.

## Qwen2.5-VL Middleware Design

The key integration class is:

```text
src/compression/hook.py::CompressedVLM
```

The wrapper has to handle Qwen-specific bookkeeping. Qwen2.5-VL does not only
consume image embeddings; it also uses:

- repeated image placeholder tokens in `input_ids`
- `attention_mask`
- `image_grid_thw`
- M-RoPE image-position logic

The compressed path does the following:

1. **Compute full visual embeddings**
   - Calls the Qwen vision tower once.
   - Splits visual embeddings per image.

2. **Apply visual-token compression**
   - Calls one of `FixedRatioPruner`, `ImportanceBasedPruner`, or `TokenMerger`.

3. **Rewrite placeholder spans**
   - Shortens each `<image_pad>` span in `input_ids` to match the compressed
     visual-token count.
   - Rebuilds `attention_mask` with the same shortened sequence length.

4. **Synthesize compatible `image_grid_thw`**
   - Creates a new grid where `prod(image_grid_thw) / spatial_merge_size^2`
     equals the compressed token count.
   - This keeps Qwen's image-token count checks and M-RoPE position assignment
     consistent.

5. **Patch `get_image_features()`**
   - Temporarily monkey-patches the inner model's `get_image_features()` so the
     normal HuggingFace `model.generate()` path receives the precomputed
     compressed embeddings.
   - Restores the original method after generation.

6. **Record diagnostics when enabled**
   - Stores visual tokens before/after compression.
   - Stores tensor shapes before/after compression.
   - Stores prepared sequence length and attention-mask shape.

This design keeps the model weights unchanged and avoids retraining, but it is
a late-compression design. The vision encoder still processes the full image,
so the strongest benefits are expected in LLM prefill and KV-cache-related
costs rather than all end-to-end memory or latency components.

## Evaluation Design

### Quality Evaluation

Quality evaluation is implemented in:

```text
src/evaluation/vqa.py
src/evaluation/experiments.py
```

The notebook first runs schema validation:

```python
validate_vqa_schema(samples)
```

The selected metric is:

- `official_vqa_accuracy` if every checked sample has 10 human answers
- `multiple_choice_exact_match` if only a multiple-choice/single answer exists
- `single_reference_exact_match` if only one reference answer exists
- `unscored` if no usable answer reference is found

Official VQA scoring enforces the 10-answer requirement. This prevents the
notebook from accidentally reporting exact match as official VQA accuracy.

### Performance Evaluation

Performance helpers are implemented in:

```text
src/evaluation/experiments.py
src/evaluation/diagnostics.py
src/utils/profiler.py
```

The main measurements include:

- latency
- peak GPU memory
- generated tokens/sec
- visual tokens before/after compression
- prepared LLM sequence length
- attention/KV-cache proxy reductions
- stage-level timing

Timing helpers use CUDA synchronization around measured sections when CUDA is
available.

## Programmatic Usage

Minimal example:

```python
import yaml
from PIL import Image

from src.models import load_model
from src.evaluation.experiments import build_qwen_inputs, build_compressed_wrapper

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

model, processor = load_model(config)
image = Image.open("path/to/image.jpg").convert("RGB")
prompt = "Answer the question with a single word or short phrase. Question: What is shown?"

inputs = build_qwen_inputs(processor, model.device, image, prompt)
wrapper = build_compressed_wrapper(
    model,
    processor,
    method="token_merging",
    retention_ratio=0.25,
)

output_ids = wrapper.generate(inputs, max_new_tokens=20, do_sample=False)
answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(answer)
```

Valid methods:

```text
none
fixed_ratio
importance
token_merging
```

## Notes on Legacy Scripts

The final analysis path is the notebook plus `src/evaluation/experiments.py` and
`src/evaluation/diagnostics.py`.

The older scripts:

```text
scripts/run_single.py
scripts/run_benchmark.py
```

are retained as scaffolding but are not the recommended source of final project
results. Use `notebooks/colab_run.ipynb` for the final report and
`scripts/run_diagnostics.py` for root-cause profiling.

## Recommended Presentation Framing

The safest interpretation of the final project is:

- Moderate visual-token compression can preserve task quality.
- Importance pruning and token merging are more stable than uniform fixed
  pruning under stronger compression.
- The middleware successfully shortens the LLM-side sequence.
- Late compression does not eliminate vision encoder cost, so end-to-end
  latency and peak memory gains are implementation- and workload-dependent.
- Future work should move compression earlier, make retention adaptive, and
  replace Python-level hooks with optimized in-model or kernel-level
  implementations.
