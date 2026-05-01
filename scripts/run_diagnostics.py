"""Run root-cause diagnostics for visual-token compression speedups.

Example:
    python scripts/run_diagnostics.py --config configs/default.yaml --output results/diagnostics

This script is intended for a GPU runtime such as Colab A100. It saves:
  * sequence_length_diagnostics.csv
  * module_shape_events.csv
  * stage_timing_per_run.csv
  * stage_timing_summary.csv
  * extreme_compression_sanity.csv
  * diagnosis_summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import load_model
from src.evaluation.diagnostics import (
    diagnose_speedup_root_cause,
    extreme_compression_sanity_check,
    sequence_length_diagnostics,
    stage_timing_breakdown,
)
from src.evaluation.experiments import make_random_image


def _parse_ratios(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Diagnose missing VLM compression speedup.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="results/diagnostics")
    parser.add_argument("--resolution", default="896,896", help="height,width")
    parser.add_argument("--methods", default="none,fixed_ratio,importance,token_merging")
    parser.add_argument("--ratios", default="1.0,0.5,0.25,0.1,0.05,0.01")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-warmup", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model, processor = load_model(config)
    model.eval()

    height, width = [int(x.strip()) for x in args.resolution.split(",")]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    ratios = _parse_ratios(args.ratios)
    image = make_random_image(height, width, seed=2026)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("Running sequence-length diagnostics...")
    sequence_df, event_df = sequence_length_diagnostics(
        model=model,
        processor=processor,
        image=image,
        prompt=args.prompt,
        methods=methods,
        retention_ratios=ratios,
        max_new_tokens=1,
        resolution=(height, width),
    )
    sequence_df.to_csv(output / "sequence_length_diagnostics.csv", index=False)
    event_df.to_csv(output / "module_shape_events.csv", index=False)

    print("Running stage timing breakdown...")
    stage_runs, stage_summary = stage_timing_breakdown(
        model=model,
        processor=processor,
        image=image,
        prompt=args.prompt,
        methods=methods,
        retention_ratios=ratios,
        max_new_tokens=args.max_new_tokens,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        resolution=(height, width),
    )
    stage_runs.to_csv(output / "stage_timing_per_run.csv", index=False)
    stage_summary.to_csv(output / "stage_timing_summary.csv", index=False)

    print("Running extreme compression sanity check...")
    extreme_df = extreme_compression_sanity_check(
        model=model,
        processor=processor,
        image=image,
        prompt=args.prompt,
        methods=methods,
        retention_ratios=ratios,
        max_new_tokens=1,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        resolution=(height, width),
    )
    extreme_df.to_csv(output / "extreme_compression_sanity.csv", index=False)

    diagnosis = diagnose_speedup_root_cause(sequence_df, stage_summary, extreme_df)
    (output / "diagnosis_summary.md").write_text(diagnosis)
    print("\nDiagnosis summary:\n")
    print(diagnosis)
    print(f"\nSaved diagnostics to {output}")


if __name__ == "__main__":
    main()
