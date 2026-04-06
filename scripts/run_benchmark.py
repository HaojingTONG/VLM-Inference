"""Main entry point: run the full benchmark grid."""

import argparse
import itertools
import json
from pathlib import Path

import yaml

from src.evaluation.evaluator import Evaluator
from src.utils.data_loader import SyntheticDataset


def main():
    parser = argparse.ArgumentParser(description="VLM Visual Token Compression Benchmark")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    methods = ["none", "fixed_ratio", "importance", "token_merging"]
    retention_ratios = base_config["benchmark"]["retention_ratios"]
    resolutions = base_config["benchmark"]["resolutions"]

    all_results = []

    for method, ratio, res in itertools.product(methods, retention_ratios, resolutions):
        # Skip compression ratios for baseline
        if method == "none" and ratio != 1.0:
            continue
        if method != "none" and ratio == 1.0:
            continue

        print(f"\n{'='*60}")
        print(f"Running: method={method}, ratio={ratio}, resolution={res}")
        print(f"{'='*60}")

        # Override config for this run
        config = base_config.copy()
        config["compression"] = {**base_config["compression"], "method": method, "retention_ratio": ratio}

        dataset = SyntheticDataset(
            num_samples=base_config["data"]["num_samples"],
            resolutions={res: SyntheticDataset().resolutions[res]},
        )

        evaluator = Evaluator.__new__(Evaluator)
        evaluator.config = config
        # TODO: Initialize model and compressor properly
        # evaluator.model, evaluator.processor = load_model(config)
        # evaluator.compressor = build_compressor(config)

        run_result = {
            "method": method,
            "retention_ratio": ratio,
            "resolution": res,
            # "metrics": evaluator.run_benchmark(dataset, args.output),
        }
        all_results.append(run_result)

    # Save summary
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "benchmark_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBenchmark complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
