"""Evaluation pipeline: run inference with compression and collect metrics."""

import json
from pathlib import Path
import torch
import yaml

from src.models import load_model
from src.compression import build_compressor
from src.utils.profiler import InferenceProfiler


class Evaluator:
    """Orchestrates benchmark evaluation across compression strategies."""

    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model, self.processor = load_model(self.config)
        self.compressor = build_compressor(self.config)
        self.profiler = InferenceProfiler(
            num_warmup=self.config["evaluation"]["num_warmup"],
            num_runs=self.config["evaluation"]["num_runs"],
        )

    def run_single(self, image, question):
        """Run a single inference with optional token compression.

        Args:
            image: PIL Image.
            question: str prompt.

        Returns:
            dict with generated_text and profiling metrics.
        """
        inputs = self.processor(
            text=question,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        # TODO: Hook into model forward to apply compression
        # This requires intercepting visual tokens after the vision encoder
        # and before they enter the LLM decoder layers.

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config["model"]["max_new_tokens"],
            )

        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}

    def run_benchmark(self, dataset, output_dir=None):
        """Run full benchmark on a dataset and save results.

        Args:
            dataset: Iterable of {"image": PIL.Image, "question": str, ...}.
            output_dir: Path to save results.

        Returns:
            List of result dicts.
        """
        results = []
        for i, sample in enumerate(dataset):
            result = self.run_single(sample["image"], sample["question"])
            if "answer" in sample:
                result["reference"] = sample["answer"]
            result["index"] = i
            results.append(result)

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            method = self.config["compression"]["method"]
            ratio = self.config["compression"]["retention_ratio"]
            fname = f"results_{method}_r{ratio}.json"
            with open(out_path / fname, "w") as f:
                json.dump(results, f, indent=2)

        return results
