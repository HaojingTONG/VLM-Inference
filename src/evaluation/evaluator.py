"""Evaluation pipeline: run inference with compression and collect metrics."""

import json
from pathlib import Path
import torch
import yaml

from src.models import load_model
from src.compression import build_compressor, CompressedVLM
from src.utils.profiler import InferenceProfiler


class Evaluator:
    """Orchestrates benchmark evaluation across compression strategies."""

    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model, self.processor = load_model(self.config)
        self.compressor = build_compressor(self.config)
        self.wrapped = CompressedVLM(self.model, self.processor, self.compressor)
        self.profiler = InferenceProfiler(
            num_warmup=self.config["evaluation"]["num_warmup"],
            num_runs=self.config["evaluation"]["num_runs"],
        )

    def _build_inputs(self, image, question):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)

    def run_single(self, image, question):
        """Run a single inference with optional token compression."""
        inputs = self._build_inputs(image, question)

        output_ids = self.wrapped.generate(
            inputs,
            max_new_tokens=self.config["model"]["max_new_tokens"],
        )
        # Strip the prompt tokens when available (only when passing input_ids)
        if "input_ids" in inputs:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        generated_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        return {"generated_text": generated_text}

    def run_benchmark(self, dataset, output_dir=None):
        """Run full benchmark on a dataset and save results."""
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

    def profile_single(self, image, question, max_new_tokens=None):
        """Profile latency / throughput / memory for a single (image, question) pair."""
        inputs = self._build_inputs(image, question)
        max_new_tokens = max_new_tokens or self.config["model"]["max_new_tokens"]

        def inference_fn():
            return self.wrapped.generate(inputs, max_new_tokens=max_new_tokens)

        return self.profiler.profile(inference_fn)
