"""Quick single-run script for debugging and demo."""

import argparse
import yaml
from PIL import Image

from src.models import load_model
from src.compression import build_compressor
from src.utils.profiler import InferenceProfiler


def main():
    parser = argparse.ArgumentParser(description="Single VLM inference with compression")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, default="Describe this image.")
    parser.add_argument("--method", type=str, default=None, help="Override compression method")
    parser.add_argument("--ratio", type=float, default=None, help="Override retention ratio")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.method:
        config["compression"]["method"] = args.method
    if args.ratio:
        config["compression"]["retention_ratio"] = args.ratio

    print(f"Loading model: {config['model']['name']}")
    model, processor = load_model(config)
    compressor = build_compressor(config)

    image = Image.open(args.image).convert("RGB")

    print(f"Compression: {config['compression']['method']} "
          f"(retention={config['compression']['retention_ratio']})")

    # TODO: Integrate compressor into inference pipeline
    profiler = InferenceProfiler()

    import torch
    inputs = processor(text=args.question, images=image, return_tensors="pt").to(model.device)

    def inference_fn(**kwargs):
        with torch.no_grad():
            return model.generate(**inputs, max_new_tokens=config["model"]["max_new_tokens"])

    metrics = profiler.profile(inference_fn)
    outputs = inference_fn()
    text = processor.decode(outputs[0], skip_special_tokens=True)

    print(f"\nGenerated: {text}")
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
