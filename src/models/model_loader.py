"""Model loading utilities for Qwen2.5-VL and LLaVA."""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


def load_model(config):
    """Load VLM model and processor based on config.

    Returns:
        model: The loaded VLM model.
        processor: The corresponding processor/tokenizer.
    """
    model_name = config["model"]["name"]
    dtype = getattr(torch, config["model"]["dtype"])
    device = config["model"]["device"]

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor
