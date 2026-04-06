"""Dataset loading utilities for VQA-v2, LLaVA-Bench, and synthetic prompts."""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class VQADataset(Dataset):
    """Wrapper for VQA-v2 subset evaluation."""

    def __init__(self, data_dir, split="val", num_samples=500, seed=42):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_samples = num_samples
        self.seed = seed
        self.samples = self._load_samples()

    def _load_samples(self):
        # TODO: Load VQA-v2 annotations and images
        # Return list of dicts: {"image_path": ..., "question": ..., "answer": ...}
        return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        return {
            "image": image,
            "question": sample["question"],
            "answer": sample["answer"],
        }


class SyntheticDataset(Dataset):
    """Synthetic image-text prompts with controllable resolution and token count."""

    def __init__(self, num_samples=100, resolutions=None):
        self.num_samples = num_samples
        self.resolutions = resolutions or {
            "low": (224, 224),
            "medium": (448, 448),
            "high": (896, 896),
        }
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        for res_name, (h, w) in self.resolutions.items():
            for i in range(self.num_samples // len(self.resolutions)):
                image = Image.new("RGB", (w, h), color=(128, 128, 128))
                samples.append({
                    "image": image,
                    "question": "Describe this image in detail.",
                    "resolution": res_name,
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
