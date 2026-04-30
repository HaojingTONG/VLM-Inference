"""Dataset loading utilities for VQA-style and synthetic prompts."""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from src.evaluation.vqa import (
    extract_human_answers,
    extract_single_reference,
    validate_vqa_schema,
)


class VQADataset(Dataset):
    """Wrapper for local VQA-style annotations.

    Expected files:
      * ``questions_<split>.json`` with records containing question_id,
        image_id, and question, or a top-level ``questions`` list.
      * ``annotations_<split>.json`` with VQA-v2 style records containing
        question_id, multiple_choice_answer, and optionally 10 human answers,
        or a top-level ``annotations`` list.

    This loader does not claim official VQA support by itself. Call
    ``schema_report`` and use its selected metric before evaluation.
    """

    def __init__(self, data_dir, split="val", num_samples=500, seed=42, image_dir=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_samples = num_samples
        self.seed = seed
        self.image_dir = Path(image_dir) if image_dir else self.data_dir / "images"
        self.samples = self._load_samples()
        self.schema_report = validate_vqa_schema(
            [self._sample_without_image(i) for i in range(len(self.samples))],
            dataset_name=f"local:{self.data_dir}",
        )

    def _load_samples(self):
        import json
        import random

        questions_path = self.data_dir / f"questions_{self.split}.json"
        annotations_path = self.data_dir / f"annotations_{self.split}.json"
        if not questions_path.exists() or not annotations_path.exists():
            raise FileNotFoundError(
                "Expected local VQA files "
                f"{questions_path.name} and {annotations_path.name} in {self.data_dir}."
            )

        with open(questions_path) as f:
            questions_obj = json.load(f)
        with open(annotations_path) as f:
            annotations_obj = json.load(f)

        questions = questions_obj.get("questions", questions_obj)
        annotations = annotations_obj.get("annotations", annotations_obj)
        annotations_by_qid = {ann["question_id"]: ann for ann in annotations}

        samples = []
        for question in questions:
            ann = annotations_by_qid.get(question["question_id"])
            if ann is None:
                continue
            image_path = self._resolve_image_path(question, ann)
            sample = {
                "question_id": question["question_id"],
                "image_id": question.get("image_id", ann.get("image_id")),
                "image_path": str(image_path),
                "question": question["question"],
                "answers": extract_human_answers(ann),
                "multiple_choice_answer": ann.get("multiple_choice_answer"),
            }
            samples.append(sample)

        rng = random.Random(self.seed)
        rng.shuffle(samples)
        return samples[: self.num_samples]

    def _resolve_image_path(self, question, annotation):
        image_id = question.get("image_id", annotation.get("image_id"))
        if image_id is None:
            raise ValueError("Cannot resolve local VQA image without image_id.")
        candidates = [
            self.image_dir / f"{image_id}.jpg",
            self.image_dir / f"{int(image_id):012d}.jpg",
            self.image_dir / f"COCO_{self.split}2014_{int(image_id):012d}.jpg",
            self.image_dir / f"COCO_val2014_{int(image_id):012d}.jpg",
            self.image_dir / f"COCO_train2014_{int(image_id):012d}.jpg",
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[1]

    def _sample_without_image(self, idx):
        sample = self.samples[idx].copy()
        sample.pop("image", None)
        return sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        return {
            "image": image,
            "question": sample["question"],
            "answers": sample.get("answers", []),
            "multiple_choice_answer": sample.get("multiple_choice_answer"),
            "reference": extract_single_reference(sample),
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
