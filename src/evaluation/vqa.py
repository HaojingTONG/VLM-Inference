"""VQA dataset validation and honest answer scoring.

The important distinction in this module is naming. Official VQA accuracy is
only valid when each evaluated question has the VQA-style set of 10 human
answers. Datasets that expose only ``multiple_choice_answer`` or one reference
can still be useful, but they should be reported as exact match instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import re


_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "lets": "let's",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyll": "they'll",
    "theyre": "they're",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whats": "what's",
    "wheres": "where's",
    "whos": "who's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
_NUM_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]
_PERIOD_STRIP = re.compile(r"(?<!\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


@dataclass
class VQASchemaReport:
    """Summary of whether a dataset split can support a named VQA metric."""

    dataset_name: str
    n_checked: int
    fields: list[str]
    image_field: str | None
    question_field: str | None
    answer_field: str | None
    multiple_choice_field: str | None
    answer_count_min: int
    answer_count_max: int
    supports_official_vqa_accuracy: bool
    selected_metric: str
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_answer(answer: Any) -> str:
    """Normalize an answer using the standard VQA text conventions."""
    text = "" if answer is None else str(answer)
    text = text.lower().strip()

    for punct in _PUNCT:
        if (punct + " " in text or " " + punct in text) or _COMMA_STRIP.search(text):
            text = text.replace(punct, "")
        else:
            text = text.replace(punct, " ")
    text = _PERIOD_STRIP.sub("", text)

    tokens = []
    for token in text.split():
        token = _NUM_WORDS.get(token, token)
        token = _CONTRACTIONS.get(token, token)
        if token in _ARTICLES:
            continue
        tokens.append(token)
    return " ".join(tokens).strip()


def exact_match_score(prediction: Any, reference: Any) -> float:
    """Normalized exact match against a single reference answer."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def official_vqa_score(prediction: Any, human_answers: Iterable[Any]) -> float:
    """Compute official VQA consensus score for one prediction.

    The official protocol expects 10 human answers. This function enforces that
    contract so callers cannot silently report official VQA accuracy on a
    dataset that only exposes one answer.
    """
    answers = [normalize_answer(a) for a in human_answers if a is not None]
    if len(answers) != 10:
        raise ValueError(
            "official_vqa_score requires exactly 10 human answers; "
            f"received {len(answers)}."
        )

    pred = normalize_answer(prediction)
    per_annotator_scores = []
    for i in range(10):
        other_answers = answers[:i] + answers[i + 1 :]
        matches = sum(1 for answer in other_answers if answer == pred)
        per_annotator_scores.append(min(matches / 3.0, 1.0))
    return sum(per_annotator_scores) / 10.0


def extract_human_answers(sample: dict[str, Any]) -> list[str]:
    """Extract VQA-style human answers from common dataset schemas."""
    answers = sample.get("answers")
    if isinstance(answers, list):
        if not answers:
            return []
        if isinstance(answers[0], dict):
            extracted = []
            for item in answers:
                value = item.get("answer")
                if value is not None:
                    extracted.append(str(value))
            return extracted
        return [str(answer) for answer in answers if answer is not None]

    if isinstance(answers, dict):
        # Some datasets store {"answer": [...]} or {"text": [...]}.
        for key in ("answer", "answers", "text"):
            value = answers.get(key)
            if isinstance(value, list):
                return [str(answer) for answer in value if answer is not None]
    return []


def extract_single_reference(sample: dict[str, Any]) -> str | None:
    """Extract the best available single-reference answer for exact match."""
    for key in ("multiple_choice_answer", "answer", "label"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value

    human_answers = extract_human_answers(sample)
    if human_answers:
        return human_answers[0]
    return None


def validate_vqa_schema(
    samples: Iterable[dict[str, Any]],
    dataset_name: str = "unknown",
    max_samples: int | None = None,
) -> VQASchemaReport:
    """Inspect samples and choose the strongest honest metric they support."""
    checked = []
    for i, sample in enumerate(samples):
        if max_samples is not None and i >= max_samples:
            break
        checked.append(sample)

    if not checked:
        return VQASchemaReport(
            dataset_name=dataset_name,
            n_checked=0,
            fields=[],
            image_field=None,
            question_field=None,
            answer_field=None,
            multiple_choice_field=None,
            answer_count_min=0,
            answer_count_max=0,
            supports_official_vqa_accuracy=False,
            selected_metric="unscored",
            issues=["No samples were available for schema validation."],
        )

    fields = sorted({key for sample in checked for key in sample.keys()})
    image_field = next((k for k in ("image", "image_path", "path") if k in fields), None)
    question_field = next((k for k in ("question", "prompt") if k in fields), None)
    multiple_choice_field = next(
        (k for k in ("multiple_choice_answer", "answer", "label") if k in fields),
        None,
    )
    answer_field = "answers" if "answers" in fields else multiple_choice_field

    answer_counts = [len(extract_human_answers(sample)) for sample in checked]
    answer_count_min = min(answer_counts) if answer_counts else 0
    answer_count_max = max(answer_counts) if answer_counts else 0

    issues = []
    if image_field is None:
        issues.append("No image field found; expected one of image, image_path, path.")
    if question_field is None:
        issues.append("No question field found; expected question or prompt.")

    supports_official = (
        image_field is not None
        and question_field is not None
        and bool(answer_counts)
        and all(count == 10 for count in answer_counts)
    )
    if supports_official:
        selected_metric = "official_vqa_accuracy"
    elif any(extract_single_reference(sample) is not None for sample in checked):
        selected_metric = (
            "multiple_choice_exact_match"
            if "multiple_choice_answer" in fields
            else "single_reference_exact_match"
        )
        issues.append(
            "Dataset does not expose exactly 10 human answers for every checked "
            f"sample (observed min={answer_count_min}, max={answer_count_max}); "
            f"using {selected_metric} instead of official VQA accuracy."
        )
    else:
        selected_metric = "unscored"
        issues.append("No usable answer references found.")

    return VQASchemaReport(
        dataset_name=dataset_name,
        n_checked=len(checked),
        fields=fields,
        image_field=image_field,
        question_field=question_field,
        answer_field=answer_field,
        multiple_choice_field=multiple_choice_field,
        answer_count_min=answer_count_min,
        answer_count_max=answer_count_max,
        supports_official_vqa_accuracy=supports_official,
        selected_metric=selected_metric,
        issues=issues,
    )


def score_vqa_prediction(
    prediction: Any,
    sample: dict[str, Any],
    metric_name: str,
) -> tuple[float | None, list[str] | str | None]:
    """Score one prediction using the metric selected by schema validation."""
    if metric_name == "official_vqa_accuracy":
        answers = extract_human_answers(sample)
        return official_vqa_score(prediction, answers), answers

    if metric_name in {"multiple_choice_exact_match", "single_reference_exact_match"}:
        reference = extract_single_reference(sample)
        if reference is None:
            return None, None
        return exact_match_score(prediction, reference), reference

    return None, None
