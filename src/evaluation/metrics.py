"""Quality and efficiency metrics for evaluating VLM outputs."""

from .vqa import (
    exact_match_score,
    official_vqa_score,
    normalize_answer,
    score_vqa_prediction,
)


def compute_exact_match_accuracy(predictions, references):
    """Compute normalized exact-match accuracy against one reference per item."""
    scores = [
        exact_match_score(pred, ref)
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_vqa_accuracy(predictions, references):
    """Backward-compatible alias for normalized exact match.

    This repository used to expose ``compute_vqa_accuracy`` as a simple
    string exact-match metric. That is not official VQA accuracy. New code
    should call ``official_vqa_score`` only after validating that each sample
    contains the 10 human answers required by the VQA protocol, or use
    ``compute_exact_match_accuracy`` for single-reference / multiple-choice
    datasets.
    """
    return compute_exact_match_accuracy(predictions, references)


def compute_token_stats(visual_tokens_before, visual_tokens_after):
    """Compute token reduction statistics.

    Args:
        visual_tokens_before: Number of tokens before compression.
        visual_tokens_after: Number of tokens after compression.

    Returns:
        Dict with reduction ratio and token counts.
    """
    return {
        "tokens_before": visual_tokens_before,
        "tokens_after": visual_tokens_after,
        "reduction_ratio": 1.0 - visual_tokens_after / visual_tokens_before,
        "retention_ratio": visual_tokens_after / visual_tokens_before,
    }
