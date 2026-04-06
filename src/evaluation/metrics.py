"""Quality metrics for evaluating VLM outputs."""


def compute_vqa_accuracy(predictions, references):
    """Simple exact-match VQA accuracy.

    Args:
        predictions: List of predicted answer strings.
        references: List of ground-truth answer strings.

    Returns:
        Accuracy as a float in [0, 1].
    """
    correct = sum(
        1 for pred, ref in zip(predictions, references)
        if pred.strip().lower() == ref.strip().lower()
    )
    return correct / len(predictions) if predictions else 0.0


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
