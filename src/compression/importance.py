"""Importance-based token pruning (attention / magnitude / similarity)."""

import torch
from .base import BaseCompressor


class ImportanceBasedPruner(BaseCompressor):
    """Prune visual tokens based on an importance score, keeping top-k."""

    def __init__(self, config):
        super().__init__(config)
        self.signal = config.get("importance_signal", "attention")
        self.pruning_layer = config.get("pruning_layer", 2)

    def compress(self, visual_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Score each visual token and keep the top-k most important.

        Args:
            visual_tokens: (batch, num_tokens, hidden_dim)
            kwargs:
                attention_weights: (batch, num_heads, seq_len, seq_len) — required for 'attention' signal
                visual_token_mask: (batch, seq_len) — boolean mask indicating visual token positions

        Returns:
            Pruned visual tokens: (batch, k, hidden_dim)
        """
        batch, num_tokens, dim = visual_tokens.shape
        k = self._num_tokens_to_keep(num_tokens)

        scores = self._compute_scores(visual_tokens, **kwargs)
        # scores: (batch, num_tokens)
        _, topk_indices = scores.topk(k, dim=-1, sorted=True)
        topk_indices, _ = topk_indices.sort(dim=-1)  # preserve spatial order

        return visual_tokens.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, dim))

    def _compute_scores(self, visual_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.signal == "attention":
            return self._attention_score(visual_tokens, **kwargs)
        elif self.signal == "magnitude":
            return visual_tokens.norm(dim=-1)
        elif self.signal == "similarity":
            return self._similarity_score(visual_tokens)
        else:
            raise ValueError(f"Unknown importance signal: {self.signal}")

    def _attention_score(self, visual_tokens, **kwargs):
        """Average attention received by each visual token from text tokens."""
        attn = kwargs.get("attention_weights")
        visual_mask = kwargs.get("visual_token_mask")
        if attn is None or visual_mask is None:
            # Fallback to magnitude
            return visual_tokens.norm(dim=-1)

        # attn: (batch, heads, seq, seq) -> average over heads
        attn_avg = attn.mean(dim=1)  # (batch, seq, seq)
        # Sum attention from text tokens to visual tokens
        text_mask = ~visual_mask
        # text->visual attention: (batch, num_text, num_visual)
        text_to_visual = attn_avg[:, text_mask[0], :][:, :, visual_mask[0]]
        scores = text_to_visual.sum(dim=1)  # (batch, num_visual)
        return scores

    def _similarity_score(self, visual_tokens):
        """Score by average cosine similarity to all other tokens (low = unique = important)."""
        normed = torch.nn.functional.normalize(visual_tokens, dim=-1)
        sim_matrix = torch.bmm(normed, normed.transpose(1, 2))  # (B, N, N)
        avg_sim = sim_matrix.mean(dim=-1)  # (B, N)
        # Lower similarity = more unique = higher importance
        return -avg_sim
