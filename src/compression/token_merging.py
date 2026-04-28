"""Token merging: merge similar visual tokens instead of dropping them."""

import torch
import torch.nn.functional as F

from .base import BaseCompressor


class TokenMerger(BaseCompressor):
    """ToMe-style bipartite token merging.

    Each pass:
      1. Splits tokens into a (even-indexed) "source" set and b (odd-indexed)
         "destination" set.
      2. For each token in a, finds its most similar partner in b.
      3. Picks the top-r src tokens with the highest similarity to merge,
         where r = min(num_to_remove, |a|).
      4. Each merged src is averaged into its best dst (running mean weighted
         by the number of tokens fused so far).

    A single pass can remove at most |a| ~= n/2 tokens, so for aggressive
    compression (retention < 50%) we run multiple passes. The total cost is
    O(log(n/k)) GPU passes instead of O(n - k) Python iterations.
    """

    def __init__(self, config):
        super().__init__(config)
        self.merge_method = config.get("merge_method", "mean")
        self.similarity_metric = config.get("similarity_metric", "cosine")

    def compress(self, visual_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Iteratively merge tokens via vectorized bipartite matching.

        Args:
            visual_tokens: (batch, num_tokens, hidden_dim)

        Returns:
            (batch, k, hidden_dim) where k = retention_ratio * num_tokens.
        """
        batch, num_tokens, _ = visual_tokens.shape
        k = self._num_tokens_to_keep(num_tokens)
        if k >= num_tokens:
            return visual_tokens

        tokens = visual_tokens
        while tokens.shape[1] > k:
            target_remove = tokens.shape[1] - k
            tokens = self._bipartite_merge(tokens, target_remove)
        return tokens

    def _bipartite_merge(self, tokens: torch.Tensor, num_to_remove: int) -> torch.Tensor:
        """One vectorized ToMe pass that removes up to |a| tokens at once."""
        B, n, D = tokens.shape

        # Even / odd split
        a = tokens[:, 0::2, :]  # (B, |a|, D)
        b = tokens[:, 1::2, :]  # (B, |b|, D)
        len_a = a.shape[1]
        len_b = b.shape[1]

        # Cap removal at |a|; one pass can't remove more than the size of a.
        r = min(num_to_remove, len_a)
        if r <= 0:
            return tokens

        # Pairwise similarity a -> b
        if self.similarity_metric == "cosine":
            a_n = F.normalize(a, dim=-1)
            b_n = F.normalize(b, dim=-1)
            sim = torch.bmm(a_n, b_n.transpose(1, 2))  # (B, |a|, |b|)
        else:  # dot
            sim = torch.bmm(a, b.transpose(1, 2))

        # For each src in a, best dst index in b and the similarity score
        sim_max, best_dst = sim.max(dim=-1)  # (B, |a|), (B, |a|)

        # Pick the top-r src indices (highest similarity to their best dst)
        sorted_score, sorted_idx = sim_max.sort(dim=-1, descending=True)
        merge_src_idx = sorted_idx[:, :r]  # (B, r)
        keep_src_idx = sorted_idx[:, r:]   # (B, |a| - r)

        # For each merged src, look up its destination in b
        merge_dst_idx = torch.gather(best_dst, 1, merge_src_idx)  # (B, r)

        # Gather the actual src embeddings to merge
        src_embeds = torch.gather(
            a, 1, merge_src_idx.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, r, D)

        # Running mean: each b token starts as itself with weight 1; each merged
        # src adds its embedding and weight 1 to the chosen dst, then we divide.
        b_sum = b.clone()
        b_sum.scatter_add_(
            1, merge_dst_idx.unsqueeze(-1).expand(-1, -1, D), src_embeds
        )
        b_count = torch.ones(B, len_b, device=b.device, dtype=b.dtype)
        ones_r = torch.ones(B, r, device=b.device, dtype=b.dtype)
        b_count.scatter_add_(1, merge_dst_idx, ones_r)
        b_avg = b_sum / b_count.unsqueeze(-1)

        # Surviving src tokens (not merged)
        keep_src = torch.gather(
            a, 1, keep_src_idx.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, |a| - r, D)

        # Concatenate surviving src + averaged dst
        return torch.cat([keep_src, b_avg], dim=1)
