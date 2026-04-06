"""Token merging: merge similar visual tokens instead of dropping them."""

import torch
from .base import BaseCompressor


class TokenMerger(BaseCompressor):
    """Merge similar visual tokens via bipartite matching (ToMe-style)."""

    def __init__(self, config):
        super().__init__(config)
        self.merge_method = config.get("merge_method", "mean")
        self.similarity_metric = config.get("similarity_metric", "cosine")

    def compress(self, visual_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Iteratively merge the most similar token pairs until target count.

        Args:
            visual_tokens: (batch, num_tokens, hidden_dim)

        Returns:
            Merged visual tokens: (batch, k, hidden_dim)
        """
        batch, num_tokens, dim = visual_tokens.shape
        k = self._num_tokens_to_keep(num_tokens)
        num_merges = num_tokens - k

        tokens = visual_tokens.clone()

        for _ in range(num_merges):
            tokens = self._merge_step(tokens)

        return tokens

    def _merge_step(self, tokens: torch.Tensor) -> torch.Tensor:
        """Single merge step: find the most similar adjacent pair and merge."""
        batch, n, dim = tokens.shape

        # ToMe-style bipartite: split into two sets (even/odd indices)
        a = tokens[:, 0::2, :]  # source set
        b = tokens[:, 1::2, :]  # destination set

        # Compute similarity between a and b
        if self.similarity_metric == "cosine":
            a_norm = torch.nn.functional.normalize(a, dim=-1)
            b_norm = torch.nn.functional.normalize(b, dim=-1)
            sim = torch.bmm(a_norm, b_norm.transpose(1, 2))  # (B, |A|, |B|)
        else:
            sim = torch.bmm(a, b.transpose(1, 2))

        # For each token in a, find the most similar token in b
        max_sim, max_idx = sim.max(dim=-1)  # (B, |A|)

        # Find the single most similar pair across all of a
        _, merge_src = max_sim.max(dim=-1)  # (B,)

        result_list = []
        for bi in range(batch):
            src_idx = merge_src[bi].item()
            dst_idx = max_idx[bi, src_idx].item()

            merged = (a[bi, src_idx] + b[bi, dst_idx]) / 2.0

            # Rebuild: keep all tokens except the merged pair, add merged token
            keep_a = [i for i in range(a.shape[1]) if i != src_idx]
            keep_b = [i for i in range(b.shape[1]) if i != dst_idx]

            parts = []
            if keep_a:
                parts.append(a[bi, keep_a])
            if keep_b:
                parts.append(b[bi, keep_b])
            parts.append(merged.unsqueeze(0))
            result_list.append(torch.cat(parts, dim=0))

        return torch.stack(result_list, dim=0)
