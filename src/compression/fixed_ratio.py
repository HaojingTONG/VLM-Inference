"""Fixed-ratio token pruning: keep a fixed percentage of visual tokens."""

import torch
from .base import BaseCompressor


class FixedRatioPruner(BaseCompressor):
    """Uniformly sample a fixed ratio of visual tokens (no importance signal)."""

    def compress(self, visual_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Keep evenly-spaced tokens to reach the target retention ratio.

        Args:
            visual_tokens: (batch, num_tokens, hidden_dim)

        Returns:
            Pruned visual tokens: (batch, k, hidden_dim)
        """
        batch, num_tokens, dim = visual_tokens.shape
        k = self._num_tokens_to_keep(num_tokens)

        # Evenly spaced indices
        indices = torch.linspace(0, num_tokens - 1, k, dtype=torch.long, device=visual_tokens.device)
        return visual_tokens[:, indices, :]
