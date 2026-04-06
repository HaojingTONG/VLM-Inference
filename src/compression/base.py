"""Base class for visual token compressors."""

from abc import ABC, abstractmethod
import torch


class BaseCompressor(ABC):
    """Abstract base class for all visual token compression strategies."""

    def __init__(self, config):
        self.retention_ratio = config.get("retention_ratio", 0.5)

    @abstractmethod
    def compress(self, visual_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compress visual tokens.

        Args:
            visual_tokens: Tensor of shape (batch, num_visual_tokens, hidden_dim).
            **kwargs: Additional context (e.g., attention weights).

        Returns:
            Compressed visual tokens tensor.
        """
        ...

    def _num_tokens_to_keep(self, num_tokens: int) -> int:
        return max(1, int(num_tokens * self.retention_ratio))
