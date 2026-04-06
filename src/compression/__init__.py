from .base import BaseCompressor
from .fixed_ratio import FixedRatioPruner
from .importance import ImportanceBasedPruner
from .token_merging import TokenMerger

COMPRESSORS = {
    "none": None,
    "fixed_ratio": FixedRatioPruner,
    "importance": ImportanceBasedPruner,
    "token_merging": TokenMerger,
}


def build_compressor(config):
    """Factory function to create a compressor from config."""
    method = config["compression"]["method"]
    if method == "none":
        return None
    cls = COMPRESSORS[method]
    return cls(config["compression"])
