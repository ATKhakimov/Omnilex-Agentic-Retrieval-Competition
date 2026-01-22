"""Citation parsing and normalization for Swiss legal citations."""

from .normalizer import CitationNormalizer
from .types import Citation, CitationType

__all__ = ["Citation", "CitationType", "CitationNormalizer"]
