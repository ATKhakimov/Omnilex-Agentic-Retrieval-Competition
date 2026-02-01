"""Graph-augmented retrieval for Swiss legal citations.

Modules:
- ingestor: Load and normalize corpus data into SourceChunks
- summarizer: Generate summaries for chunks (LLM or heuristic)
- embedder: Build embeddings and FAISS/BM25 indices
- graph_builder: Build SIMILAR_TO, CO_CITED_WITH, PART_OF edges
- reranker: LLM-based reranking for final candidate selection
"""

from .ingestor import Ingestor, SourceChunk
from .summarizer import Summarizer, Summary, create_llm_client, create_openai_client
from .embedder import Embedder
from .graph_builder import GraphBuilder
from .reranker import LLMReranker, RerankerConfig, QueryPreprocessor

__all__ = [
    "Ingestor",
    "SourceChunk",
    "Summarizer",
    "Summary",
    "create_llm_client",
    "create_openai_client",
    "Embedder",
    "GraphBuilder",
    "LLMReranker",
    "RerankerConfig",
    "QueryPreprocessor",
]
