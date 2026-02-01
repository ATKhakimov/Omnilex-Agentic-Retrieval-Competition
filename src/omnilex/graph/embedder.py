"""Embedder: Build embeddings and search indices.

Input: summaries.parquet (summary_type=retrieval)
Output: embeddings.npy, faiss_index.bin, bm25_index.pkl
"""

import pickle
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from rank_bm25 import BM25Okapi


# Default embedding model (multilingual, good for DE/FR/IT)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    """Build embeddings and search indices for summaries."""
    
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str | None = None,
    ):
        """Initialize embedder.
        
        Args:
            model_name: Sentence transformer model name
            device: Device for embedding ("cpu", "cuda", or None for auto)
        """
        self.model_name = model_name
        self.device = device
        
        self._model: SentenceTransformer | None = None
        self._chunk_ids: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._faiss_index = None
        self._bm25_index: BM25Okapi | None = None
        self._bm25_corpus: list[list[str]] = []
    
    def _load_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model
    
    def build_embeddings(
        self,
        summaries_df: pd.DataFrame,
        summary_type: str = "retrieval",
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Build embeddings for summaries.
        
        Args:
            summaries_df: DataFrame with chunk_id, summary_type, summary_text
            summary_type: Which summary type to embed (default: retrieval)
            batch_size: Batch size for embedding
            show_progress: Whether to show progress bar
            
        Returns:
            Embeddings array of shape (N, embedding_dim)
        """
        # Filter to requested summary type
        df = summaries_df[summaries_df["summary_type"] == summary_type].copy()
        df = df.drop_duplicates(subset=["chunk_id"])
        
        self._chunk_ids = df["chunk_id"].tolist()
        texts = df["summary_text"].tolist()
        
        # Embed
        model = self._load_model()
        self._embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        
        return self._embeddings
    
    def build_faiss_index(
        self,
        embeddings: np.ndarray | None = None,
        index_type: Literal["flat", "ivf", "hnsw"] = "flat",
        nlist: int = 100,
    ):
        """Build FAISS index for vector search.
        
        Args:
            embeddings: Embeddings array (uses cached if None)
            index_type: Type of FAISS index
            nlist: Number of clusters for IVF index
        """
        if not HAS_FAISS:
            raise ImportError("faiss required. Install with: pip install faiss-cpu")
        
        if embeddings is None:
            embeddings = self._embeddings
        if embeddings is None:
            raise ValueError("No embeddings available. Run build_embeddings first.")
        
        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]
        
        if index_type == "flat":
            self._faiss_index = faiss.IndexFlatIP(dim)  # Inner product (cosine after norm)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            self._faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self._faiss_index.train(embeddings)
        elif index_type == "hnsw":
            self._faiss_index = faiss.IndexHNSWFlat(dim, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self._faiss_index.add(embeddings)
    
    def build_bm25_index(
        self,
        summaries_df: pd.DataFrame,
        summary_type: str = "retrieval",
    ) -> None:
        """Build BM25 index for keyword search.
        
        Args:
            summaries_df: DataFrame with chunk_id, summary_type, summary_text
            summary_type: Which summary type to index
        """
        df = summaries_df[summaries_df["summary_type"] == summary_type].copy()
        df = df.drop_duplicates(subset=["chunk_id"])
        
        # Tokenize
        self._bm25_corpus = []
        for text in df["summary_text"]:
            tokens = self._tokenize(text)
            self._bm25_corpus.append(tokens)
        
        self._bm25_index = BM25Okapi(self._bm25_corpus)
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        text = text.lower()
        tokens = re.split(r"\W+", text)
        return [t for t in tokens if t and len(t) > 1]
    
    def search_vector(
        self,
        query: str,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Search using vector similarity.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of (chunk_id, score) tuples
        """
        if self._faiss_index is None:
            raise ValueError("FAISS index not built. Run build_faiss_index first.")
        
        # Embed query
        model = self._load_model()
        query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vec)
        
        # Search
        scores, indices = self._faiss_index.search(query_vec, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self._chunk_ids):
                results.append((self._chunk_ids[idx], float(score)))
        
        return results
    
    def search_bm25(
        self,
        query: str,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Search using BM25.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of (chunk_id, score) tuples
        """
        if self._bm25_index is None:
            raise ValueError("BM25 index not built. Run build_bm25_index first.")
        
        tokens = self._tokenize(query)
        if not tokens:
            return []
        
        scores = self._bm25_index.get_scores(tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._chunk_ids[idx], float(scores[idx])))
        
        return results
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = 50,
        top_k_each: int = 100,
        rrf_k: int = 60,
    ) -> list[tuple[str, float]]:
        """Hybrid search using reciprocal rank fusion.
        
        Args:
            query: Query text
            top_k: Final number of results
            top_k_each: Results from each method
            rrf_k: RRF constant (higher = less weight to top ranks)
            
        Returns:
            List of (chunk_id, score) tuples
        """
        # Get results from both methods
        vec_results = self.search_vector(query, top_k=top_k_each)
        bm25_results = self.search_bm25(query, top_k=top_k_each)
        
        # Compute RRF scores
        rrf_scores: dict[str, float] = {}
        
        for rank, (chunk_id, _) in enumerate(vec_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rrf_k + rank + 1)
        
        for rank, (chunk_id, _) in enumerate(bm25_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rrf_k + rank + 1)
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results[:top_k]
    
    def get_chunk_ids(self) -> list[str]:
        """Get list of indexed chunk IDs."""
        return self._chunk_ids.copy()
    
    def get_id_to_idx(self) -> dict[str, int]:
        """Get mapping from chunk_id to index."""
        return {cid: idx for idx, cid in enumerate(self._chunk_ids)}
    
    def save(self, output_dir: Path | str) -> None:
        """Save all indices to directory.
        
        Saves:
        - embeddings.npy
        - chunk_ids.pkl
        - faiss_index.bin (if built)
        - bm25_index.pkl (if built)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if self._embeddings is not None:
            np.save(output_dir / "embeddings.npy", self._embeddings)
        
        # Save chunk IDs
        with open(output_dir / "chunk_ids.pkl", "wb") as f:
            pickle.dump(self._chunk_ids, f)
        
        # Save FAISS index
        if self._faiss_index is not None and HAS_FAISS:
            faiss.write_index(self._faiss_index, str(output_dir / "faiss_index.bin"))
        
        # Save BM25 index
        if self._bm25_index is not None:
            with open(output_dir / "bm25_index.pkl", "wb") as f:
                pickle.dump({
                    "index": self._bm25_index,
                    "corpus": self._bm25_corpus,
                }, f)
    
    @classmethod
    def load(cls, input_dir: Path | str, model_name: str = EMBEDDING_MODEL) -> "Embedder":
        """Load embedder from saved files.
        
        Args:
            input_dir: Directory with saved indices
            model_name: Model name for embedding new queries
            
        Returns:
            Loaded Embedder instance
        """
        input_dir = Path(input_dir)
        embedder = cls(model_name=model_name)
        
        # Load chunk IDs
        with open(input_dir / "chunk_ids.pkl", "rb") as f:
            embedder._chunk_ids = pickle.load(f)
        
        # Load embeddings
        embeddings_path = input_dir / "embeddings.npy"
        if embeddings_path.exists():
            embedder._embeddings = np.load(embeddings_path)
        
        # Load FAISS index
        faiss_path = input_dir / "faiss_index.bin"
        if faiss_path.exists() and HAS_FAISS:
            embedder._faiss_index = faiss.read_index(str(faiss_path))
        
        # Load BM25 index
        bm25_path = input_dir / "bm25_index.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
                embedder._bm25_index = data["index"]
                embedder._bm25_corpus = data["corpus"]
        
        return embedder
