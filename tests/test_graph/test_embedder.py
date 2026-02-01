"""Tests for graph embedder module."""

import pytest
import numpy as np
import pandas as pd

from omnilex.graph.embedder import Embedder, EMBEDDING_MODEL


class TestEmbedder:
    """Test suite for Embedder."""

    def test_init(self):
        """Test embedder initialization."""
        embedder = Embedder()
        assert embedder.model_name == EMBEDDING_MODEL
        assert embedder._model is None  # Lazy loading

    def test_init_custom_model(self):
        """Test embedder with custom model name."""
        custom_model = "sentence-transformers/all-MiniLM-L6-v2"
        embedder = Embedder(model_name=custom_model)
        assert embedder.model_name == custom_model

    def test_tokenize(self):
        """Test BM25 tokenization."""
        embedder = Embedder()
        
        tokens = embedder._tokenize("Das Bundesgericht hat entschieden.")
        assert "bundesgericht" in tokens
        assert "hat" in tokens
        assert len(tokens) == 4

    def test_tokenize_filters_short(self):
        """Test that short tokens are filtered."""
        embedder = Embedder()
        
        tokens = embedder._tokenize("I am a test")
        assert "i" not in tokens  # Single char filtered
        assert "am" in tokens

    @pytest.mark.slow
    def test_build_embeddings(self, sample_summaries_df):
        """Test building embeddings from summaries."""
        embedder = Embedder()
        
        embeddings = embedder.build_embeddings(
            sample_summaries_df,
            summary_type="retrieval",
            show_progress=False,
        )
        
        # Should have one embedding per unique chunk_id
        unique_chunks = sample_summaries_df[
            sample_summaries_df["summary_type"] == "retrieval"
        ]["chunk_id"].nunique()
        
        assert embeddings.shape[0] == unique_chunks
        assert embeddings.shape[1] > 0  # Has embedding dimension

    def test_build_bm25_index(self, sample_summaries_df):
        """Test building BM25 index."""
        embedder = Embedder()
        embedder._chunk_ids = sample_summaries_df[
            sample_summaries_df["summary_type"] == "retrieval"
        ]["chunk_id"].tolist()
        
        embedder.build_bm25_index(sample_summaries_df, summary_type="retrieval")
        
        assert embedder._bm25_index is not None
        assert len(embedder._bm25_corpus) > 0

    def test_search_bm25(self, sample_summaries_df):
        """Test BM25 search."""
        embedder = Embedder()
        embedder._chunk_ids = sample_summaries_df[
            sample_summaries_df["summary_type"] == "retrieval"
        ]["chunk_id"].drop_duplicates().tolist()
        
        embedder.build_bm25_index(sample_summaries_df, summary_type="retrieval")
        
        results = embedder.search_bm25("legal law contract", top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Each result should be (chunk_id, score)
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)

    def test_search_bm25_empty_query(self, sample_summaries_df):
        """Test BM25 search with empty query."""
        embedder = Embedder()
        embedder._chunk_ids = sample_summaries_df[
            sample_summaries_df["summary_type"] == "retrieval"
        ]["chunk_id"].drop_duplicates().tolist()
        
        embedder.build_bm25_index(sample_summaries_df, summary_type="retrieval")
        
        results = embedder.search_bm25("", top_k=3)
        assert results == []

    def test_get_chunk_ids(self, sample_summaries_df):
        """Test getting indexed chunk IDs."""
        embedder = Embedder()
        embedder._chunk_ids = ["Art. 1 ZGB", "Art. 2 ZGB"]
        
        chunk_ids = embedder.get_chunk_ids()
        assert chunk_ids == ["Art. 1 ZGB", "Art. 2 ZGB"]
        
        # Should be a copy
        chunk_ids.append("Art. 3 ZGB")
        assert len(embedder._chunk_ids) == 2

    def test_get_id_to_idx(self):
        """Test getting ID to index mapping."""
        embedder = Embedder()
        embedder._chunk_ids = ["Art. 1 ZGB", "Art. 2 ZGB", "Art. 11 OR"]
        
        mapping = embedder.get_id_to_idx()
        
        assert mapping["Art. 1 ZGB"] == 0
        assert mapping["Art. 2 ZGB"] == 1
        assert mapping["Art. 11 OR"] == 2


class TestEmbedderIntegration:
    """Integration tests for Embedder (requires model download)."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_workflow(self, sample_summaries_df, tmp_path):
        """Test full embedding workflow: build, search, save, load."""
        embedder = Embedder()
        
        # Build embeddings
        embeddings = embedder.build_embeddings(
            sample_summaries_df,
            summary_type="retrieval",
            show_progress=False,
        )
        
        # Build indices
        embedder.build_faiss_index(embeddings)
        embedder.build_bm25_index(sample_summaries_df, summary_type="retrieval")
        
        # Search
        vec_results = embedder.search_vector("law contract", top_k=2)
        bm25_results = embedder.search_bm25("law contract", top_k=2)
        hybrid_results = embedder.search_hybrid("law contract", top_k=2)
        
        assert len(vec_results) <= 2
        assert len(bm25_results) <= 2
        assert len(hybrid_results) <= 2
        
        # Save
        embedder.save(tmp_path)
        
        assert (tmp_path / "embeddings.npy").exists()
        assert (tmp_path / "chunk_ids.pkl").exists()
        assert (tmp_path / "bm25_index.pkl").exists()
        
        # Load
        loaded = Embedder.load(tmp_path)
        
        assert len(loaded.get_chunk_ids()) == len(embedder.get_chunk_ids())
        
        # Search on loaded embedder
        loaded_results = loaded.search_bm25("law contract", top_k=2)
        assert len(loaded_results) == len(bm25_results)
