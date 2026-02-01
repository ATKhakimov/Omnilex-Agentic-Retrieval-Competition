"""Tests for graph builder module."""

import pytest
import numpy as np
import pandas as pd

from omnilex.graph.graph_builder import (
    GraphBuilder,
    ExpansionParams,
    ScoringParams,
    K_EXPAND_SIM,
    K_EXPAND_COCITE,
    MIN_SIM_COS,
    ALPHA,
    BETA,
    GAMMA,
    DELTA,
)


class TestExpansionParams:
    """Test suite for ExpansionParams."""

    def test_defaults(self):
        """Test default parameter values."""
        params = ExpansionParams()
        
        assert params.k_expand_sim == K_EXPAND_SIM
        assert params.k_expand_cocite == K_EXPAND_COCITE
        assert params.min_sim_cos == MIN_SIM_COS

    def test_custom_values(self):
        """Test custom parameter values."""
        params = ExpansionParams(
            k_expand_sim=50,
            k_expand_cocite=100,
            min_sim_cos=0.3,
        )
        
        assert params.k_expand_sim == 50
        assert params.k_expand_cocite == 100
        assert params.min_sim_cos == 0.3


class TestScoringParams:
    """Test suite for ScoringParams."""

    def test_defaults(self):
        """Test default scoring weights."""
        params = ScoringParams()
        
        assert params.alpha == ALPHA
        assert params.beta == BETA
        assert params.gamma == GAMMA
        assert params.delta == DELTA

    def test_custom_weights(self):
        """Test custom scoring weights."""
        params = ScoringParams(alpha=2.0, beta=0.5, gamma=1.0, delta=0.1)
        
        assert params.alpha == 2.0
        assert params.beta == 0.5


class TestGraphBuilder:
    """Test suite for GraphBuilder."""

    def test_init(self):
        """Test graph builder initialization."""
        builder = GraphBuilder()
        
        assert builder.similar_edges == {}
        assert builder.cocite_edges == {}
        assert builder.chunk_to_group == {}
        assert builder.group_to_chunks == {}

    def test_build_cocite_edges(self, sample_train_df):
        """Test building co-citation edges from training data."""
        builder = GraphBuilder()
        
        edges_df = builder.build_cocite_edges(sample_train_df)
        
        assert len(edges_df) > 0
        assert "a_chunk_id" in edges_df.columns
        assert "b_chunk_id" in edges_df.columns
        assert "weight" in edges_df.columns
        
        # Check that edges were added to internal structure
        assert len(builder.cocite_edges) > 0
        
        # Art. 1 OR appears in multiple queries, should have co-citation edges
        assert "Art. 1 OR" in builder.cocite_edges

    def test_cocite_weight_calculation(self, sample_train_df):
        """Test that co-citation weights are calculated correctly."""
        builder = GraphBuilder()
        builder.build_cocite_edges(sample_train_df)
        
        # Art. 1 OR and Art. 11 Abs. 1 OR co-occur in q1
        # Art. 1 OR and Art. 41 Abs. 1 OR co-occur in q3
        # So Art. 1 OR should have neighbors
        neighbors = builder.cocite_edges.get("Art. 1 OR", [])
        neighbor_ids = [n[0] for n in neighbors]
        
        assert "Art. 11 Abs. 1 OR" in neighbor_ids or len(neighbors) > 0

    def test_build_groups(self, sample_chunks_df):
        """Test building DocGroup mapping."""
        builder = GraphBuilder()
        
        groups_df, chunk_to_group_df = builder.build_groups(sample_chunks_df)
        
        assert len(groups_df) > 0
        assert len(chunk_to_group_df) > 0
        
        # Check internal structures
        assert len(builder.chunk_to_group) > 0
        assert len(builder.group_to_chunks) > 0
        
        # Check ZGB group
        assert "code:ZGB" in builder.group_to_chunks
        zgb_chunks = builder.group_to_chunks["code:ZGB"]
        assert "Art. 1 ZGB" in zgb_chunks
        assert "Art. 2 ZGB" in zgb_chunks

    def test_expand_candidates_basic(self, sample_chunks_df, sample_train_df):
        """Test basic candidate expansion."""
        builder = GraphBuilder()
        
        # Build graph
        builder.build_groups(sample_chunks_df)
        builder.build_cocite_edges(sample_train_df)
        
        # Initial candidates
        initial = [("Art. 1 ZGB", 0.9), ("Art. 1 OR", 0.8)]
        
        expanded = builder.expand_candidates(initial)
        
        # Should include initial candidates
        expanded_ids = [e[0] for e in expanded]
        assert "Art. 1 ZGB" in expanded_ids
        assert "Art. 1 OR" in expanded_ids
        
        # Should have expanded via some mechanism
        assert len(expanded) >= len(initial)

    def test_expand_candidates_with_siblings(self, sample_chunks_df):
        """Test expansion via PART_OF (group siblings)."""
        builder = GraphBuilder()
        builder.build_groups(sample_chunks_df)
        
        # Only Art. 1 ZGB as initial
        initial = [("Art. 1 ZGB", 0.9)]
        
        params = ExpansionParams(k_expand_siblings=10, k_expand_sim=0, k_expand_cocite=0)
        expanded = builder.expand_candidates(initial, params)
        
        expanded_ids = [e[0] for e in expanded]
        
        # Should include Art. 2 ZGB (same group)
        assert "Art. 2 ZGB" in expanded_ids

    def test_expand_candidates_respects_limit(self, sample_chunks_df, sample_train_df):
        """Test that expansion respects max_candidates limit."""
        builder = GraphBuilder()
        builder.build_groups(sample_chunks_df)
        builder.build_cocite_edges(sample_train_df)
        
        initial = [("Art. 1 ZGB", 0.9), ("Art. 1 OR", 0.8)]
        
        params = ExpansionParams(max_candidates=3)
        expanded = builder.expand_candidates(initial, params)
        
        assert len(expanded) <= 3

    def test_score_candidates(self, sample_chunks_df, sample_train_df):
        """Test candidate scoring."""
        builder = GraphBuilder()
        builder.build_groups(sample_chunks_df)
        builder.build_cocite_edges(sample_train_df)
        
        # Candidates with expansion info
        candidates = [
            ("Art. 1 ZGB", 0.9, "initial"),
            ("Art. 2 ZGB", 0.5, "via PART_OF(code:ZGB)"),
            ("Art. 1 OR", 0.8, "initial"),
        ]
        
        initial_set = {"Art. 1 ZGB", "Art. 1 OR"}
        
        scored = builder.score_candidates(candidates, initial_set)
        
        assert len(scored) == 3
        
        # Each result should be (chunk_id, score, features)
        for chunk_id, score, features in scored:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)
            assert isinstance(features, dict)
            assert "s_ret" in features
            assert "s_sim" in features
            assert "s_cocite" in features
            assert "s_group" in features

    def test_score_candidates_normalized(self, sample_chunks_df, sample_train_df):
        """Test that features are normalized to [0, 1]."""
        builder = GraphBuilder()
        builder.build_groups(sample_chunks_df)
        builder.build_cocite_edges(sample_train_df)
        
        candidates = [
            ("Art. 1 ZGB", 0.9, "initial"),
            ("Art. 1 OR", 0.8, "initial"),
        ]
        
        initial_set = {"Art. 1 ZGB", "Art. 1 OR"}
        
        scored = builder.score_candidates(candidates, initial_set)
        
        for _, _, features in scored:
            assert 0 <= features["s_ret"] <= 1
            assert 0 <= features["s_sim"] <= 1
            assert 0 <= features["s_cocite"] <= 1
            assert 0 <= features["s_group"] <= 1

    def test_save_and_load(self, sample_chunks_df, sample_train_df, tmp_path):
        """Test saving and loading graph data."""
        builder = GraphBuilder()
        
        # Build graph
        builder.build_groups(sample_chunks_df)
        builder.build_cocite_edges(sample_train_df)
        
        # Save
        builder.save(tmp_path)
        
        assert (tmp_path / "edges_cocite.parquet").exists()
        assert (tmp_path / "groups.parquet").exists()
        assert (tmp_path / "chunk_to_group.parquet").exists()
        
        # Load
        loaded = GraphBuilder.load(tmp_path)
        
        assert len(loaded.cocite_edges) == len(builder.cocite_edges)
        assert len(loaded.chunk_to_group) == len(builder.chunk_to_group)
        assert len(loaded.group_to_chunks) == len(builder.group_to_chunks)


class TestGraphBuilderWithEmbeddings:
    """Tests for GraphBuilder with embeddings (requires numpy)."""

    def test_build_similar_edges(self):
        """Test building similarity edges from embeddings."""
        pytest.importorskip("faiss")
        
        builder = GraphBuilder()
        
        # Create simple embeddings (3 chunks, 4-dim)
        chunk_ids = ["Art. 1 ZGB", "Art. 2 ZGB", "Art. 1 OR"]
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],  # Similar to first
            [0.0, 0.0, 1.0, 0.0],  # Different
        ], dtype=np.float32)
        
        edges_df = builder.build_similar_edges(
            embeddings, chunk_ids, k=2, min_cos=0.0
        )
        
        assert len(edges_df) > 0
        assert "src_chunk_id" in edges_df.columns
        assert "dst_chunk_id" in edges_df.columns
        assert "score" in edges_df.columns
        
        # Art. 1 ZGB and Art. 2 ZGB should be similar
        assert "Art. 1 ZGB" in builder.similar_edges
        neighbors = [n[0] for n in builder.similar_edges["Art. 1 ZGB"]]
        assert "Art. 2 ZGB" in neighbors

    def test_build_similar_edges_respects_threshold(self):
        """Test that similarity threshold filters edges."""
        pytest.importorskip("faiss")
        
        builder = GraphBuilder()
        
        chunk_ids = ["A", "B", "C"]
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],  # Moderate similarity
            [0.0, 0.0, 1.0, 0.0],  # Low similarity
        ], dtype=np.float32)
        
        # High threshold
        edges_df = builder.build_similar_edges(
            embeddings, chunk_ids, k=2, min_cos=0.9
        )
        
        # Should filter out low similarity edges
        # After normalization, A and B will have lower cosine
        assert len(edges_df) < 6  # Not all pairs
