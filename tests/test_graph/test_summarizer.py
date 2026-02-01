"""Tests for graph summarizer module."""

import pytest
import pandas as pd

from omnilex.graph.summarizer import Summarizer, SummaryType


class TestSummarizer:
    """Test suite for Summarizer."""

    def test_init_heuristic_mode(self):
        """Test initializing summarizer in heuristic mode."""
        summarizer = Summarizer(mode="heuristic")
        assert summarizer.mode == "heuristic"

    def test_init_llm_mode_requires_client(self):
        """Test that LLM mode requires a client."""
        with pytest.raises(ValueError, match="llm_client required"):
            Summarizer(mode="llm")

    def test_init_llm_mode_with_client(self):
        """Test initializing summarizer with LLM client."""
        mock_client = lambda x: "Mock response"
        summarizer = Summarizer(mode="llm", llm_client=mock_client)
        assert summarizer.mode == "llm"

    def test_heuristic_short_summary(self):
        """Test heuristic short summary generation."""
        summarizer = Summarizer(mode="heuristic")
        
        text = "Das Bundesgericht hat festgestellt. Der Beschwerdeführer hat die Frist versäumt. Die Beschwerde wird abgewiesen."
        summary = summarizer._heuristic_short(text)
        
        assert summary.summary_type == SummaryType.SHORT
        assert len(summary.summary_text) > 0
        assert len(summary.summary_text) <= 500  # Should be bounded

    def test_heuristic_retrieval_summary(self):
        """Test heuristic retrieval summary generation."""
        summarizer = Summarizer(mode="heuristic")
        
        text = "Zum Abschlusse eines Vertrages ist die übereinstimmende gegenseitige Willensäusserung der Parteien erforderlich. Art. 1 OR regelt die Grundlagen. BGE 119 II 449 enthält wichtige Präzedenzfälle."
        summary = summarizer._heuristic_retrieval(text)
        
        assert summary.summary_type == SummaryType.RETRIEVAL
        assert "Topic:" in summary.summary_text or len(summary.summary_text) > 0

    def test_heuristic_entities_summary(self):
        """Test heuristic entities extraction."""
        summarizer = Summarizer(mode="heuristic")
        
        text = "Art. 1 ZGB regelt die Rechtsfähigkeit. Gemäss BGE 116 Ia 56 E. 2b ist der Vertrag nach Art. 11 Abs. 2 OR gültig."
        summary = summarizer._heuristic_entities(text)
        
        assert summary.summary_type == SummaryType.ENTITIES
        assert len(summary.entities) > 0
        # Should extract legal references
        assert any("Art." in e for e in summary.entities) or any("BGE" in e for e in summary.entities)

    def test_extract_keyphrases(self):
        """Test keyphrase extraction."""
        summarizer = Summarizer(mode="heuristic")
        
        text = "Vertrag Vertrag Vertrag Rechtsfähigkeit Rechtsfähigkeit Person Gesetz"
        keyphrases = summarizer._extract_keyphrases(text, top_k=3)
        
        assert len(keyphrases) <= 3
        assert "vertrag" in keyphrases  # Most frequent

    def test_extract_legal_references(self):
        """Test legal reference extraction."""
        summarizer = Summarizer(mode="heuristic")
        
        text = "Gemäss Art. 1 ZGB und Art. 11 Abs. 2 OR sowie BGE 116 Ia 56 E. 2b ist der Sachverhalt klar."
        refs = summarizer._extract_legal_references(text)
        
        assert len(refs) > 0
        assert any("Art." in r for r in refs)

    def test_summarize_all(self, sample_chunks_df):
        """Test generating all summaries for all chunks."""
        summarizer = Summarizer(mode="heuristic")
        
        summaries_df = summarizer.summarize_all(sample_chunks_df)
        
        # Should have 3 summary types per chunk
        assert len(summaries_df) == len(sample_chunks_df) * 3
        
        # Check all summary types present
        assert set(summaries_df["summary_type"].unique()) == {"short", "retrieval", "entities"}
        
        # Check required columns
        assert "chunk_id" in summaries_df.columns
        assert "summary_type" in summaries_df.columns
        assert "summary_text" in summaries_df.columns
        assert "entities" in summaries_df.columns

    def test_save_and_load(self, sample_chunks_df, tmp_path):
        """Test saving and loading summaries."""
        summarizer = Summarizer(mode="heuristic")
        summaries_df = summarizer.summarize_all(sample_chunks_df)
        
        output_path = tmp_path / "summaries.parquet"
        summarizer.save(summaries_df, output_path)
        
        assert output_path.exists()
        
        loaded_df = Summarizer.load_summaries(output_path)
        assert len(loaded_df) == len(summaries_df)

    def test_llm_mode_generates_summary(self):
        """Test LLM mode summary generation."""
        responses = {
            "short": "This is a short summary.",
            "retrieval": "Topic: Contract law | Terms: agreement, offer",
            "entities": "contract; law; agreement; ZGB",
        }
        
        def mock_client(prompt):
            if "1-2 sentences" in prompt:
                return responses["short"]
            elif "structured retrieval" in prompt:
                return responses["retrieval"]
            else:
                return responses["entities"]
        
        summarizer = Summarizer(mode="llm", llm_client=mock_client)
        
        summary = summarizer._generate_llm("Some legal text", SummaryType.SHORT)
        assert summary.summary_text == responses["short"]
        
        summary = summarizer._generate_llm("Some legal text", SummaryType.ENTITIES)
        assert len(summary.entities) > 0
