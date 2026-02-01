"""Tests for graph ingestor module."""

import pytest
import pandas as pd

from omnilex.graph.ingestor import Ingestor, ChunkType, Language


class TestIngestor:
    """Test suite for Ingestor."""

    def test_load_laws(self, sample_data_dir):
        """Test loading laws corpus."""
        ingestor = Ingestor(sample_data_dir)
        df = ingestor._load_laws()
        
        assert len(df) == 5
        assert all(df["chunk_type"] == ChunkType.LAW.value)
        assert "Art. 1 ZGB" in df["chunk_id"].values
        assert "Art. 11 Abs. 1 OR" in df["chunk_id"].values

    def test_load_courts(self, sample_data_dir):
        """Test loading courts corpus."""
        ingestor = Ingestor(sample_data_dir)
        df = ingestor._load_courts()
        
        assert len(df) == 5
        assert all(df["chunk_type"] == ChunkType.CASE.value)
        assert "BGE 116 Ia 56 E. 2b" in df["chunk_id"].values
        assert "5A_800/2019 E. 2" in df["chunk_id"].values

    def test_load_all(self, sample_data_dir):
        """Test loading all corpus data."""
        ingestor = Ingestor(sample_data_dir)
        df = ingestor.load_all()
        
        assert len(df) == 10
        assert "chunk_id" in df.columns
        assert "chunk_type" in df.columns
        assert "group_id" in df.columns
        assert "lang" in df.columns
        assert "text_raw" in df.columns

    def test_normalize_text(self, sample_data_dir):
        """Test text normalization."""
        ingestor = Ingestor(sample_data_dir)
        
        assert ingestor._normalize_text("  hello   world  ") == "hello world"
        assert ingestor._normalize_text("test\n\ntext") == "test text"

    def test_detect_language_german(self, sample_data_dir):
        """Test German language detection."""
        ingestor = Ingestor(sample_data_dir)
        
        german_text = "Das Bundesgericht hat festgestellt, dass der Vertrag gültig ist und die Parteien sich einig sind."
        assert ingestor._detect_language(german_text) == Language.DE

    def test_detect_language_french(self, sample_data_dir):
        """Test French language detection."""
        ingestor = Ingestor(sample_data_dir)
        
        french_text = "Le recourant fait valoir une violation de son droit d'être entendu dans la procédure."
        assert ingestor._detect_language(french_text) == Language.FR

    def test_detect_language_italian(self, sample_data_dir):
        """Test Italian language detection."""
        ingestor = Ingestor(sample_data_dir)
        
        italian_text = "Il ricorrente contesta la valutazione delle prove effettuata dall'autorità cantonale nel suo giudizio."
        assert ingestor._detect_language(italian_text) == Language.IT

    def test_extract_group_id_law(self, sample_data_dir):
        """Test group ID extraction for laws."""
        ingestor = Ingestor(sample_data_dir)
        
        assert ingestor._extract_group_id("Art. 1 ZGB", ChunkType.LAW) == "code:ZGB"
        assert ingestor._extract_group_id("Art. 11 Abs. 2 OR", ChunkType.LAW) == "code:OR"
        assert ingestor._extract_group_id("Art. 117 StGB", ChunkType.LAW) == "code:StGB"

    def test_extract_group_id_bge(self, sample_data_dir):
        """Test group ID extraction for BGE decisions."""
        ingestor = Ingestor(sample_data_dir)
        
        assert ingestor._extract_group_id("BGE 116 Ia 56 E. 2b", ChunkType.CASE) == "bge:116_Ia_56"
        assert ingestor._extract_group_id("BGE 145 II 32 E. 3.1", ChunkType.CASE) == "bge:145_II_32"

    def test_extract_group_id_docket(self, sample_data_dir):
        """Test group ID extraction for docket-style decisions."""
        ingestor = Ingestor(sample_data_dir)
        
        assert ingestor._extract_group_id("5A_800/2019 E. 2", ChunkType.CASE) == "docket:5A_800_2019"

    def test_unique_chunk_ids(self, sample_data_dir):
        """Test that all chunk IDs are unique after loading."""
        ingestor = Ingestor(sample_data_dir)
        df = ingestor.load_all()
        
        assert df["chunk_id"].nunique() == len(df)

    def test_save_and_load(self, sample_data_dir, tmp_path):
        """Test saving and loading chunks."""
        ingestor = Ingestor(sample_data_dir)
        df = ingestor.load_all()
        
        output_path = tmp_path / "chunks.parquet"
        ingestor.save(df, output_path)
        
        assert output_path.exists()
        
        loaded_df = Ingestor.load_chunks(output_path)
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)
