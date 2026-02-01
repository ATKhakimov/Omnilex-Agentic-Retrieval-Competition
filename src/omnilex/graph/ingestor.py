"""Ingestor: Load and normalize corpus data into SourceChunks.

Input: laws_de.csv, court_considerations.csv
Output: chunks.parquet
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm


class ChunkType(Enum):
    """Type of source chunk."""
    LAW = "law"
    CASE = "case"


class Language(Enum):
    """Detected language of chunk text."""
    DE = "de"
    FR = "fr"
    IT = "it"
    UNKNOWN = "unknown"


@dataclass
class SourceChunk:
    """A single source chunk from the corpus.
    
    1:1 mapping to corpus rows (laws_de.csv or court_considerations.csv).
    """
    chunk_id: str           # = citation (canonical, unique)
    chunk_type: ChunkType
    group_id: str | None    # DocGroup FK (e.g., "code:OR", "bge:145_II_32")
    lang: Language
    text_raw: str


# Language detection stopwords
_DE_STOPWORDS = {"der", "die", "das", "und", "in", "ist", "von", "den", "mit", "zu", "des", "für", "nicht", "werden", "kann", "auf", "ein", "eine", "einer", "eines", "wird", "sind", "hat", "nach", "bei", "auch", "wenn", "oder", "als", "aus", "dem", "dass", "im", "sie", "es"}
_FR_STOPWORDS = {"le", "la", "les", "de", "du", "des", "un", "une", "et", "est", "en", "que", "qui", "dans", "pour", "par", "sur", "avec", "ce", "il", "ne", "pas", "au", "aux", "son", "sa", "ses", "être", "avoir", "sont", "cette", "ont"}
_IT_STOPWORDS = {"il", "la", "le", "di", "del", "della", "dei", "un", "una", "e", "è", "in", "che", "per", "con", "non", "si", "da", "sono", "al", "alla", "gli", "lo", "nel", "nella", "suo", "sua", "essere", "hanno", "questa", "questo"}


class Ingestor:
    """Load and normalize corpus data into SourceChunks."""
    
    def __init__(self, data_dir: Path | str, show_progress: bool = True):
        """Initialize ingestor.
        
        Args:
            data_dir: Path to data directory containing CSV files
            show_progress: Whether to show tqdm progress bars
        """
        self.data_dir = Path(data_dir)
        self.laws_csv = self.data_dir / "laws_de.csv"
        self.courts_csv = self.data_dir / "court_considerations.csv"
        self.show_progress = show_progress
    
    def _iter_with_progress(
        self,
        iterable,
        total: int,
        desc: str,
    ):
        """Wrap iterable with tqdm if progress is enabled."""
        if self.show_progress:
            return tqdm(iterable, total=total, desc=desc)
        return iterable
    
    def load_all(self) -> pd.DataFrame:
        """Load and merge all corpus data.
        
        Returns:
            DataFrame with columns: chunk_id, chunk_type, group_id, lang, text_raw
        """
        chunks = []
        
        # Load laws
        if self.laws_csv.exists():
            if self.show_progress:
                print(f"Loading laws from {self.laws_csv.name}...")
            laws_df = self._load_laws()
            chunks.append(laws_df)
        
        # Load court considerations
        if self.courts_csv.exists():
            if self.show_progress:
                print(f"Loading court considerations from {self.courts_csv.name}...")
            courts_df = self._load_courts()
            chunks.append(courts_df)
        
        if not chunks:
            raise FileNotFoundError(f"No corpus files found in {self.data_dir}")
        
        # Merge and validate
        df = pd.concat(chunks, ignore_index=True)
        df = self._validate_unique_ids(df)
        
        if self.show_progress:
            print(f"Total chunks loaded: {len(df)}")
        
        return df
    
    def _load_laws(self) -> pd.DataFrame:
        """Load federal laws corpus."""
        df = pd.read_csv(self.laws_csv)
        
        records = []
        for _, row in self._iter_with_progress(df.iterrows(), total=len(df), desc="Processing laws"):
            citation = self._normalize_text(str(row["citation"]))
            text = self._normalize_text(str(row.get("text", "")))
            
            records.append({
                "chunk_id": citation,
                "chunk_type": ChunkType.LAW.value,
                "group_id": self._extract_group_id(citation, ChunkType.LAW),
                "lang": self._detect_language(text).value,
                "text_raw": text,
            })
        
        return pd.DataFrame(records)
    
    def _load_courts(self) -> pd.DataFrame:
        """Load court considerations corpus."""
        df = pd.read_csv(self.courts_csv)
        
        records = []
        for _, row in self._iter_with_progress(df.iterrows(), total=len(df), desc="Processing courts"):
            citation = self._normalize_text(str(row["citation"]))
            text = self._normalize_text(str(row.get("text", "")))
            
            records.append({
                "chunk_id": citation,
                "chunk_type": ChunkType.CASE.value,
                "group_id": self._extract_group_id(citation, ChunkType.CASE),
                "lang": self._detect_language(text).value,
                "text_raw": text,
            })
        
        return pd.DataFrame(records)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text: trim and collapse whitespace."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text
    
    def _detect_language(self, text: str) -> Language:
        """Detect language using stopword heuristic."""
        if not text:
            return Language.UNKNOWN
        
        # Tokenize (simple)
        words = set(re.findall(r"\b[a-zäöüàâçéèêëïîôùûœæ]+\b", text.lower()))
        
        # Count stopword matches
        de_count = len(words & _DE_STOPWORDS)
        fr_count = len(words & _FR_STOPWORDS)
        it_count = len(words & _IT_STOPWORDS)
        
        max_count = max(de_count, fr_count, it_count)
        if max_count < 3:
            return Language.UNKNOWN
        
        if de_count == max_count:
            return Language.DE
        elif fr_count == max_count:
            return Language.FR
        else:
            return Language.IT
    
    def _extract_group_id(self, citation: str, chunk_type: ChunkType) -> str | None:
        """Extract DocGroup ID from citation.
        
        Law: "Art. 11 Abs. 2 OR" → "code:OR"
        BGE: "BGE 145 II 32 E. 3.1" → "bge:145_II_32"
        Docket: "5A_800/2019 E. 2" → "docket:5A_800_2019"
        """
        if chunk_type == ChunkType.LAW:
            # Extract law code (last word typically)
            # Pattern: Art. X [Abs. Y] CODE
            match = re.search(r"(?:Art\.|Artikel)\s+\d+[a-z]?(?:\s+Abs\.\s+\d+[a-z]?)?\s+(\w+)", citation)
            if match:
                return f"code:{match.group(1)}"
            # Fallback: last word
            words = citation.split()
            if words:
                return f"code:{words[-1]}"
        
        elif chunk_type == ChunkType.CASE:
            # BGE pattern: BGE 145 II 32 [E. ...]
            bge_match = re.match(r"BGE\s+(\d+)\s+([IVX]+[a-z]?)\s+(\d+)", citation)
            if bge_match:
                vol, sec, page = bge_match.groups()
                return f"bge:{vol}_{sec}_{page}"
            
            # Docket pattern: 5A_800/2019 [E. ...]
            docket_match = re.match(r"(\d+[A-Z]_\d+/\d+)", citation)
            if docket_match:
                docket = docket_match.group(1).replace("/", "_")
                return f"docket:{docket}"
        
        return None
    
    def _validate_unique_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all chunk_ids are unique."""
        duplicates = df[df.duplicated(subset=["chunk_id"], keep=False)]
        if len(duplicates) > 0:
            # Keep first occurrence
            df = df.drop_duplicates(subset=["chunk_id"], keep="first")
            print(f"Warning: Removed {len(duplicates) - len(df)} duplicate chunk_ids")
        return df
    
    def save(self, df: pd.DataFrame, output_path: Path | str) -> None:
        """Save chunks to parquet file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
    
    @staticmethod
    def load_chunks(path: Path | str) -> pd.DataFrame:
        """Load chunks from parquet file."""
        return pd.read_parquet(path)
