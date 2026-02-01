"""Summarizer: Generate summaries for chunks.

Input: chunks.parquet
Output: summaries.parquet

Supports two modes:
1. LLM mode (API) - for testing/development
2. Heuristic mode - for Kaggle offline execution

Environment variables for LLM mode:
- OPENAI_API_KEY: API key for OpenAI or compatible provider
- PROXYAPI_BASE_URL: Base URL for API (default: https://api.proxyapi.ru/openai/v1)
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Literal

import pandas as pd


# =============================================================================
# API Configuration
# =============================================================================
DEFAULT_OPENAI_BASE_URL = "https://api.proxyapi.ru/openai/v1"
DEFAULT_GOOGLE_BASE_URL = "https://api.proxyapi.ru/google"
DEFAULT_LLM_MODEL = "gpt-4o"


def create_llm_client(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = DEFAULT_LLM_MODEL,
    provider: Literal["openai", "google"] = "openai",
) -> Callable[[str], str]:
    """Create an LLM client for Summarizer.
    
    Args:
        api_key: API key. If None, reads from OPENAI_API_KEY env var.
        base_url: Base URL for API. If None, uses default ProxyAPI URL for provider.
        model: Model name to use (default: gpt-4o for openai, gemini-2.0-flash for google).
        provider: "openai" or "google" - which API to use.
    
    Returns:
        Callable that takes prompt string and returns response string.
        
    Example:
        >>> # OpenAI/ProxyAPI
        >>> client = create_llm_client(provider="openai", model="gpt-4o")
        >>> 
        >>> # Google Gemini via ProxyAPI
        >>> client = create_llm_client(provider="google", model="gemini-2.5-flash-lite")
    """
    # Resolve configuration
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    
    if not resolved_api_key:
        raise ValueError(
            "API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter."
        )
    
    if provider == "google":
        return _create_google_client(resolved_api_key, base_url, model)
    else:
        return _create_openai_client(resolved_api_key, base_url, model)


def _create_openai_client(api_key: str, base_url: str | None, model: str) -> Callable[[str], str]:
    """Create OpenAI-compatible client."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    
    resolved_base_url = base_url or DEFAULT_OPENAI_BASE_URL
    client = OpenAI(api_key=api_key, base_url=resolved_base_url)
    
    def call_llm(prompt: str) -> str:
        """Call the LLM with a prompt."""
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    
    # Store info for debugging
    call_llm.model = model  # type: ignore
    call_llm.base_url = resolved_base_url  # type: ignore
    call_llm.provider = "openai"  # type: ignore
    
    return call_llm


def _create_google_client(api_key: str, base_url: str | None, model: str) -> Callable[[str], str]:
    """Create Google Gemini client."""
    try:
        from google import genai
    except ImportError:
        raise ImportError("google-genai package required. Install with: pip install google-genai")
    
    resolved_base_url = base_url or DEFAULT_GOOGLE_BASE_URL
    client = genai.Client(api_key=api_key, http_options={"base_url": resolved_base_url})
    
    def call_llm(prompt: str) -> str:
        """Call the LLM with a prompt."""
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
        )
        return response.text or ""
    
    # Store info for debugging
    call_llm.model = model  # type: ignore
    call_llm.base_url = resolved_base_url  # type: ignore
    call_llm.provider = "google"  # type: ignore
    
    return call_llm


# Backward compatibility alias
def create_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = DEFAULT_LLM_MODEL,
    provider: Literal["openai", "google"] = "openai",
) -> Callable[[str], str]:
    """Alias for create_llm_client (backward compatibility)."""
    return create_llm_client(api_key=api_key, base_url=base_url, model=model, provider=provider)


class SummaryType(Enum):
    """Type of summary."""
    SHORT = "short"           # 1-2 sentences, 200-400 chars
    RETRIEVAL = "retrieval"   # Structured, 400-1200 chars
    ENTITIES = "entities"     # 5-30 keyphrases


@dataclass
class Summary:
    """A summary for a source chunk."""
    chunk_id: str
    summary_type: SummaryType
    summary_text: str
    entities: list[str]  # Only populated for type=entities


# Common legal terms for entity extraction
_LEGAL_TERM_PATTERNS = [
    r"Art\.\s*\d+[a-z]?",           # Article references
    r"Abs\.\s*\d+[a-z]?",           # Paragraph references
    r"BGE\s+\d+\s+[IVX]+[a-z]?\s+\d+",  # BGE references
    r"E\.\s*\d+(?:\.\d+)*[a-z]?",   # Consideration references
    r"\d+[A-Z]_\d+/\d+",            # Docket numbers
]


class Summarizer:
    """Generate summaries for source chunks."""
    
    def __init__(
        self,
        mode: Literal["heuristic", "llm"] = "heuristic",
        llm_client: Callable[[str], str] | None = None,
    ):
        """Initialize summarizer.
        
        Args:
            mode: "heuristic" for offline/Kaggle, "llm" for API-based
            llm_client: Callable that takes prompt and returns response (for LLM mode)
        """
        self.mode = mode
        self.llm_client = llm_client
        
        if mode == "llm" and llm_client is None:
            raise ValueError("llm_client required for LLM mode")
    
    def summarize_all(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """Generate all summary types for all chunks.
        
        Args:
            chunks_df: DataFrame with chunk_id, text_raw columns
            
        Returns:
            DataFrame with chunk_id, summary_type, summary_text, entities
        """
        records = []
        
        for _, row in chunks_df.iterrows():
            chunk_id = row["chunk_id"]
            text = row["text_raw"]
            
            # Generate each summary type
            for summary_type in SummaryType:
                summary = self._generate_summary(text, summary_type)
                records.append({
                    "chunk_id": chunk_id,
                    "summary_type": summary_type.value,
                    "summary_text": summary.summary_text,
                    "entities": summary.entities,
                })
        
        return pd.DataFrame(records)
    
    def _generate_summary(self, text: str, summary_type: SummaryType) -> Summary:
        """Generate a single summary."""
        if self.mode == "llm":
            return self._generate_llm(text, summary_type)
        else:
            return self._generate_heuristic(text, summary_type)
    
    def _generate_heuristic(self, text: str, summary_type: SummaryType) -> Summary:
        """Generate summary using heuristics (for Kaggle offline)."""
        if summary_type == SummaryType.SHORT:
            return self._heuristic_short(text)
        elif summary_type == SummaryType.RETRIEVAL:
            return self._heuristic_retrieval(text)
        else:  # ENTITIES
            return self._heuristic_entities(text)
    
    def _heuristic_short(self, text: str) -> Summary:
        """Generate short summary: first 1-2 sentences, 200-400 chars."""
        # Split into sentences (handling abbreviations)
        # Look for [.!?] followed by space, BUT NOT preceded by Nr/Art/Abs/BGE/ca
        sentences = re.split(r"(?<!\bNr)(?<!\bArt)(?<!\bAbs)(?<!\bBGE)(?<!\bca)(?<!\bz\.\s?B)(?<=[.!?])\s+", text)
        
        # Take first sentences up to ~300 chars
        summary_parts = []
        total_len = 0
        for sent in sentences[:3]:
            if total_len + len(sent) > 400:
                break
            summary_parts.append(sent)
            total_len += len(sent)
        
        summary_text = " ".join(summary_parts)
        
        # Ensure minimum length
        if len(summary_text) < 50 and text:
            summary_text = text[:400]
        
        return Summary(
            chunk_id="",  # Will be set by caller
            summary_type=SummaryType.SHORT,
            summary_text=summary_text.strip(),
            entities=[],
        )
    
    def _heuristic_retrieval(self, text: str) -> Summary:
        """Generate retrieval summary: structured with keyphrases."""
        # Extract keyphrases using simple TF heuristic
        keyphrases = self._extract_keyphrases(text, top_k=10)
        
        # Extract legal references
        legal_refs = self._extract_legal_references(text)
        
        # Build structured summary
        parts = []
        
        # First sentence as topic (using better splitter)
        sentences = re.split(r"(?<!\bNr)(?<!\bArt)(?<!\bAbs)(?<!\bBGE)(?<!\bca)(?<!\bz\.\s?B)(?<=[.!?])\s+", text)
        if sentences:
            parts.append(f"Topic: {sentences[0][:200]}")
        
        # Key terms
        if keyphrases:
            parts.append(f"Terms: {', '.join(keyphrases[:7])}")
        
        # Legal references
        if legal_refs:
            parts.append(f"References: {', '.join(legal_refs[:5])}")
        
        # Additional context (middle sentences)
        if len(sentences) > 2:
            context = " ".join(sentences[1:3])[:300]
            parts.append(f"Context: {context}")
        
        summary_text = " | ".join(parts)
        
        # Ensure within bounds (400-1200 chars)
        if len(summary_text) > 1200:
            summary_text = summary_text[:1200]
        elif len(summary_text) < 100:
            summary_text = text[:600]
        
        return Summary(
            chunk_id="",
            summary_type=SummaryType.RETRIEVAL,
            summary_text=summary_text.strip(),
            entities=[],
        )
    
    def _heuristic_entities(self, text: str) -> Summary:
        """Generate entity list: 5-30 keyphrases/terms."""
        # Combine keyphrases and legal references
        keyphrases = self._extract_keyphrases(text, top_k=20)
        legal_refs = self._extract_legal_references(text)
        
        # Deduplicate and limit
        entities = list(dict.fromkeys(legal_refs + keyphrases))[:30]
        
        # Ensure minimum
        if len(entities) < 5 and text:
            # Add first few significant words
            words = re.findall(r"\b[A-ZÄÖÜ][a-zäöüß]{3,}\b", text)
            entities.extend(words[:10])
            entities = list(dict.fromkeys(entities))[:30]
        
        return Summary(
            chunk_id="",
            summary_type=SummaryType.ENTITIES,
            summary_text="; ".join(entities),
            entities=entities,
        )
    
    def _extract_keyphrases(self, text: str, top_k: int = 10) -> list[str]:
        """Extract keyphrases using simple word frequency."""
        # Tokenize (at least 4 chars)
        words = re.findall(r"\b[A-Za-zÄÖÜäöüß]{4,}\b", text)
        
        # Count frequencies
        freq: dict[str, int] = {}
        for word in words:
            word_lower = word.lower()
            freq[word_lower] = freq.get(word_lower, 0) + 1
        
        # Filter stopwords (extended German list)
        stopwords = {
            # Articles & Pronouns
            "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", "einem", "einen",
            "ich", "du", "er", "sie", "es", "wir", "ihr", "sie", "mich", "dich", "uns", "euch",
            "mein", "dein", "sein", "ihr", "unser", "euer", 
            "diese", "dieser", "dieses", "diesem", "diesen", "jenes", "jener", "jenem",
            "welcher", "welche", "welches", "welchen", "welchem", "demselben", "derselben",
            # Prepositions & Conjunctions
            "und", "oder", "aber", "doch", "sondern", "denn", "weil", "wenn", "als", "wie", "dass",
            "von", "bism", "zu", "nach", "bei", "mit", "für", "um", "durch", "ohne", "gegen",
            "über", "unter", "zwischen", "vor", "hinter", "auf", "aus", "seit", "während", "wegen",
            # Verbs (forms of sein, haben, werden, modal)
            "ist", "sind", "war", "waren", "gewesen", "bin", "bist", "seid",
            "haben", "hat", "hatte", "hatten", "gehabt",
            "werden", "wird", "wurde", "wurden", "geworden",
            "kann", "können", "muss", "müssen", "soll", "sollen", "will", "wollen",
            "darf", "dürfen", "lassen", "lässt", "lies", "tritt", "treten",
            # Common vague words
            "falle", "fall", "davon", "dazu", "dabei", "dafür", "damit", "darauf", "daran",
            "hier", "dort", "jetzt", "dann", "schon", "noch", "immer", "nie",
            "etwas", "nichts", "alles", "viel", "wenig", "alle", "keine", "kein",
            "gemäss", "laut", "zufolge", "sowie", "auch", "nicht", "nein", "ja",
            "ungefähr", "etwa", "fast", "kaum", "sehr", "ganz", "besonders",
            "fragen", "frage", "sagen", "gesagt", "machen", "gemacht", "tun",
            "sehen", "gesehen", "gehen", "gegangen", "kommen", "gekommen",
        }
        
        freq = {k: v for k, v in freq.items() if k not in stopwords}
        
        # Sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words[:top_k]]
    
    def _extract_legal_references(self, text: str) -> list[str]:
        """Extract legal references (Art., BGE, etc.) from text."""
        refs = []
        for pattern in _LEGAL_TERM_PATTERNS:
            matches = re.findall(pattern, text)
            refs.extend(matches)
        return list(dict.fromkeys(refs))  # Dedupe while preserving order
    
    def _generate_llm(self, text: str, summary_type: SummaryType) -> Summary:
        """Generate summary using LLM API."""
        if self.llm_client is None:
            raise ValueError("LLM client not configured")
        
        prompts = {
            SummaryType.SHORT: f"""Summarize this Swiss legal text in 1-2 sentences (200-400 characters):

{text[:2000]}

Summary:""",
            
            SummaryType.RETRIEVAL: f"""Create a structured retrieval summary for this Swiss legal text.
Include: topic, key conditions, exceptions, and important terms.
Length: 400-1200 characters.

{text[:2000]}

Structured summary:""",
            
            SummaryType.ENTITIES: f"""Extract 10-30 key legal terms and references from this Swiss legal text.
Include: law codes, article numbers, legal concepts, party types.
Format: semicolon-separated list.

{text[:2000]}

Entities:""",
        }
        
        response = self.llm_client(prompts[summary_type])
        
        entities = []
        if summary_type == SummaryType.ENTITIES:
            entities = [e.strip() for e in response.split(";") if e.strip()]
        
        return Summary(
            chunk_id="",
            summary_type=summary_type,
            summary_text=response.strip(),
            entities=entities,
        )
    
    def save(self, df: pd.DataFrame, output_path: Path | str) -> None:
        """Save summaries to parquet file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
    
    @staticmethod
    def load_summaries(path: Path | str) -> pd.DataFrame:
        """Load summaries from parquet file."""
        return pd.read_parquet(path)
