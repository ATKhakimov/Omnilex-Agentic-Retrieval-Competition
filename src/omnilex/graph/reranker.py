"""LLM Reranker: Point-wise reranking of candidates using local LLM.

This module provides LLM-based reranking for the final stage of retrieval.
Designed for Kaggle offline execution with local LLM (e.g., llama.cpp).

Usage in pipeline:
1. Fast scoring produces C2 (top 80-150 candidates)
2. LLM Reranker filters/reorders C2 to final predictions
3. Budget: 1-3 LLM calls per query (batch candidates)
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

import pandas as pd


@dataclass
class RerankerConfig:
    """Configuration for LLM Reranker."""
    
    # Number of candidates to rerank (from fast scoring)
    top_k_to_rerank: int = 100
    
    # Final number of citations to return
    top_k_final: int = 20
    
    # Batch size for LLM calls (candidates per prompt)
    batch_size: int = 20
    
    # Whether to use LLM for reranking (fallback to scores if False)
    use_llm: bool = True
    
    # Relevance threshold (0-1) for filtering
    relevance_threshold: float = 0.5
    
    # Max text length per candidate in prompt
    max_text_length: int = 300


# =============================================================================
# Prompt Templates
# =============================================================================

RERANK_PROMPT_TEMPLATE = """You are a legal expert assistant. Given a legal query and a list of candidate Swiss legal sources, rate each candidate's relevance to the query.

## Query
{query}

## Candidates
{candidates_text}

## Instructions
For each candidate, output a relevance score from 0 to 10:
- 0-2: Not relevant
- 3-5: Partially relevant  
- 6-8: Relevant
- 9-10: Highly relevant

Output format (one line per candidate):
ID: <id> | Score: <score>

Only output the scores, no explanations."""


QUERY_SUMMARY_PROMPT = """Summarize this legal query for retrieval purposes. Extract:
1. Main legal topic/area
2. Key legal concepts
3. Relevant Swiss law areas (e.g., OR, ZGB, StGB)

Query: {query}

Output a concise retrieval-focused summary (2-3 sentences max)."""


ADAPTIVE_K_PROMPT = """Given this legal query and {n_candidates} candidate citations with their relevance scores, how many citations should be returned?

Query: {query}

Score distribution:
- High (7-10): {n_high}
- Medium (4-6): {n_medium}  
- Low (0-3): {n_low}

Consider:
- Complex queries may need more citations
- Simple queries may need fewer
- Only include clearly relevant citations

Output a single number between 1 and {max_k}."""


# =============================================================================
# Reranker Class
# =============================================================================

class LLMReranker:
    """LLM-based reranker for candidate citations."""
    
    def __init__(
        self,
        llm_client: Callable[[str], str] | None = None,
        config: RerankerConfig | None = None,
    ):
        """Initialize reranker.
        
        Args:
            llm_client: Callable that takes prompt and returns response.
                       If None, falls back to score-based ranking.
            config: Reranker configuration.
        """
        self.llm_client = llm_client
        self.config = config or RerankerConfig()
        
    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float, dict]],
        chunks_df: pd.DataFrame | None = None,
        summaries_df: pd.DataFrame | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Rerank candidates using LLM.
        
        Args:
            query: The search query.
            candidates: List of (chunk_id, score, features) from fast scoring.
            chunks_df: Optional chunks dataframe for text lookup.
            summaries_df: Optional summaries dataframe for text lookup.
            
        Returns:
            Reranked list of (chunk_id, new_score, features).
        """
        if not candidates:
            return []
            
        # Take top K for reranking
        to_rerank = candidates[:self.config.top_k_to_rerank]
        
        # If no LLM client, just return top K by score
        if not self.config.use_llm or self.llm_client is None:
            return to_rerank[:self.config.top_k_final]
        
        # Get text for each candidate
        candidate_texts = self._get_candidate_texts(
            [c[0] for c in to_rerank],
            chunks_df,
            summaries_df,
        )
        
        # Batch rerank
        all_scores: dict[str, float] = {}
        
        for i in range(0, len(to_rerank), self.config.batch_size):
            batch = to_rerank[i:i + self.config.batch_size]
            batch_ids = [c[0] for c in batch]
            batch_texts = {cid: candidate_texts.get(cid, "") for cid in batch_ids}
            
            batch_scores = self._rerank_batch(query, batch_ids, batch_texts)
            all_scores.update(batch_scores)
        
        # Combine LLM scores with original scores
        reranked = []
        for chunk_id, orig_score, features in to_rerank:
            llm_score = all_scores.get(chunk_id, 5.0) / 10.0  # Normalize to 0-1
            
            # Weighted combination: 60% LLM, 40% original
            combined_score = 0.6 * llm_score + 0.4 * self._normalize_score(orig_score, to_rerank)
            
            new_features = features.copy()
            new_features["llm_score"] = llm_score
            new_features["orig_score"] = orig_score
            
            reranked.append((chunk_id, combined_score, new_features))
        
        # Sort by new score and apply threshold
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by relevance threshold
        if self.config.relevance_threshold > 0:
            reranked = [
                r for r in reranked 
                if r[2].get("llm_score", 1.0) >= self.config.relevance_threshold
            ]
        
        return reranked[:self.config.top_k_final]
    
    def _get_candidate_texts(
        self,
        chunk_ids: list[str],
        chunks_df: pd.DataFrame | None,
        summaries_df: pd.DataFrame | None,
    ) -> dict[str, str]:
        """Get text representation for each candidate."""
        texts = {}
        
        for chunk_id in chunk_ids:
            text = ""
            
            # Try to get summary first (preferred - more concise)
            if summaries_df is not None:
                summary_row = summaries_df[
                    (summaries_df["chunk_id"] == chunk_id) & 
                    (summaries_df["summary_type"] == "short")
                ]
                if not summary_row.empty:
                    text = summary_row.iloc[0]["summary_text"]
            
            # Fallback to raw text
            if not text and chunks_df is not None:
                chunk_row = chunks_df[chunks_df["chunk_id"] == chunk_id]
                if not chunk_row.empty:
                    text = chunk_row.iloc[0]["text_raw"]
            
            # Truncate if needed
            if len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length] + "..."
            
            texts[chunk_id] = text or f"[Citation: {chunk_id}]"
        
        return texts
    
    def _rerank_batch(
        self,
        query: str,
        chunk_ids: list[str],
        texts: dict[str, str],
    ) -> dict[str, float]:
        """Rerank a batch of candidates using LLM."""
        # Format candidates for prompt
        candidates_text = "\n".join([
            f"[{i+1}] {cid}\n{texts.get(cid, '')}\n"
            for i, cid in enumerate(chunk_ids)
        ])
        
        prompt = RERANK_PROMPT_TEMPLATE.format(
            query=query,
            candidates_text=candidates_text,
        )
        
        try:
            response = self.llm_client(prompt)
            return self._parse_rerank_response(response, chunk_ids)
        except Exception as e:
            print(f"LLM rerank error: {e}")
            # Fallback: return neutral scores
            return {cid: 5.0 for cid in chunk_ids}
    
    def _parse_rerank_response(
        self,
        response: str,
        chunk_ids: list[str],
    ) -> dict[str, float]:
        """Parse LLM reranking response."""
        scores = {}
        
        # Try to parse "ID: ... | Score: X" format
        pattern = r"(?:ID:\s*)?(.+?)\s*\|\s*Score:\s*(\d+(?:\.\d+)?)"
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        if matches:
            for id_text, score_text in matches:
                id_text = id_text.strip()
                try:
                    score = float(score_text)
                    # Match to chunk_id
                    for cid in chunk_ids:
                        if id_text in cid or cid in id_text:
                            scores[cid] = min(10.0, max(0.0, score))
                            break
                except ValueError:
                    continue
        
        # Alternative: try numbered format "[1] Score: X"
        if not scores:
            pattern = r"\[(\d+)\].*?(?:Score|:)\s*(\d+(?:\.\d+)?)"
            matches = re.findall(pattern, response, re.IGNORECASE)
            
            for idx_text, score_text in matches:
                try:
                    idx = int(idx_text) - 1
                    score = float(score_text)
                    if 0 <= idx < len(chunk_ids):
                        scores[chunk_ids[idx]] = min(10.0, max(0.0, score))
                except (ValueError, IndexError):
                    continue
        
        # Fill in missing with neutral score
        for cid in chunk_ids:
            if cid not in scores:
                scores[cid] = 5.0
        
        return scores
    
    def _normalize_score(
        self,
        score: float,
        candidates: list[tuple[str, float, dict]],
    ) -> float:
        """Normalize score to 0-1 based on min/max in candidates."""
        if not candidates:
            return 0.5
        
        scores = [c[1] for c in candidates]
        min_s, max_s = min(scores), max(scores)
        
        if max_s == min_s:
            return 0.5
        
        return (score - min_s) / (max_s - min_s)
    
    def summarize_query(self, query: str) -> str:
        """Generate a retrieval-focused summary of the query.
        
        Args:
            query: The original query text.
            
        Returns:
            Summarized query for better retrieval.
        """
        if self.llm_client is None:
            return query
        
        prompt = QUERY_SUMMARY_PROMPT.format(query=query)
        
        try:
            return self.llm_client(prompt)
        except Exception as e:
            print(f"Query summary error: {e}")
            return query
    
    def predict_k(
        self,
        query: str,
        scores: list[float],
        max_k: int = 50,
    ) -> int:
        """Predict optimal number of citations to return.
        
        Args:
            query: The query.
            scores: LLM relevance scores (0-10) for candidates.
            max_k: Maximum allowed citations.
            
        Returns:
            Predicted number of citations.
        """
        if self.llm_client is None or not scores:
            return min(20, max_k)
        
        # Count by score range
        n_high = sum(1 for s in scores if s >= 7)
        n_medium = sum(1 for s in scores if 4 <= s < 7)
        n_low = sum(1 for s in scores if s < 4)
        
        prompt = ADAPTIVE_K_PROMPT.format(
            query=query,
            n_candidates=len(scores),
            n_high=n_high,
            n_medium=n_medium,
            n_low=n_low,
            max_k=max_k,
        )
        
        try:
            response = self.llm_client(prompt)
            # Extract number from response
            numbers = re.findall(r"\d+", response)
            if numbers:
                k = int(numbers[0])
                return min(max(1, k), max_k)
        except Exception as e:
            print(f"Predict K error: {e}")
        
        # Fallback: return high + half of medium
        return min(n_high + n_medium // 2 + 1, max_k)


# =============================================================================
# Query Preprocessor
# =============================================================================

class QueryPreprocessor:
    """Preprocess queries for better retrieval."""
    
    def __init__(self, llm_client: Callable[[str], str] | None = None):
        self.llm_client = llm_client
    
    def preprocess(self, query: str, use_llm: bool = False) -> str:
        """Preprocess query.
        
        Args:
            query: Original query text.
            use_llm: Whether to use LLM for summarization.
            
        Returns:
            Preprocessed query.
        """
        # Basic normalization
        query = self._normalize_text(query)
        
        # LLM summarization (optional)
        if use_llm and self.llm_client is not None:
            query = self._summarize_query(query)
        
        return query
    
    def _normalize_text(self, text: str) -> str:
        """Basic text normalization."""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Normalize quotes
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")
        
        return text.strip()
    
    def _summarize_query(self, query: str) -> str:
        """Use LLM to create retrieval-focused summary."""
        if self.llm_client is None:
            return query
            
        prompt = QUERY_SUMMARY_PROMPT.format(query=query)
        
        try:
            summary = self.llm_client(prompt)
            # Return both original and summary for hybrid retrieval
            return f"{query}\n\n{summary}"
        except Exception:
            return query
