"""Graph Builder: Build edges (SIMILAR_TO, CO_CITED_WITH, PART_OF).

Input: embeddings, train.csv, chunks.parquet
Output: edges_similar.parquet, edges_cocite.parquet, groups.parquet, chunk_to_group.parquet
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# =============================================================================
# Default Parameters (with tuning ranges in comments)
# =============================================================================

# Expansion parameters
K_EXPAND_SIM = 20           # Range: 10-50 | Similar neighbors per candidate
K_EXPAND_COCITE = 30        # Range: 10-100 | Co-cited neighbors per candidate
K_EXPAND_SIBLINGS = 10      # Range: 0-30 | DocGroup siblings per candidate
MIN_SIM_COS = 0.25          # Range: 0.15-0.35 | Cosine threshold for SIMILAR_TO
MAX_CANDIDATES_AFTER_EXPANSION = 800  # Range: 300-2000 | Cap to prevent timeout

# Scoring parameters (linear combination, features normalized to [0,1])
ALPHA = 1.0   # Range: 0.5-2.0 | Retrieval score weight
BETA = 0.6    # Range: 0.0-1.5 | SIMILAR_TO edge weight
GAMMA = 0.8   # Range: 0.0-2.0 | CO_CITED_WITH edge weight
DELTA = 0.2   # Range: 0.0-0.8 | DocGroup coherence weight


@dataclass
class ScoringParams:
    """Parameters for candidate scoring."""
    alpha: float = ALPHA
    beta: float = BETA
    gamma: float = GAMMA
    delta: float = DELTA


@dataclass
class ExpansionParams:
    """Parameters for graph expansion."""
    k_expand_sim: int = K_EXPAND_SIM
    k_expand_cocite: int = K_EXPAND_COCITE
    k_expand_siblings: int = K_EXPAND_SIBLINGS
    min_sim_cos: float = MIN_SIM_COS
    max_candidates: int = MAX_CANDIDATES_AFTER_EXPANSION


class GraphBuilder:
    """Build graph edges for retrieval augmentation."""
    
    def __init__(self):
        """Initialize graph builder."""
        # SIMILAR_TO edges: chunk_id -> [(neighbor_id, score), ...]
        self.similar_edges: dict[str, list[tuple[str, float]]] = {}
        
        # CO_CITED_WITH edges: chunk_id -> [(neighbor_id, weight), ...]
        self.cocite_edges: dict[str, list[tuple[str, float]]] = {}
        
        # DocGroup membership: chunk_id -> group_id
        self.chunk_to_group: dict[str, str] = {}
        
        # Group members: group_id -> [chunk_id, ...]
        self.group_to_chunks: dict[str, list[str]] = {}
    
    # =========================================================================
    # Build SIMILAR_TO edges
    # =========================================================================
    
    def build_similar_edges(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        k: int = 50,
        min_cos: float = MIN_SIM_COS,
    ) -> pd.DataFrame:
        """Build SIMILAR_TO edges using embedding similarity.
        
        Args:
            embeddings: Normalized embedding vectors (N, dim)
            chunk_ids: List of chunk IDs corresponding to embeddings
            k: Number of neighbors per node
            min_cos: Minimum cosine similarity threshold
            
        Returns:
            DataFrame with src_chunk_id, dst_chunk_id, score
        """
        if not HAS_FAISS:
            raise ImportError("faiss required. Install with: pip install faiss-cpu")
        
        # Normalize embeddings
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Build index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        
        # Search for k+1 neighbors (includes self)
        scores, indices = index.search(embeddings, k + 1)
        
        # Build edges
        records = []
        for i, chunk_id in enumerate(chunk_ids):
            neighbors = []
            for j, (idx, score) in enumerate(zip(indices[i], scores[i])):
                if idx == i:  # Skip self
                    continue
                if score < min_cos:
                    continue
                if idx >= 0 and idx < len(chunk_ids):
                    neighbor_id = chunk_ids[idx]
                    neighbors.append((neighbor_id, float(score)))
                    records.append({
                        "src_chunk_id": chunk_id,
                        "dst_chunk_id": neighbor_id,
                        "score": float(score),
                    })
            self.similar_edges[chunk_id] = neighbors
        
        return pd.DataFrame(records)
    
    # =========================================================================
    # Build CO_CITED_WITH edges
    # =========================================================================
    
    def build_cocite_edges(
        self,
        train_df: pd.DataFrame,
        gold_col: str = "gold_citations",
        top_m: int = 50,
    ) -> pd.DataFrame:
        """Build CO_CITED_WITH edges from training data.
        
        Args:
            train_df: Training DataFrame with gold_citations column
            gold_col: Column name for gold citations
            top_m: Max neighbors to keep per node
            
        Returns:
            DataFrame with a_chunk_id, b_chunk_id, weight
        """
        # Count co-occurrences
        cooccur: dict[tuple[str, str], int] = defaultdict(int)
        freq: dict[str, int] = defaultdict(int)
        
        for _, row in train_df.iterrows():
            citations_str = row.get(gold_col, "")
            if pd.isna(citations_str) or not citations_str:
                continue
            
            citations = [c.strip() for c in str(citations_str).split(";") if c.strip()]
            
            # Update frequencies
            for c in citations:
                freq[c] += 1
            
            # Update co-occurrences (pairs)
            for i, c1 in enumerate(citations):
                for c2 in citations[i + 1:]:
                    key = tuple(sorted([c1, c2]))
                    cooccur[key] += 1
        
        # Compute weights: count / sqrt(freq_i * freq_j)
        records = []
        edges: dict[str, list[tuple[str, float]]] = defaultdict(list)
        
        for (c1, c2), count in cooccur.items():
            if freq[c1] > 0 and freq[c2] > 0:
                weight = count / math.sqrt(freq[c1] * freq[c2])
                edges[c1].append((c2, weight))
                edges[c2].append((c1, weight))
                records.append({
                    "a_chunk_id": c1,
                    "b_chunk_id": c2,
                    "weight": weight,
                })
        
        # Keep top_m neighbors per node
        for chunk_id, neighbors in edges.items():
            neighbors.sort(key=lambda x: x[1], reverse=True)
            self.cocite_edges[chunk_id] = neighbors[:top_m]
        
        return pd.DataFrame(records)
    
    # =========================================================================
    # Build DocGroup mapping
    # =========================================================================
    
    def build_groups(self, chunks_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build DocGroup mapping from chunks.
        
        Args:
            chunks_df: DataFrame with chunk_id, group_id columns
            
        Returns:
            Tuple of (groups_df, chunk_to_group_df)
        """
        # Build mappings
        for _, row in chunks_df.iterrows():
            chunk_id = row["chunk_id"]
            group_id = row.get("group_id")
            
            if pd.notna(group_id) and group_id:
                self.chunk_to_group[chunk_id] = group_id
                if group_id not in self.group_to_chunks:
                    self.group_to_chunks[group_id] = []
                self.group_to_chunks[group_id].append(chunk_id)
        
        # Create groups DataFrame
        groups_records = []
        for group_id in self.group_to_chunks.keys():
            group_type = "law_code" if group_id.startswith("code:") else "decision"
            groups_records.append({
                "group_id": group_id,
                "group_type": group_type,
                "num_chunks": len(self.group_to_chunks[group_id]),
            })
        groups_df = pd.DataFrame(groups_records)
        
        # Create chunk_to_group DataFrame
        chunk_to_group_records = [
            {"chunk_id": cid, "group_id": gid}
            for cid, gid in self.chunk_to_group.items()
        ]
        chunk_to_group_df = pd.DataFrame(chunk_to_group_records)
        
        return groups_df, chunk_to_group_df
    
    # =========================================================================
    # Graph Expansion (for inference)
    # =========================================================================
    
    def expand_candidates(
        self,
        initial_candidates: list[tuple[str, float]],
        params: ExpansionParams | None = None,
    ) -> list[tuple[str, float, str]]:
        """Expand candidate set using graph edges.
        
        Args:
            initial_candidates: List of (chunk_id, retrieval_score) tuples
            params: Expansion parameters (uses defaults if None)
            
        Returns:
            List of (chunk_id, score, expansion_reason) tuples
        """
        if params is None:
            params = ExpansionParams()
        
        expanded: dict[str, tuple[float, str]] = {}
        
        # Add initial candidates
        for chunk_id, score in initial_candidates:
            expanded[chunk_id] = (score, "initial")
        
        # Expand via SIMILAR_TO
        for chunk_id, _ in initial_candidates:
            neighbors = self.similar_edges.get(chunk_id, [])
            for neighbor_id, sim_score in neighbors[:params.k_expand_sim]:
                if sim_score < params.min_sim_cos:
                    continue
                if neighbor_id not in expanded:
                    expanded[neighbor_id] = (sim_score, f"via SIMILAR_TO({chunk_id})")
        
        # Expand via CO_CITED_WITH
        for chunk_id, _ in initial_candidates:
            neighbors = self.cocite_edges.get(chunk_id, [])
            for neighbor_id, weight in neighbors[:params.k_expand_cocite]:
                if neighbor_id not in expanded:
                    expanded[neighbor_id] = (weight, f"via CO_CITED_WITH({chunk_id})")
        
        # Expand via PART_OF (siblings in same group)
        for chunk_id, _ in initial_candidates:
            group_id = self.chunk_to_group.get(chunk_id)
            if group_id:
                siblings = self.group_to_chunks.get(group_id, [])
                for sibling_id in siblings[:params.k_expand_siblings]:
                    if sibling_id != chunk_id and sibling_id not in expanded:
                        expanded[sibling_id] = (0.5, f"via PART_OF({group_id})")
        
        # Convert to list and limit
        result = [
            (chunk_id, score, reason)
            for chunk_id, (score, reason) in expanded.items()
        ]
        
        # Sort by score descending and limit
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:params.max_candidates]
    
    # =========================================================================
    # Scoring
    # =========================================================================
    
    def score_candidates(
        self,
        candidates: list[tuple[str, float, str]],
        initial_candidates: set[str],
        params: ScoringParams | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Score candidates using linear combination of features.
        
        Args:
            candidates: List of (chunk_id, expansion_score, reason) tuples
            initial_candidates: Set of initial candidate IDs
            params: Scoring parameters (uses defaults if None)
            
        Returns:
            List of (chunk_id, final_score, feature_breakdown) tuples, sorted by score
        """
        if params is None:
            params = ScoringParams()
        
        # Collect raw features
        features: dict[str, dict[str, float]] = {}
        for chunk_id, exp_score, reason in candidates:
            features[chunk_id] = {
                "s_ret": exp_score if chunk_id in initial_candidates else 0.0,
                "s_sim": 0.0,
                "s_cocite": 0.0,
                "s_group": 0.0,
                "reason": reason,
            }
        
        # Compute similarity scores (max from initial candidates)
        for chunk_id in features:
            if chunk_id not in initial_candidates:
                for init_id in initial_candidates:
                    neighbors = self.similar_edges.get(init_id, [])
                    for neighbor_id, score in neighbors:
                        if neighbor_id == chunk_id:
                            features[chunk_id]["s_sim"] = max(
                                features[chunk_id]["s_sim"], score
                            )
        
        # Compute co-citation scores
        for chunk_id in features:
            if chunk_id not in initial_candidates:
                for init_id in initial_candidates:
                    neighbors = self.cocite_edges.get(init_id, [])
                    for neighbor_id, weight in neighbors:
                        if neighbor_id == chunk_id:
                            features[chunk_id]["s_cocite"] = max(
                                features[chunk_id]["s_cocite"], weight
                            )
        
        # Compute group coherence (fraction of initial candidates in same group)
        for chunk_id in features:
            group_id = self.chunk_to_group.get(chunk_id)
            if group_id:
                group_members = set(self.group_to_chunks.get(group_id, []))
                overlap = len(group_members & initial_candidates)
                features[chunk_id]["s_group"] = overlap / max(len(initial_candidates), 1)
        
        # Normalize features to [0, 1]
        for feat_name in ["s_ret", "s_sim", "s_cocite", "s_group"]:
            values = [f[feat_name] for f in features.values()]
            max_val = max(values) if values else 1.0
            if max_val > 0:
                for f in features.values():
                    f[feat_name] /= max_val
        
        # Compute final scores
        results = []
        for chunk_id, f in features.items():
            final_score = (
                params.alpha * f["s_ret"]
                + params.beta * f["s_sim"]
                + params.gamma * f["s_cocite"]
                + params.delta * f["s_group"]
            )
            results.append((chunk_id, final_score, f))
        
        # Sort by final score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    # =========================================================================
    # Save / Load
    # =========================================================================
    
    def save(self, output_dir: Path | str) -> None:
        """Save all graph data to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save similar edges
        if self.similar_edges:
            records = []
            for src, neighbors in self.similar_edges.items():
                for dst, score in neighbors:
                    records.append({"src_chunk_id": src, "dst_chunk_id": dst, "score": score})
            pd.DataFrame(records).to_parquet(output_dir / "edges_similar.parquet", index=False)
        
        # Save cocite edges
        if self.cocite_edges:
            records = []
            for src, neighbors in self.cocite_edges.items():
                for dst, weight in neighbors:
                    records.append({"a_chunk_id": src, "b_chunk_id": dst, "weight": weight})
            pd.DataFrame(records).to_parquet(output_dir / "edges_cocite.parquet", index=False)
        
        # Save groups
        if self.group_to_chunks:
            records = [
                {"group_id": gid, "group_type": "law_code" if gid.startswith("code:") else "decision"}
                for gid in self.group_to_chunks.keys()
            ]
            pd.DataFrame(records).to_parquet(output_dir / "groups.parquet", index=False)
        
        # Save chunk_to_group
        if self.chunk_to_group:
            records = [{"chunk_id": cid, "group_id": gid} for cid, gid in self.chunk_to_group.items()]
            pd.DataFrame(records).to_parquet(output_dir / "chunk_to_group.parquet", index=False)
    
    @classmethod
    def load(cls, input_dir: Path | str) -> "GraphBuilder":
        """Load graph data from directory."""
        input_dir = Path(input_dir)
        builder = cls()
        
        # Load similar edges
        similar_path = input_dir / "edges_similar.parquet"
        if similar_path.exists():
            df = pd.read_parquet(similar_path)
            for _, row in df.iterrows():
                src = row["src_chunk_id"]
                if src not in builder.similar_edges:
                    builder.similar_edges[src] = []
                builder.similar_edges[src].append((row["dst_chunk_id"], row["score"]))
        
        # Load cocite edges
        cocite_path = input_dir / "edges_cocite.parquet"
        if cocite_path.exists():
            df = pd.read_parquet(cocite_path)
            for _, row in df.iterrows():
                a, b = row["a_chunk_id"], row["b_chunk_id"]
                weight = row["weight"]
                if a not in builder.cocite_edges:
                    builder.cocite_edges[a] = []
                builder.cocite_edges[a].append((b, weight))
                if b not in builder.cocite_edges:
                    builder.cocite_edges[b] = []
                builder.cocite_edges[b].append((a, weight))
        
        # Load chunk_to_group
        c2g_path = input_dir / "chunk_to_group.parquet"
        if c2g_path.exists():
            df = pd.read_parquet(c2g_path)
            for _, row in df.iterrows():
                chunk_id, group_id = row["chunk_id"], row["group_id"]
                builder.chunk_to_group[chunk_id] = group_id
                if group_id not in builder.group_to_chunks:
                    builder.group_to_chunks[group_id] = []
                builder.group_to_chunks[group_id].append(chunk_id)
        
        return builder
