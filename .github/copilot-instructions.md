# Copilot Instructions for Omnilex Agentic Retrieval Competition

## Project Overview
**Kaggle code competition** for Swiss legal citation retrieval. Given an English legal query, predict relevant Swiss legal sources as semicolon-separated German citations. Primary metric: **Macro F1** at citation level.

**Constraints:** Kaggle offline notebook ≤12h runtime. Output only citations (no text generation). Citation strings must be exact canonical matches.

---

## Target Architecture: Summary-centric KG + Graph-augmented Retrieval

### Data Model (Entities)

#### SourceChunk (1:1 with corpus rows)
```python
# Artifact: data/processed/chunks.parquet
chunk_id: str      # = citation (canonical, unique)
chunk_type: Literal["law", "case"]
group_id: str | None  # DocGroup FK
lang: Literal["de", "fr", "it", "unknown"]
text_raw: str
```
- Source: `laws_de.csv` → `chunk_type=law`, `court_considerations.csv` → `chunk_type=case`

#### Summary (per-chunk, multiple types)
```python
# Artifact: data/processed/summaries.parquet
chunk_id: str
summary_type: Literal["short", "retrieval", "entities"]
summary_text: str
entities: list[str]  # only for type=entities
```
- `short`: 1-2 sentences, 200-400 chars
- `retrieval`: structured (topic, conditions, exceptions, terms), 400-1200 chars
- `entities`: 5-30 keyphrases/terms

#### DocGroup (structural skeleton)
```python
# Artifact: data/processed/groups.parquet, chunk_to_group.parquet
group_id: str   # e.g., "code:OR", "bge:145_II_32", "docket:5A_800_2019"
group_type: Literal["law_code", "decision"]
```
- Extract via regex from `chunk_id`: `Art. ... <CODE>` → CODE, `BGE X Y Z` → decision ID

### Edges (Graph)

| Edge Type | Description | Artifact |
|-----------|-------------|----------|
| `SIMILAR_TO` | cosine similarity between summary embeddings | `edges_similar.parquet` |
| `CO_CITED_WITH` | co-occurrence in `train.csv.gold_citations` | `edges_cocite.parquet` |
| `PART_OF` | chunk → group membership | `chunk_to_group.parquet` |

---

## Implementation Modules

### M1: Ingestor (`src/omnilex/graph/ingestor.py`)
**Input:** `laws_de.csv`, `court_considerations.csv`  
**Output:** `chunks.parquet`
- Normalize text (trim, dedupe spaces)
- Detect language (heuristic: stopwords or `langdetect`)
- Ensure unique `chunk_id`

### M2: Summarizer (`src/omnilex/graph/summarizer.py`)
**Input:** `chunks.parquet`  
**Output:** `summaries.parquet`

Two modes:
1. **LLM mode** (for testing via API): batch generation
2. **Heuristic mode** (for Kaggle): 
   - `short`: first N sentences + compression
   - `retrieval`: keyphrase extraction (YAKE/TF-IDF) + template
   - `entities`: top keyphrases + regex (`Art.|BGE|Abs.|E.`)

### M3: Embedder & Index Builder (`src/omnilex/graph/embedder.py`)
**Input:** `summaries.parquet` (use `summary_type=retrieval`)  
**Output:** `embeddings.npy`, `faiss_index.bin`, `bm25_index.pkl`

**Default model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Configurable via `EMBEDDING_MODEL` constant
- Build embeddings from `summary_retrieval` text
- Store `row_idx → chunk_id` mapping
- Target: <50ms/query on CPU

### M4: Graph Builder (`src/omnilex/graph/graph_builder.py`)

#### SIMILAR_TO edges
- FAISS kNN on embeddings, k=20-100
- Filter by `min_cos` threshold

#### CO_CITED_WITH edges
- Parse `train.csv.gold_citations`, count co-occurrences
- Weight: `count / sqrt(freq_i * freq_j)` or PMI
- Keep topM neighbors per node (~50)

#### DocGroup mapping
- Regex parse `chunk_id` to extract group

---

## Inference Pipeline

### Step 1: Candidate Generation (Recall)
```python
# Multi-channel retrieval
candidates_bm25 = bm25_index.search(query, top_k=topN_bm25)
candidates_vec = faiss_index.search(embed(query), top_k=topN_vec)
C0 = reciprocal_rank_fusion(candidates_bm25, candidates_vec)
```

### Step 2: Graph Expansion
```python
C1 = set(C0)
for c in C0:
    C1 |= get_similar_neighbors(c, k=k_expand_sim)
    C1 |= get_cocite_neighbors(c, k=k_expand_cocite)
    C1 |= get_group_siblings(c, k=k_expand_siblings)
C1 = C1[:max_candidates_after_expansion]  # Prevent explosion
```

**Expansion Parameters (defaults / tuning range):**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `k_expand_sim` | 20 | 10–50 | Similar neighbors per candidate |
| `k_expand_cocite` | 30 | 10–100 | Co-cited neighbors per candidate |
| `k_expand_siblings` | 10 | 0–30 | DocGroup siblings per candidate |
| `min_sim_cos` | 0.25 | 0.15–0.35 | Cosine threshold for SIMILAR_TO |
| `max_candidates_after_expansion` | 800 | 300–2000 | Cap to prevent timeout |

### Step 3: Scoring / Reranking
```python
# Normalize features to [0,1] before combining!
score = α*norm(s_ret) + β*norm(s_sim) + γ*norm(s_cocite) + δ*norm(s_group)
```

**Scoring Parameters (defaults / tuning range):**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `α` (retrieval) | 1.0 | 0.5–2.0 | BM25/vector score weight |
| `β` (similarity) | 0.6 | 0.0–1.5 | SIMILAR_TO edge weight |
| `γ` (co-citation) | 0.8 | 0.0–2.0 | CO_CITED_WITH edge weight |
| `δ` (docgroup) | 0.2 | 0.0–0.8 | Group coherence bonus |

Features per candidate (normalize each to [0,1]):
- `s_ret`: retrieval score (BM25/vector)
- `s_sim`: max similarity edge from initial candidates
- `s_cocite`: max co-citation weight from initial candidates
- `s_group`: group coherence bonus

### Step 4: Output
```python
# Top K candidates → semicolon-separated string
predictions = ";".join([c.chunk_id for c in sorted_candidates[:K]])
```

---

## LLM Integration (Dual Mode)

```python
# src/omnilex/llm/loader.py
def get_llm(mode: Literal["local", "api"] = "auto"):
    if mode == "api" or (mode == "auto" and not is_kaggle_env()):
        return APIClient()  # For testing/dev
    else:
        return load_gguf_model()  # For Kaggle submission
```

---

## Key Commands
```bash
pip install -e .
pytest tests/ -v
python scripts/evaluate_submission.py submission.csv --split val
python utils/build_indices.py
```

## Artifacts Checklist

| Artifact | Location | Built By |
|----------|----------|----------|
| `chunks.parquet` | `data/processed/` | M1: Ingestor |
| `summaries.parquet` | `data/processed/` | M2: Summarizer |
| `embeddings.npy` | `data/processed/` | M3: Embedder |
| `faiss_index.bin` | `data/processed/` | M3: Embedder |
| `bm25_index.pkl` | `data/processed/` | M3: Embedder |
| `edges_similar.parquet` | `data/processed/` | M4: Graph |
| `edges_cocite.parquet` | `data/processed/` | M4: Graph |
| `groups.parquet` | `data/processed/` | M4: Graph |
| `debug_runs.jsonl` | `output/` | Inference |

## Debug Logging
For each query, log to `debug_runs.jsonl`:
- Top retrieval candidates (BM25/vec scores)
- Expanded candidates with expansion reason (`via SIMILAR_TO(X)`, `via CO_CITED_WITH(Y)`, `via PART_OF(Z)`)
- Final topK with feature breakdown

## Citation Format (Critical)
- ✅ `Art. 11 Abs. 2 OR`
- ❌ `Art. 11 OR Abs. 2` (wrong order)
- ❌ `Art. 11 Absatz 2 OR` (use `Abs.` not `Absatz`)

Use `CitationNormalizer` from `src/omnilex/citations/` for parsing.

## MVP vs Extensions

### MVP (implement first)
- Heuristic `summary_retrieval`
- Embeddings + FAISS index
- `SIMILAR_TO` + `CO_CITED_WITH` edges
- Linear scoring with α,β,γ,δ tuning

### V2 (after MVP works)
- Entity extraction + entity-based bonuses
- Separate indices for law/case
- LightGBM ranker
