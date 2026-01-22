# Omnilex Agentic Retrieval Competition

A Kaggle competition for Swiss legal citation retrieval. Given a legal query in English, retrieve relevant Swiss federal law and Federal Court decision (BGE) citations.

## Overview

| Aspect         | Details                                     |
| -------------- | ------------------------------------------- |
| **Task**       | Cross-lingual legal citation retrieval      |
| **Input**      | Legal queries (English)                     |
| **Output**     | Swiss legal citations (law/BGE format)      |
| **Metric**     | Macro F1 (average F1 across queries)        |
| **Submission** | Kaggle Notebook (code competition)          |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Omnilex-AI/Omnilex-Agentic-Retrieval-Competition.git
cd Omnilex-Agentic-Retrieval-Competition

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing/linting

# Install package in development mode
pip install -e .
```

### Download Data

```bash
# Download training data and create sample files
python scripts/download_data.py

# Build search indices
python scripts/build_indices.py
```

### Run Baselines

Two baseline notebooks are provided:

1. **Direct Generation** (`notebooks/01_direct_generation_baseline.ipynb`)
   - Prompts LLM to directly generate citations
   - Simple but prone to hallucination

2. **Agentic Retrieval** (`notebooks/02_agentic_retrieval_baseline.ipynb`)
   - Uses ReAct-style agent with search tools
   - Grounded in actual legal documents

Both notebooks work in VSCode and can be submitted to Kaggle.

### Validate Submission

```bash
python scripts/validate_submission.py submission.csv
python scripts/validate_submission.py submission.csv --gold gold.csv  # with scoring
```

## Data Format

### Training Data (`train.csv`)

| Column           | Description                                  |
| ---------------- | -------------------------------------------- |
| `query_id`       | Unique query identifier                      |
| `query`          | Legal question (English)                     |
| `gold_citations` | Ground truth citations (semicolon-separated) |

### Test Data (`test.csv`)

| Column     | Description              |
| ---------- | ------------------------ |
| `query_id` | Unique query identifier  |
| `query`    | Legal question (English) |

### Submission Format (`submission.csv`)

```csv
query_id,predicted_citations
q_001,"Art. 1 ZGB;BGE 116 Ia 56"
q_002,"Art. 117 StGB"
```

## Citation Formats

### Federal Laws

Format: `Art. [article] [Abs. paragraph] [LAW]`

Examples:

- `Art. 1 ZGB` - Civil Code, Article 1
- `Art. 11 Abs. 2 OR` - Code of Obligations, Art. 11 para. 2
- `Art. 117 StGB` - Criminal Code, Article 117

All 1026 Swiss law abbreviations are supported, loaded from `data/abbrev-translations.json`.

Common abbreviations: ZGB (Civil Code), OR (Code of Obligations), StGB (Criminal Code), BV (Constitution)

### Court Decisions (BGE)

Format: `BGE [volume] [section] [page] [E. consideration]`

Examples:

- `BGE 116 Ia 56` - Volume 116, Constitutional Law section, page 56
- `BGE 121 III 38 E. 2b` - With consideration reference
- `BGE 141 III 513 E. 5.3.1` - With decimal consideration reference

See `docs/swiss_citation_primer.md` for detailed format documentation.

## Evaluation

The primary metric is **Macro F1**:

1. For each query, compute F1 between predicted and gold citation sets
2. Average F1 scores across all queries

```python
from omnilex.evaluation import macro_f1

scores = macro_f1(predictions, gold)
print(f"Macro F1: {scores['macro_f1']:.4f}")
```

## Project Structure

```
├── src/omnilex/           # Core library
│   ├── citations/         # Citation parsing & normalization
│   ├── evaluation/        # Metrics & scoring
│   ├── retrieval/         # BM25 search & tools
│   └── llm/               # LLM loading & prompts
├── notebooks/             # Baseline notebooks
├── scripts/               # Data & utility scripts
├── tests/                 # Test suite
├── docs/                  # Documentation
└── data/                  # Data directory
```

## Requirements

- Python >= 3.10
- llama-cpp-python (for local LLM inference)
- rank-bm25 (for keyword search)
- pandas, numpy, scikit-learn

For Kaggle submissions, you'll need to:

1. Upload your GGUF model as a Kaggle dataset
2. Upload pre-built indices as a Kaggle dataset
3. Package the `omnilex` library

## Rules Summary

- Code competition: Submit Kaggle notebooks
- No internet access during evaluation
- External pretrained models allowed (uploaded as datasets)
- Teams up to 5 members
- See `docs/competition_rules.md` for full rules

## Resources

- [Swiss Citation Format Primer](docs/swiss_citation_primer.md)
- [Competition Rules](docs/competition_rules.md)
- [Submission Guide](docs/submission_guide.md)
- [LEXam Dataset](https://huggingface.co/datasets/LEXam-Benchmark/LEXam)
- [Fedlex Portal](https://www.fedlex.admin.ch/)

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Contact

For questions about the competition, please open an issue on this repository.
