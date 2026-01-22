# Omnilex Agentic Retrieval Competition Starter Repo

Official starter repo for Kaggle competiton https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/host/launch-checklist

## Quick Start

### Installation

(Tested with Ubuntu-24.04 in WSL)

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

Get it from Kaggle into `data` directory

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
```

## Data Format

See Kaggle

## Project Structure

```
├── src/omnilex/           # Core library
│   ├── citations/         # Citation parsing & normalization
│   ├── evaluation/        # Metrics & scoring
│   ├── retrieval/         # BM25 search & tools
│   └── llm/               # LLM loading & prompts
├── notebooks/             # Baseline notebooks
├── utils/                 # Data & utility scripts
├── tests/                 # Test suite
└── data/                  # Data directory
```

## Requirements

- Python >= 3.10
- llama-cpp-python (for local LLM inference)
- rank-bm25 (for keyword search)
- pandas, numpy, scikit-learn

For Kaggle submissions, you may need to (depending on your solution):

1. Upload your GGUF model as a Kaggle dataset
2. Upload pre-built indices as a Kaggle dataset
3. Package the `omnilex` library

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Contact

For public questions about the competition please use the "Discussion" tab or open an issue on this repository. For private questions reahc out to host on Kaggle or ari.jordan@omnilex.ai
