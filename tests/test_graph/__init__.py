"""Test fixtures for graph module tests."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_laws_csv(tmp_path):
    """Create a sample laws_de.csv file."""
    data = {
        "citation": [
            "Art. 1 ZGB",
            "Art. 2 ZGB",
            "Art. 11 Abs. 1 OR",
            "Art. 11 Abs. 2 OR",
            "Art. 41 Abs. 1 OR",
        ],
        "text": [
            "Das Recht wird durch das Gesetz geregelt. Die Rechtsfähigkeit des Menschen beginnt mit dem Leben.",
            "Jedermann handelt in gutem Glauben. Der gute Glaube wird vermutet.",
            "Zum Abschlusse eines Vertrages ist die übereinstimmende gegenseitige Willensäusserung der Parteien erforderlich.",
            "Sie kann eine ausdrückliche oder stillschweigende sein.",
            "Wer einem andern widerrechtlich Schaden zufügt, sei es mit Absicht, sei es aus Fahrlässigkeit, wird ihm zum Ersatze verpflichtet.",
        ],
    }
    csv_path = tmp_path / "laws_de.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_courts_csv(tmp_path):
    """Create a sample court_considerations.csv file."""
    data = {
        "citation": [
            "BGE 116 Ia 56 E. 2b",
            "BGE 119 II 449 E. 3.4",
            "BGE 145 II 32 E. 3.1",
            "5A_800/2019 E. 2",
            "5A_123/2020 E. 3.1",
        ],
        "text": [
            "Die Vorinstanz hat festgestellt, dass der Beschwerdeführer die Frist versäumt hat. Diese Feststellung ist nicht willkürlich.",
            "Nach der bundesgerichtlichen Rechtsprechung ist ein Vertrag dann zustande gekommen, wenn die Parteien sich über die wesentlichen Punkte geeinigt haben.",
            "Le recourant fait valoir une violation de son droit d'être entendu. Ce grief est mal fondé.",
            "Il ricorrente contesta la valutazione delle prove effettuata dall'autorità cantonale.",
            "Das Bundesgericht prüft die Anwendung des Bundesrechts frei.",
        ],
    }
    csv_path = tmp_path / "court_considerations.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_data_dir(tmp_path, sample_laws_csv, sample_courts_csv):
    """Create a data directory with both CSV files."""
    return tmp_path


@pytest.fixture
def sample_chunks_df():
    """Create a sample chunks DataFrame."""
    return pd.DataFrame({
        "chunk_id": [
            "Art. 1 ZGB",
            "Art. 2 ZGB",
            "Art. 11 Abs. 1 OR",
            "BGE 116 Ia 56 E. 2b",
            "5A_800/2019 E. 2",
        ],
        "chunk_type": ["law", "law", "law", "case", "case"],
        "group_id": ["code:ZGB", "code:ZGB", "code:OR", "bge:116_Ia_56", "docket:5A_800_2019"],
        "lang": ["de", "de", "de", "de", "it"],
        "text_raw": [
            "Das Recht wird durch das Gesetz geregelt.",
            "Jedermann handelt in gutem Glauben.",
            "Zum Abschlusse eines Vertrages ist die übereinstimmende Willensäusserung erforderlich.",
            "Die Vorinstanz hat festgestellt, dass der Beschwerdeführer die Frist versäumt hat.",
            "Il ricorrente contesta la valutazione delle prove.",
        ],
    })


@pytest.fixture
def sample_summaries_df():
    """Create a sample summaries DataFrame."""
    records = []
    chunk_ids = ["Art. 1 ZGB", "Art. 2 ZGB", "Art. 11 Abs. 1 OR"]
    
    for chunk_id in chunk_ids:
        records.extend([
            {
                "chunk_id": chunk_id,
                "summary_type": "short",
                "summary_text": f"Short summary for {chunk_id}",
                "entities": [],
            },
            {
                "chunk_id": chunk_id,
                "summary_type": "retrieval",
                "summary_text": f"Topic: Legal provision | Terms: law, contract | Context: Swiss law",
                "entities": [],
            },
            {
                "chunk_id": chunk_id,
                "summary_type": "entities",
                "summary_text": "law; contract; ZGB",
                "entities": ["law", "contract", "ZGB"],
            },
        ])
    
    return pd.DataFrame(records)


@pytest.fixture
def sample_train_df():
    """Create a sample training DataFrame with gold citations."""
    return pd.DataFrame({
        "query_id": ["q1", "q2", "q3"],
        "query": [
            "What are the requirements for contract formation?",
            "When does legal capacity begin?",
            "What is the standard of care for negligence?",
        ],
        "gold_citations": [
            "Art. 1 OR;Art. 11 Abs. 1 OR;BGE 119 II 449 E. 3.4",
            "Art. 1 ZGB;Art. 2 ZGB",
            "Art. 41 Abs. 1 OR;BGE 116 Ia 56 E. 2b;Art. 1 OR",
        ],
    })
