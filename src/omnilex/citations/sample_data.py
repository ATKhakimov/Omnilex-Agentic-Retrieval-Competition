"""Sample data for testing and development.

Shared sample corpus definitions used by:
- scripts/download_data.py (create_sample_data function)
- tests/conftest.py (pytest fixtures)
"""

SAMPLE_LAWS = [
    {
        "citation": "Art. 1 OR",
        "text": (
            "Zum Abschlusse eines Vertrages ist die übereinstimmende "
            "gegenseitige Willensäusserung der Parteien erforderlich. "
            "Sie kann eine ausdrückliche oder stillschweigende sein."
        ),
        "title": "Obligationenrecht - Abschluss des Vertrages",
    },
    {
        "citation": "Art. 1 ZGB",
        "text": (
            "Das Gesetz findet auf alle Rechtsfragen Anwendung, "
            "für die es nach Wortlaut oder Auslegung eine Bestimmung enthält."
        ),
        "title": "Zivilgesetzbuch - Anwendung des Rechts",
    },
    {
        "citation": "Art. 117 StGB",
        "text": (
            "Wer fahrlässig den Tod eines Menschen verursacht, "
            "wird mit Freiheitsstrafe bis zu drei Jahren oder Geldstrafe bestraft."
        ),
        "title": "Strafgesetzbuch - Fahrlässige Tötung",
    },
]

SAMPLE_COURTS = [
    {
        "citation": "BGE 119 II 449 E. 3.4",
        "text": (
            "Der Vertrag kommt durch übereinstimmende Willenserklärungen zustande. "
            "Die Annahme kann auch durch schlüssiges Verhalten erfolgen."
        ),
    },
    {
        "citation": "BGE 116 Ia 56 E. 8.2.1",
        "text": (
            "Die Meinungsfreiheit schützt die freie Bildung und Äusserung "
            "von Meinungen sowie deren Verbreitung."
        ),
    },
]

# Sample queries for training data
SAMPLE_TRAIN_QUERIES = [
    {
        "query_id": "train_0001",
        "query": "What are the requirements for a valid contract under Swiss law?",
        "gold_citations": "Art. 1 OR;Art. 11 OR;BGE 119 II 449 E. 3.4",
    },
    {
        "query_id": "train_0002",
        "query": "When can a marriage be annulled in Switzerland?",
        "gold_citations": "Art. 104 ZGB;Art. 105 ZGB;BGE 121 III 38 E. 2b",
    },
    {
        "query_id": "train_0003",
        "query": "What constitutes negligent homicide under Swiss criminal law?",
        "gold_citations": "Art. 117 StGB;BGE 116 IV 306 E. 2.1",
    },
]

# Sample queries for test data (no gold citations)
SAMPLE_TEST_QUERIES = [
    {
        "query_id": "test_0001",
        "query": "What are the grounds for divorce in Swiss law?",
    },
    {
        "query_id": "test_0002",
        "query": "How is inheritance distributed under Swiss law?",
    },
]

# Sample submission format
SAMPLE_SUBMISSION = [
    {"query_id": "test_0001", "predicted_citations": "Art. 111 ZGB;Art. 114 ZGB"},
    {"query_id": "test_0002", "predicted_citations": "Art. 457 ZGB;Art. 462 ZGB"},
]
