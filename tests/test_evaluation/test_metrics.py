"""Tests for evaluation metrics."""

import pytest

from omnilex.evaluation.metrics import (
    average_precision,
    citation_f1,
    macro_f1,
    mean_average_precision,
    micro_f1,
    ndcg_at_k,
)


class TestCitationF1:
    """Tests for single-query F1 calculation."""

    def test_perfect_match(self):
        """Test F1 with perfect match."""
        pred = ["SR 210 Art. 1", "BGE 116 Ia 56 E. 2b"]
        gold = ["SR 210 Art. 1", "BGE 116 Ia 56 E. 2b"]

        result = citation_f1(pred, gold)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_match(self):
        """Test F1 with partial match."""
        pred = ["SR 210 Art. 1", "SR 220 Art. 1"]
        gold = ["SR 210 Art. 1", "BGE 116 Ia 56 E. 2b"]

        result = citation_f1(pred, gold)

        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5

    def test_no_match(self):
        """Test F1 with no overlap."""
        pred = ["SR 210 Art. 1"]
        gold = ["BGE 116 Ia 56 E. 2b"]

        result = citation_f1(pred, gold)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_empty_prediction(self):
        """Test F1 with empty prediction."""
        pred = []
        gold = ["SR 210 Art. 1"]

        result = citation_f1(pred, gold)

        assert result["f1"] == 0.0

    def test_empty_gold(self):
        """Test F1 with empty gold."""
        pred = ["SR 210 Art. 1"]
        gold = []

        result = citation_f1(pred, gold)

        # Predicting when nothing expected is wrong
        assert result["f1"] == 0.0

    def test_both_empty(self):
        """Test F1 when both empty."""
        result = citation_f1([], [])

        # Perfect "match" when nothing expected and nothing predicted
        assert result["f1"] == 1.0

    def test_more_predictions_than_gold(self):
        """Test F1 when predicting more than expected."""
        pred = ["SR 210 Art. 1", "SR 220 Art. 1", "BGE 116 Ia 56 E. 2b"]
        gold = ["SR 210 Art. 1"]

        result = citation_f1(pred, gold)

        # Precision: 1/3, Recall: 1/1
        assert result["precision"] == pytest.approx(1 / 3)
        assert result["recall"] == 1.0

    def test_fewer_predictions_than_gold(self):
        """Test F1 when predicting less than expected."""
        pred = ["SR 210 Art. 1"]
        gold = ["SR 210 Art. 1", "SR 220 Art. 1", "BGE 116 Ia 56 E. 2b"]

        result = citation_f1(pred, gold)

        # Precision: 1/1, Recall: 1/3
        assert result["precision"] == 1.0
        assert result["recall"] == pytest.approx(1 / 3)


class TestMacroF1:
    """Tests for Macro F1 calculation."""

    def test_macro_f1_perfect(self, sample_predictions, sample_gold):
        """Test macro F1 with provided fixtures."""
        # Use same predictions as gold for perfect score
        result = macro_f1(sample_gold, sample_gold)

        assert result["macro_f1"] == 1.0

    def test_macro_f1_basic(self):
        """Test basic macro F1 calculation."""
        predictions = [
            ["SR 210 Art. 1"],  # Perfect match
            ["BGE 116 Ia 56 E. 2b"],  # No match
        ]
        gold = [
            ["SR 210 Art. 1"],
            ["SR 220 Art. 1"],
        ]

        result = macro_f1(predictions, gold)

        # Query 1: F1=1.0, Query 2: F1=0.0
        # Macro F1 = (1.0 + 0.0) / 2 = 0.5
        assert result["macro_f1"] == 0.5

    def test_macro_f1_length_mismatch(self):
        """Test that length mismatch raises error."""
        predictions = [["SR 210 Art. 1"]]
        gold = [["SR 210 Art. 1"], ["BGE 116 Ia 56 E. 2b"]]

        with pytest.raises(ValueError):
            macro_f1(predictions, gold)

    def test_macro_f1_empty_lists(self):
        """Test macro F1 with empty lists."""
        result = macro_f1([], [])

        assert result["macro_f1"] == 0.0


class TestMicroF1:
    """Tests for Micro F1 calculation."""

    def test_micro_f1_basic(self):
        """Test basic micro F1 calculation."""
        predictions = [
            ["SR 210 Art. 1", "BGE 116 Ia 56 E. 2b"],  # 2 correct
            ["SR 220 Art. 1"],  # 0 correct
        ]
        gold = [
            ["SR 210 Art. 1", "BGE 116 Ia 56 E. 2b"],
            ["BGE 119 II 449 E. 3.4"],
        ]

        result = micro_f1(predictions, gold)

        # TP=2, FP=1, FN=1
        # Precision = 2/3, Recall = 2/3
        assert result["micro_precision"] == pytest.approx(2 / 3)
        assert result["micro_recall"] == pytest.approx(2 / 3)


class TestAveragePrecision:
    """Tests for Average Precision calculation."""

    def test_ap_perfect_ranking(self):
        """Test AP with perfect ranking."""
        predicted = ["A", "B", "C"]
        gold = ["A", "B", "C"]

        ap = average_precision(predicted, gold)

        # All relevant at positions 1, 2, 3
        # AP = (1/1 + 2/2 + 3/3) / 3 = 1.0
        assert ap == 1.0

    def test_ap_imperfect_ranking(self):
        """Test AP with imperfect ranking."""
        predicted = ["X", "A", "Y", "B"]
        gold = ["A", "B"]

        ap = average_precision(predicted, gold)

        # Relevant at positions 2 and 4
        # AP = (1/2 + 2/4) / 2 = 0.5
        assert ap == 0.5

    def test_ap_empty_gold(self):
        """Test AP with empty gold."""
        ap = average_precision(["A", "B"], [])

        assert ap == 0.0


class TestMAP:
    """Tests for Mean Average Precision calculation."""

    def test_map_basic(self):
        """Test basic MAP calculation."""
        predictions = [
            ["A", "B"],  # Perfect
            ["X", "A"],  # AP = 0.5
        ]
        gold = [
            ["A", "B"],
            ["A"],
        ]

        result = mean_average_precision(predictions, gold)

        # MAP = (1.0 + 1/2) / 2 = 0.75
        assert result == pytest.approx(0.75)


class TestNDCG:
    """Tests for NDCG calculation."""

    def test_ndcg_perfect(self):
        """Test NDCG with perfect ranking."""
        predicted = ["A", "B", "C"]
        gold = ["A", "B", "C"]

        ndcg = ndcg_at_k(predicted, gold, k=3)

        assert ndcg == 1.0

    def test_ndcg_imperfect(self):
        """Test NDCG with imperfect ranking."""
        predicted = ["X", "A", "B"]
        gold = ["A", "B"]

        ndcg = ndcg_at_k(predicted, gold, k=3)

        # DCG: 0 + 1/log2(3) + 1/log2(4)
        # IDCG: 1/log2(2) + 1/log2(3)
        assert 0 < ndcg < 1

    def test_ndcg_empty_gold(self):
        """Test NDCG with empty gold."""
        ndcg = ndcg_at_k(["A", "B"], [], k=3)

        assert ndcg == 0.0
