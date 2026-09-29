from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from metrics import fbeta_set, macro_fbeta, parse_id_list  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_extra_false_match_example(self) -> None:
        score = fbeta_set({"A", "B"}, {"A", "B", "C"})
        self.assertTrue(math.isclose(score, 5 / 7))

    def test_conservative_prediction_example(self) -> None:
        score = fbeta_set({"A", "B"}, {"A"})
        self.assertTrue(math.isclose(score, 5 / 6))

    def test_singleton_rules(self) -> None:
        self.assertEqual(fbeta_set(set(), set()), 1.0)
        self.assertEqual(fbeta_set(set(), {"A"}), 0.0)
        self.assertEqual(fbeta_set({"A"}, set()), 0.0)

    def test_macro_weights_entities_equally(self) -> None:
        truth = {"S1-a": set(), "S1-b": {"A", "B"}}
        prediction = {"S1-a": {"X"}, "S1-b": {"A", "B"}}
        self.assertEqual(macro_fbeta(truth, prediction), 0.5)

    def test_parser_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            parse_id_list("S2-1,S2-1")


if __name__ == "__main__":
    unittest.main()

