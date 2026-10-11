from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from voxcaps.data_refinement.ranking import rank_scores, rank_preferences
from voxcaps.pipeline import prepare_record
import test_core
from test_core import raw, write_wav, unknown_attributes


class RankingTests(unittest.TestCase):
    def test_existing_scores_sorted_with_explicit_ties(self):
        records = [{"id": "c", "quality_score": {"mos": 4}}, {"id": "b", "quality_score": {"mos": 3}},
                   {"id": "a", "quality_score": {"mos": 4}}]
        result = rank_scores(records, field="quality_score.mos")
        self.assertEqual([r["id"] for r in result["ranking"]], ["a", "c", "b"])
        self.assertEqual([r["rank"] for r in result["ranking"]], [1, 1, 3])

    def test_invalid_scores_and_duplicate_ids_rejected(self):
        for records in ([{"id": "a", "score": float("nan")}],
                        [{"id": "a", "score": 1}, {"id": "a", "score": 2}]):
            with self.assertRaises(ValueError): rank_scores(records)

    def test_existing_preferences_sorted_into_topological_frontiers(self):
        pairs = [{"preferred": "a", "dispreferred": "b"}, {"preferred": "a", "dispreferred": "c"},
                 {"preferred": "b", "dispreferred": "d"}, {"preferred": "c", "dispreferred": "d"}]
        self.assertEqual(rank_preferences(pairs + pairs)["tiers"], [["a"], ["b", "c"], ["d"]])

    def test_conflicting_preferences_have_no_invented_order(self):
        with self.assertRaisesRegex(ValueError, "contains_cycle"):
            rank_preferences([{"preferred": "a", "dispreferred": "b"}, {"preferred": "b", "dispreferred": "a"}])

    def test_automatic_preparation_has_no_manual_validity_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            write_wav(base / "mix.wav")
            record = {"id": "r", "audio": "mix.wav", "annotation": raw(),
                      "quality_score": test_core.QualityTests.boundary, "attribute_annotations": unknown_attributes(),
                      "human_valid": False}
            result = prepare_record(record, base)
            self.assertNotIn("human_valid", result)
            self.assertNotIn("human_review", result)


if __name__ == "__main__": unittest.main()
