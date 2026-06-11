"""Tests for the frontend data compiler."""

from pathlib import Path
import tempfile
import unittest

from scripts.build_frontend_data import (
    canonical_team,
    discover_latest_snapshot,
    fixture_key,
    normalize_grid,
    parse_kickoff_utc,
    prediction_metrics,
    repository_tree_url,
)


class FrontendDataCompilerTests(unittest.TestCase):
    """Validate data normalization and snapshot selection."""

    def test_latest_snapshot_uses_iso_dates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "2026-06-10").mkdir()
            (root / "notes").mkdir()
            latest = root / "2026-06-11"
            latest.mkdir()

            self.assertEqual(discover_latest_snapshot(root), latest)

    def test_team_aliases_and_fixture_orientation(self):
        self.assertEqual(canonical_team("USA"), "United States")
        self.assertEqual(
            fixture_key("2026-06-12", "USA", "Paraguay"),
            fixture_key("2026-06-12", "Paraguay", "United States"),
        )
        self.assertEqual(
            canonical_team("Bosnia & Herzegovina"), "Bosnia and Herzegovina"
        )

    def test_kickoff_offset_is_converted_to_utc(self):
        self.assertEqual(
            parse_kickoff_utc("2026-06-11", "13:00 UTC-6"),
            "2026-06-11T19:00:00Z",
        )

    def test_score_grid_is_normalized_and_metrics_are_derived(self):
        normalized = normalize_grid([[2.0, 1.0], [1.0, 0.0]])
        self.assertAlmostEqual(sum(map(sum, normalized)), 1.0)

        metrics = prediction_metrics([[2.0, 1.0], [1.0, 0.0]])
        self.assertEqual(metrics["mostLikelyScore"], [0, 0])
        self.assertAlmostEqual(metrics["expectedGoals"]["home"], 0.25)
        self.assertAlmostEqual(metrics["expectedGoals"]["away"], 0.25)

    def test_latest_snapshot_source_url_uses_selected_iso_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            older = root / "2026-06-10"
            latest = root / "2026-06-11"
            older.mkdir()
            latest.mkdir()

            self.assertEqual(discover_latest_snapshot(root).name, "2026-06-11")
            # URL is dynamically detected from git remote, so just verify it contains the path
            url = repository_tree_url(Path("outputs/predictions/2026-06-11"))
            self.assertIn("/tree/main/outputs/predictions/2026-06-11", url)
            self.assertTrue(url.startswith("https://github.com/"))


if __name__ == "__main__":
    unittest.main()
