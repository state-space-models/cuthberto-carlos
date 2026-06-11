"""Tests for the frontend data compiler."""

from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from scripts.build_frontend_data import (
    apply_actual_result_overrides,
    canonical_team,
    discover_latest_snapshot,
    fixture_key,
    get_repository_slug,
    get_repository_url,
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
            url = repository_tree_url(Path("outputs/predictions/2026-06-11"))
            self.assertEqual(
                url,
                "https://github.com/state-space-models/cuthberto-carlos/"
                "tree/main/outputs/predictions/2026-06-11",
            )

    def test_repository_uses_actions_metadata_with_canonical_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                get_repository_slug(), "state-space-models/cuthberto-carlos"
            )
        with patch.dict(
            os.environ, {"GITHUB_REPOSITORY": "example-org/example-repo"}
        ):
            self.assertEqual(get_repository_slug(), "example-org/example-repo")
            self.assertEqual(
                get_repository_url(),
                "https://github.com/example-org/example-repo",
            )

    def test_actual_result_overrides_are_merged_into_schedule_fixture(self):
        fixtures = {
            fixture_key("2026-06-11", "Mexico", "South Africa"): {
                "date": "2026-06-11",
                "team1": "Mexico",
                "team2": "South Africa",
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            overrides_path = Path(temporary_directory) / "actual_results.json"
            overrides_path.write_text(
                '{"matches":[{"date":"2026-06-11","team1":"Mexico",'
                '"team2":"South Africa","score":{"ft":[1,0]}}]}'
            )
            apply_actual_result_overrides(fixtures, overrides_path)

        fixture = fixtures[fixture_key("2026-06-11", "Mexico", "South Africa")]
        self.assertEqual(fixture["score"]["ft"], [1, 0])


if __name__ == "__main__":
    unittest.main()
