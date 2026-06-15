"""Tests for the frontend data compiler."""

from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request

from scripts.build_frontend_data import (
    canonical_scorer_name,
    canonical_team,
    collect_prediction_versions,
    discover_prediction_snapshots,
    discover_latest_snapshot,
    extract_actual_result,
    fetch_schedule,
    fetch_squads,
    fixture_key,
    get_repository_slug,
    get_repository_url,
    normalize_grid,
    normalize_player_name,
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
            latest = root / "2026-06-11"
            latest.mkdir()

            self.assertEqual(discover_latest_snapshot(root), latest)

    def test_snapshot_discovery_returns_all_dates_in_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latest = root / "2026-06-15"
            older = root / "2026-06-11"
            latest.mkdir()
            older.mkdir()

            self.assertEqual(
                discover_prediction_snapshots(root),
                [older, latest],
            )

    def test_snapshot_discovery_rejects_malformed_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "2026-06-11").mkdir()
            (root / "notes").mkdir()

            with self.assertRaisesRegex(
                ValueError, "Invalid prediction snapshot directory"
            ):
                discover_prediction_snapshots(root)

    def test_prediction_versions_allow_incomplete_latest_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            older = root / "2026-06-11"
            latest = root / "2026-06-15"
            older.mkdir()
            latest.mkdir()
            fixtures = {
                fixture_key("2026-06-18", "Mexico", "South Korea"): {},
                fixture_key("2026-06-11", "Mexico", "South Africa"): {},
            }

            def write_prediction(snapshot, folder, date, home, away):
                destination = snapshot / folder
                destination.mkdir()
                (destination / "match_data.json").write_text(
                    json.dumps(
                        {"date": date, "home_team": home, "away_team": away}
                    )
                )
                (destination / "predictions.json").write_text("{}")

            write_prediction(
                older,
                "mexico-south-korea",
                "2026-06-18",
                "Mexico",
                "South Korea",
            )
            write_prediction(
                older,
                "mexico-south-africa",
                "2026-06-11",
                "Mexico",
                "South Africa",
            )
            write_prediction(
                latest,
                "mexico-south-korea",
                "2026-06-18",
                "Mexico",
                "South Korea",
            )

            versions = collect_prediction_versions([older, latest], fixtures)

            shared = versions[
                fixture_key("2026-06-18", "Mexico", "South Korea")
            ]
            completed = versions[
                fixture_key("2026-06-11", "Mexico", "South Africa")
            ]
            self.assertEqual(
                [version["snapshotDate"] for version in shared],
                ["2026-06-15", "2026-06-11"],
            )
            self.assertEqual(
                [version["snapshotDate"] for version in completed],
                ["2026-06-11"],
            )

    def test_prediction_versions_reject_same_snapshot_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            snapshot = Path(temporary_directory) / "2026-06-11"
            snapshot.mkdir()
            fixtures = {
                fixture_key("2026-06-18", "Mexico", "South Korea"): {}
            }
            for folder in ("first", "second"):
                destination = snapshot / folder
                destination.mkdir()
                (destination / "match_data.json").write_text(
                    json.dumps(
                        {
                            "date": "2026-06-18",
                            "home_team": "Mexico",
                            "away_team": "South Korea",
                        }
                    )
                )
                (destination / "predictions.json").write_text("{}")

            with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
                collect_prediction_versions([snapshot], fixtures)

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

    def test_actual_result_follows_prediction_home_away_orientation(self):
        fixture = {
            "team1": "South Africa",
            "team2": "Mexico",
            "score": {"ft": [1, 2]},
            "goals1": [{"name": "Away scorer", "minute": "45+2"}],
            "goals2": [{"name": "Home Scórer", "minute": 10}],
        }
        squad_players = {
            "Mexico": [{"name": "Home scorer"}],
            "South Africa": [{"name": "Away scorer"}],
        }

        result = extract_actual_result(
            fixture, "Mexico", "South Africa", squad_players
        )

        self.assertEqual(result["homeScore"], 2)
        self.assertEqual(result["awayScore"], 1)
        self.assertEqual(result["homeGoals"][0]["name"], "Home scorer")
        self.assertEqual(result["awayGoals"][0]["name"], "Away scorer")
        self.assertEqual(result["homeGoals"][0]["minute"], "10")
        self.assertEqual(result["awayGoals"][0]["minute"], "45+2")

    def test_player_names_match_case_and_accents_but_require_uniqueness(self):
        players = [{"name": "Ladislav Krejčí"}]
        self.assertEqual(normalize_player_name("Ladislav KrejcÍ"), "ladislav krejci")
        self.assertEqual(
            canonical_scorer_name("Ladislav Krejcí", players),
            "Ladislav Krejčí",
        )
        self.assertEqual(
            canonical_scorer_name("Unknown Player", players), "Unknown Player"
        )
        self.assertEqual(
            canonical_scorer_name(
                "ALEX SMITH", [{"name": "Álex Smith"}, {"name": "Alex Smith"}]
            ),
            "ALEX SMITH",
        )

    @patch("scripts.build_frontend_data.urlopen")
    def test_schedule_fetch_bypasses_caches(self, mocked_urlopen):
        response = mocked_urlopen.return_value.__enter__.return_value
        response.read.return_value = b'{"matches": []}'

        self.assertEqual(fetch_schedule(), {"matches": []})

        request = mocked_urlopen.call_args.args[0]
        self.assertIsInstance(request, Request)
        self.assertIn("refresh=", request.full_url)
        self.assertEqual(request.get_header("Cache-control"), "no-cache")
        self.assertEqual(request.get_header("Pragma"), "no-cache")

    @patch("scripts.build_frontend_data.urlopen")
    def test_squad_fetch_bypasses_caches(self, mocked_urlopen):
        response = mocked_urlopen.return_value.__enter__.return_value
        response.read.return_value = b'[{"name": "Mexico"}]'

        self.assertEqual(fetch_squads(), [{"name": "Mexico"}])

        request = mocked_urlopen.call_args.args[0]
        self.assertIsInstance(request, Request)
        self.assertIn("worldcup.squads.json", request.full_url)
        self.assertIn("refresh=", request.full_url)
        self.assertEqual(request.get_header("Cache-control"), "no-cache")

    def test_latest_snapshot_source_url_uses_selected_iso_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            older = root / "2026-06-10"
            latest = root / "2026-06-11"
            older.mkdir()
            latest.mkdir()

            self.assertEqual(discover_latest_snapshot(root).name, "2026-06-11")
            url = repository_tree_url(
                Path("outputs/predictions/2026-06-11"),
                "https://github.com/state-space-models/cuthberto-carlos",
            )
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

if __name__ == "__main__":
    unittest.main()
