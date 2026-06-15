"""Tests for generated frontend dataset validation."""

import copy
import json
import os
import unittest
from unittest.mock import patch

from scripts.build_frontend_data import (
    DEFAULT_REPOSITORY,
    ROOT,
    SCHEDULE_DATA_URL,
    SCHEDULE_REF,
    SCHEDULE_SOURCE_URL,
    SQUADS_DATA_URL,
    SQUADS_SOURCE_URL,
    validate_dataset,
)


class FrontendDataValidationTests(unittest.TestCase):
    """Validate the checked-in dataset and important failure modes."""

    def setUp(self):
        dataset_path = ROOT / "frontend" / "src" / "data" / "tournament.json"
        self.dataset = json.loads(dataset_path.read_text())
        self.repository_url = f"https://github.com/{DEFAULT_REPOSITORY}"

    def validate(self, dataset):
        with patch.dict(os.environ, {"GITHUB_REPOSITORY": DEFAULT_REPOSITORY}):
            validate_dataset(dataset)

    def test_checked_in_dataset_is_valid(self):
        self.validate(self.dataset)

    def test_dataset_has_complete_unique_fixture_sets(self):
        self.assertEqual(len(self.dataset["groupMatches"]), 72)
        self.assertEqual(len(self.dataset["knockoutMatches"]), 32)
        self.assertEqual(len(self.dataset["groups"]), 12)
        match_ids = [
            match["id"]
            for match in self.dataset["groupMatches"]
            + self.dataset["knockoutMatches"]
        ]
        self.assertEqual(len(match_ids), len(set(match_ids)))
        for match in self.dataset["groupMatches"]:
            probabilities = match["prediction"]["probabilities"]
            self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=7)
            self.assertTrue(all(0 <= value <= 1 for value in probabilities.values()))

    def test_dataset_records_openfootball_master_provenance(self):
        schedule = self.dataset["sources"]["schedule"]
        self.assertEqual(schedule["url"], SCHEDULE_SOURCE_URL)
        self.assertEqual(schedule["dataUrl"], SCHEDULE_DATA_URL)
        self.assertEqual(schedule["ref"], SCHEDULE_REF)
        squads = self.dataset["sources"]["squads"]
        self.assertEqual(squads["url"], SQUADS_SOURCE_URL)
        self.assertEqual(squads["dataUrl"], SQUADS_DATA_URL)
        self.assertEqual(squads["ref"], SCHEDULE_REF)

    def test_dataset_has_complete_valid_squads(self):
        self.assertEqual(self.dataset["schemaVersion"], 5)
        self.assertEqual(len(self.dataset["teams"]), 48)
        for name, team in self.dataset["teams"].items():
            self.assertEqual(team["name"], name)
            self.assertTrue(team["players"])
            numbers = [player["number"] for player in team["players"]]
            self.assertEqual(numbers, sorted(numbers))
            self.assertEqual(len(numbers), len(set(numbers)))

    def test_goal_minutes_are_strings(self):
        goals = [
            goal
            for match in self.dataset["groupMatches"]
            for goal in (
                match.get("actualResult", {}).get("homeGoals", [])
                + match.get("actualResult", {}).get("awayGoals", [])
            )
        ]
        self.assertTrue(goals)
        self.assertTrue(all(isinstance(goal["minute"], str) for goal in goals))

    def test_project_urls_use_canonical_repository(self):
        self.assertEqual(self.dataset["repositoryUrl"], self.repository_url)
        self.assertTrue(self.dataset["snapshotUrl"].startswith(self.repository_url))
        self.assertTrue(
            all(
                match["sourceUrl"].startswith(self.repository_url)
                for match in self.dataset["groupMatches"]
            )
        )
        self.assertTrue(
            all(
                historical["sourceUrl"].startswith(self.repository_url)
                for match in self.dataset["groupMatches"]
                for historical in match["predictionHistory"]
            )
        )

    def test_duplicate_match_id_is_rejected(self):
        changed = copy.deepcopy(self.dataset)
        changed["knockoutMatches"][0]["id"] = changed["groupMatches"][0]["id"]
        with self.assertRaisesRegex(ValueError, "Match IDs must be unique"):
            self.validate(changed)

    def test_invalid_probability_total_is_rejected(self):
        changed = copy.deepcopy(self.dataset)
        changed["groupMatches"][0]["prediction"]["probabilities"] = {
            "homeWin": 0.5,
            "draw": 0.5,
            "awayWin": 0.5,
        }
        with self.assertRaisesRegex(ValueError, "probabilities do not sum to one"):
            self.validate(changed)

    def test_noncanonical_match_source_is_rejected(self):
        changed = copy.deepcopy(self.dataset)
        changed["groupMatches"][0]["sourceUrl"] = (
            "https://github.com/example/fork/tree/main/outputs/predictions/match"
        )
        with self.assertRaisesRegex(ValueError, "canonical repository"):
            self.validate(changed)

    def test_prediction_history_must_be_unique_and_older(self):
        changed = copy.deepcopy(self.dataset)
        match = next(
            match
            for match in changed["groupMatches"]
            if match["predictionHistory"]
        )
        match["predictionHistory"].append(copy.deepcopy(match["predictionHistory"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate dates"):
            self.validate(changed)

        changed = copy.deepcopy(self.dataset)
        match = next(
            match
            for match in changed["groupMatches"]
            if match["predictionHistory"]
        )
        match["predictionHistory"][0]["predictionDate"] = match["predictionDate"]
        match["predictionHistory"][0]["sourceUrl"] = match["sourceUrl"]
        with self.assertRaisesRegex(ValueError, "not older than latest"):
            self.validate(changed)

    def test_prediction_history_requires_prediction_data(self):
        changed = copy.deepcopy(self.dataset)
        match = next(
            match
            for match in changed["groupMatches"]
            if match["predictionHistory"]
        )
        del match["predictionHistory"][0]["prediction"]
        with self.assertRaisesRegex(ValueError, "missing prediction data"):
            self.validate(changed)


if __name__ == "__main__":
    unittest.main()
