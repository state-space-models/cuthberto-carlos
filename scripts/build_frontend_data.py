"""Compile prediction outputs into a compact dataset for the static frontend."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_ROOT = ROOT / "outputs" / "predictions"
DEFAULT_OUTPUT = ROOT / "frontend" / "src" / "data" / "tournament.json"
DEFAULT_REPOSITORY = "state-space-models/cuthberto-carlos"
SCHEDULE_REF = "master"
SCHEDULE_SOURCE_URL = (
    "https://github.com/openfootball/worldcup.json/blob/master/2026/worldcup.json"
)
SCHEDULE_DATA_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/"
    "master/2026/worldcup.json"
)


def get_repository_slug() -> str:
    """Return the canonical owner/name, preferring GitHub Actions metadata."""
    repository = os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY).strip()
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repository):
        raise ValueError(f"Invalid GITHUB_REPOSITORY value: {repository!r}")
    return repository


def get_repository_url() -> str:
    """Return the canonical GitHub repository URL."""
    return f"https://github.com/{get_repository_slug()}"

TEAM_ALIASES = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "USA": "United States",
}

FLAG_CODES = {
    "Algeria": "dz",
    "Argentina": "ar",
    "Australia": "au",
    "Austria": "at",
    "Belgium": "be",
    "Bosnia and Herzegovina": "ba",
    "Brazil": "br",
    "Canada": "ca",
    "Cape Verde": "cv",
    "Colombia": "co",
    "Croatia": "hr",
    "Curaçao": "cw",
    "Czech Republic": "cz",
    "DR Congo": "cd",
    "Ecuador": "ec",
    "Egypt": "eg",
    "England": "gb-eng",
    "France": "fr",
    "Germany": "de",
    "Ghana": "gh",
    "Haiti": "ht",
    "Iran": "ir",
    "Iraq": "iq",
    "Ivory Coast": "ci",
    "Japan": "jp",
    "Jordan": "jo",
    "Mexico": "mx",
    "Morocco": "ma",
    "Netherlands": "nl",
    "New Zealand": "nz",
    "Norway": "no",
    "Panama": "pa",
    "Paraguay": "py",
    "Portugal": "pt",
    "Qatar": "qa",
    "Saudi Arabia": "sa",
    "Scotland": "gb-sct",
    "Senegal": "sn",
    "South Africa": "za",
    "South Korea": "kr",
    "Spain": "es",
    "Sweden": "se",
    "Switzerland": "ch",
    "Tunisia": "tn",
    "Turkey": "tr",
    "United States": "us",
    "Uruguay": "uy",
    "Uzbekistan": "uz",
}

KNOCKOUT_NUMBERS = {
    "Match for third place": 103,
    "Final": 104,
}


def canonical_team(name: str) -> str:
    """Return the project spelling for a schedule team name."""
    return TEAM_ALIASES.get(name, name)


def slugify(value: str) -> str:
    """Create a stable ASCII-ish identifier for generated records."""
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def repository_tree_url(path: Path, repository_url: str | None = None) -> str:
    """Return a GitHub tree URL for a repository-relative source directory."""
    base_url = repository_url or get_repository_url()
    return f"{base_url}/tree/main/{path.as_posix()}"


def discover_latest_snapshot(predictions_root: Path = PREDICTIONS_ROOT) -> Path:
    """Find the latest prediction directory whose name is an ISO date."""
    candidates: list[tuple[datetime, Path]] = []
    if not predictions_root.exists():
        raise FileNotFoundError(f"Prediction root does not exist: {predictions_root}")

    for path in predictions_root.iterdir():
        if not path.is_dir():
            continue
        try:
            parsed = datetime.strptime(path.name, "%Y-%m-%d")
        except ValueError:
            continue
        candidates.append((parsed, path))

    if not candidates:
        raise ValueError(f"No dated prediction snapshots found in {predictions_root}")
    return max(candidates, key=lambda item: item[0])[1]


def fetch_schedule(url: str = SCHEDULE_DATA_URL) -> dict[str, Any]:
    """Download the current CC0 World Cup schedule from openfootball master."""
    request = Request(url, headers={"User-Agent": "cuthberto-carlos-data-builder"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - pinned HTTPS URL
        return json.load(response)


def fixture_key(date: str, team1: str, team2: str) -> tuple[str, tuple[str, str]]:
    """Build an orientation-independent fixture key."""
    teams = tuple(sorted((canonical_team(team1), canonical_team(team2))))
    return date, teams


def extract_actual_result(fixture: dict[str, Any]) -> dict[str, Any] | None:
    """Extract actual match result from schedule fixture if available."""
    score = fixture.get("score")
    if not score or "ft" not in score:
        return None

    home_score, away_score = score["ft"]
    goals1 = fixture.get("goals1", [])
    goals2 = fixture.get("goals2", [])

    # Determine which team is home/away based on fixture orientation
    # The fixture has team1 and team2, but our canonical home/away may differ
    return {
        "homeScore": home_score,
        "awayScore": away_score,
        "homeGoals": [
            {"name": g["name"], "minute": g["minute"], "penalty": g.get("penalty")}
            for g in goals1
        ],
        "awayGoals": [
            {"name": g["name"], "minute": g["minute"], "penalty": g.get("penalty")}
            for g in goals2
        ],
    }


def parse_kickoff_utc(date: str, time_value: str) -> str:
    """Convert an openfootball local timestamp with UTC offset to UTC ISO format."""
    match = re.fullmatch(r"(\d{2}):(\d{2}) UTC([+-]\d{1,2})", time_value)
    if not match:
        raise ValueError(f"Unsupported schedule time: {time_value!r}")
    hour, minute, offset = (int(part) for part in match.groups())
    local_tz = timezone(timedelta(hours=offset))
    local_dt = datetime.strptime(date, "%Y-%m-%d").replace(
        hour=hour,
        minute=minute,
        tzinfo=local_tz,
    )
    return local_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_grid(grid: list[list[float]]) -> list[list[float]]:
    """Normalize a rectangular score probability grid."""
    if not grid or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("Score grid must be non-empty and rectangular")
    total = sum(sum(float(value) for value in row) for row in grid)
    if total <= 0:
        raise ValueError("Score grid probability mass must be positive")
    return [[float(value) / total for value in row] for row in grid]


def prediction_metrics(grid: list[list[float]]) -> dict[str, Any]:
    """Derive score and expected-goal metrics from a score probability grid."""
    normalized = normalize_grid(grid)
    best_home, best_away, best_probability = 0, 0, -1.0
    expected_home = 0.0
    expected_away = 0.0

    for home_goals, row in enumerate(normalized):
        for away_goals, probability in enumerate(row):
            expected_home += home_goals * probability
            expected_away += away_goals * probability
            if probability > best_probability:
                best_home, best_away, best_probability = (
                    home_goals,
                    away_goals,
                    probability,
                )

    return {
        "scoreGrid": [[round(value, 8) for value in row] for row in normalized],
        "mostLikelyScore": [best_home, best_away],
        "mostLikelyScoreProbability": round(best_probability, 8),
        "expectedGoals": {
            "home": round(expected_home, 4),
            "away": round(expected_away, 4),
        },
    }


def source_commit(root: Path = ROOT) -> str:
    """Return the source commit, preferring GitHub's exact workflow SHA."""
    workflow_sha = os.environ.get("GITHUB_SHA")
    if workflow_sha:
        return workflow_sha[:12]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _team_record(
    name: str, team_colors: dict[str, list[str]]
) -> dict[str, str | list[str]]:
    if name not in FLAG_CODES:
        raise ValueError(f"Missing flag mapping for {name}")
    colors = team_colors.get(name, ["#777777", "#dddddd"])
    return {"name": name, "flagCode": FLAG_CODES[name], "colors": colors}


def _skill_record(
    means: list[list[float]], covariance: list[list[list[float]]], index: int
) -> dict[str, dict[str, float]]:
    return {
        "attack": {
            "mean": round(float(means[index][0]), 6),
            "sd": round(math.sqrt(float(covariance[index][0][0])), 6),
        },
        "defence": {
            "mean": round(float(means[index][1]), 6),
            "sd": round(math.sqrt(float(covariance[index][1][1])), 6),
        },
    }


def _group_projection(
    group_name: str,
    team_names: Iterable[str],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = {
        name: {
            "team": name,
            "expectedPoints": 0.0,
            "expectedGoalsFor": 0.0,
            "expectedGoalsAgainst": 0.0,
            "expectedGoalDifference": 0.0,
        }
        for name in team_names
    }

    for match in matches:
        home = rows[match["homeTeam"]]
        away = rows[match["awayTeam"]]
        probabilities = match["prediction"]["probabilities"]
        expected_goals = match["prediction"]["expectedGoals"]

        home["expectedPoints"] += 3 * probabilities["homeWin"] + probabilities["draw"]
        away["expectedPoints"] += 3 * probabilities["awayWin"] + probabilities["draw"]
        home["expectedGoalsFor"] += expected_goals["home"]
        home["expectedGoalsAgainst"] += expected_goals["away"]
        away["expectedGoalsFor"] += expected_goals["away"]
        away["expectedGoalsAgainst"] += expected_goals["home"]

    for row in rows.values():
        row["expectedGoalDifference"] = (
            row["expectedGoalsFor"] - row["expectedGoalsAgainst"]
        )
        for key in (
            "expectedPoints",
            "expectedGoalsFor",
            "expectedGoalsAgainst",
            "expectedGoalDifference",
        ):
            row[key] = round(row[key], 3)

    projection = sorted(
        rows.values(),
        key=lambda row: (
            -row["expectedPoints"],
            -row["expectedGoalDifference"],
            -row["expectedGoalsFor"],
            row["team"],
        ),
    )
    for rank, row in enumerate(projection, start=1):
        row["rank"] = rank

    group_id = group_name.removeprefix("Group ")
    return {
        "id": group_id,
        "name": group_name,
        "projection": projection,
        "matchIds": [match["id"] for match in matches],
    }


def compile_dataset(
    root: Path = ROOT,
    schedule_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile the newest prediction snapshot and schedule into frontend data."""
    snapshot = discover_latest_snapshot(root / "outputs" / "predictions")
    repository_url = get_repository_url()
    schedule = schedule_data or fetch_schedule()
    schedule_matches = schedule.get("matches", [])
    group_schedule = [match for match in schedule_matches if match.get("group")]
    knockout_schedule = [match for match in schedule_matches if not match.get("group")]

    if len(group_schedule) != 72:
        raise ValueError(f"Expected 72 group fixtures, found {len(group_schedule)}")
    if len(knockout_schedule) != 32:
        raise ValueError(f"Expected 32 knockout fixtures, found {len(knockout_schedule)}")

    fixture_index: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
    for fixture in group_schedule:
        key = fixture_key(fixture["date"], fixture["team1"], fixture["team2"])
        if key in fixture_index:
            raise ValueError(f"Duplicate group fixture in schedule: {key}")
        fixture_index[key] = fixture

    team_colors = json.loads((root / "assets" / "team_colors.json").read_text())
    match_records: list[dict[str, Any]] = []
    matched_keys: set[tuple[str, tuple[str, str]]] = set()

    prediction_files = sorted(snapshot.glob("*/predictions.json"))
    if len(prediction_files) != 72:
        raise ValueError(
            f"Expected 72 prediction files in {snapshot}, found {len(prediction_files)}"
        )

    for prediction_path in prediction_files:
        match_path = prediction_path.with_name("match_data.json")
        if not match_path.exists():
            raise ValueError(f"Missing match_data.json beside {prediction_path}")
        match_data = json.loads(match_path.read_text())
        predictions = json.loads(prediction_path.read_text())
        key = fixture_key(
            match_data["date"], match_data["home_team"], match_data["away_team"]
        )
        fixture = fixture_index.get(key)
        if fixture is None:
            raise ValueError(f"Prediction does not match a group fixture: {key}")
        if key in matched_keys:
            raise ValueError(f"Duplicate prediction for group fixture: {key}")
        matched_keys.add(key)

        raw_results = [float(value) for value in predictions["probs_results"]]
        result_total = sum(raw_results)
        if result_total <= 0:
            raise ValueError(f"Invalid result probabilities in {prediction_path}")
        draw, home_win, away_win = (value / result_total for value in raw_results)
        metrics = prediction_metrics(predictions["probs_grid"])
        home_team = canonical_team(match_data["home_team"])
        away_team = canonical_team(match_data["away_team"])
        group_id = fixture["group"].removeprefix("Group ")

        record = {
            "id": slugify(
                f"{match_data['date']}-{home_team}-{away_team}"
            ),
            "date": match_data["date"],
            "kickoffUtc": parse_kickoff_utc(fixture["date"], fixture["time"]),
            "group": group_id,
            "venue": fixture["ground"],
            "homeTeam": home_team,
            "awayTeam": away_team,
            "sourceUrl": (
                repository_tree_url(
                    prediction_path.parent.relative_to(root), repository_url
                )
            ),
            "prediction": {
                "probabilities": {
                    "homeWin": round(home_win, 8),
                    "draw": round(draw, 8),
                    "awayWin": round(away_win, 8),
                },
                **metrics,
                "skills": {
                    "home": _skill_record(
                        predictions["skills_mean"], predictions["skills_cov"], 0
                    ),
                    "away": _skill_record(
                        predictions["skills_mean"], predictions["skills_cov"], 1
                    ),
                },
            },
        }

        # Add actual result if available from schedule
        actual_result = extract_actual_result(fixture)
        if actual_result:
            record["actualResult"] = actual_result

        match_records.append(record)

    if len(matched_keys) != len(fixture_index):
        missing = sorted(set(fixture_index) - matched_keys)
        raise ValueError(f"Missing predictions for schedule fixtures: {missing}")

    match_records.sort(key=lambda match: (match["kickoffUtc"], match["id"]))
    team_names = sorted(
        {match["homeTeam"] for match in match_records}
        | {match["awayTeam"] for match in match_records}
    )
    teams = {name: _team_record(name, team_colors) for name in team_names}

    groups = []
    for group_id in "ABCDEFGHIJKL":
        group_matches = [match for match in match_records if match["group"] == group_id]
        if len(group_matches) != 6:
            raise ValueError(f"Expected six matches in Group {group_id}")
        group_teams = sorted(
            {match["homeTeam"] for match in group_matches}
            | {match["awayTeam"] for match in group_matches}
        )
        if len(group_teams) != 4:
            raise ValueError(f"Expected four teams in Group {group_id}")
        groups.append(
            _group_projection(f"Group {group_id}", group_teams, group_matches)
        )

    knockout_matches = []
    for fixture in knockout_schedule:
        match_number = fixture.get("num") or KNOCKOUT_NUMBERS.get(fixture["round"])
        if not match_number:
            raise ValueError(f"Missing match number for knockout fixture: {fixture}")
        knockout_matches.append(
            {
                "id": f"match-{match_number}",
                "matchNumber": match_number,
                "round": fixture["round"],
                "date": fixture["date"],
                "kickoffUtc": parse_kickoff_utc(fixture["date"], fixture["time"]),
                "venue": fixture["ground"],
                "team1Slot": fixture["team1"],
                "team2Slot": fixture["team2"],
            }
        )
    knockout_matches.sort(key=lambda match: match["matchNumber"])

    return {
        "schemaVersion": 2,
        "repositoryUrl": repository_url,
        "snapshotDate": snapshot.name,
        "snapshotPath": str(snapshot.relative_to(root)),
        "snapshotUrl": repository_tree_url(snapshot.relative_to(root), repository_url),
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sourceCommit": source_commit(root),
        "model": {
            "name": "Cuthberto Carlos bivariate Poisson model",
            "resultProbabilityOrder": ["draw", "homeWin", "awayWin"],
            "scoreGridMaxGoals": 8,
        },
        "sources": {
            "schedule": {
                "name": "openfootball/worldcup.json",
                "url": SCHEDULE_SOURCE_URL,
                "dataUrl": SCHEDULE_DATA_URL,
                "ref": SCHEDULE_REF,
                "license": "CC0-1.0",
            },
            "historicalResults": {
                "name": "martj42/international_results",
                "url": "https://github.com/martj42/international_results",
                "license": "CC0-1.0",
            },
        },
        "teams": teams,
        "groupMatches": match_records,
        "groups": groups,
        "knockoutMatches": knockout_matches,
    }


def validate_dataset(dataset: dict[str, Any]) -> None:
    """Validate generated frontend data and its source provenance."""
    required_keys = {
        "schemaVersion",
        "repositoryUrl",
        "snapshotDate",
        "snapshotPath",
        "snapshotUrl",
        "generatedAt",
        "sourceCommit",
        "model",
        "sources",
        "teams",
        "groupMatches",
        "groups",
        "knockoutMatches",
    }
    missing = required_keys - set(dataset)
    if missing:
        raise ValueError(f"Dataset is missing required keys: {sorted(missing)}")

    if dataset["schemaVersion"] != 2:
        raise ValueError(f"Unsupported schema version: {dataset['schemaVersion']!r}")
    try:
        datetime.strptime(dataset["snapshotDate"], "%Y-%m-%d")
        datetime.fromisoformat(dataset["generatedAt"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Dataset date metadata is invalid") from error
    if not re.fullmatch(r"[0-9a-f]{12}|unknown", dataset["sourceCommit"]):
        raise ValueError("Dataset source commit is invalid")

    repository_url = dataset["repositoryUrl"]
    if repository_url != get_repository_url():
        raise ValueError(f"Unexpected repository URL: {repository_url}")
    if not dataset["snapshotUrl"].startswith(f"{repository_url}/tree/main/"):
        raise ValueError("Snapshot URL does not use the canonical repository URL")

    schedule = dataset["sources"].get("schedule", {})
    expected_schedule = {
        "url": SCHEDULE_SOURCE_URL,
        "dataUrl": SCHEDULE_DATA_URL,
        "ref": SCHEDULE_REF,
    }
    for key, expected in expected_schedule.items():
        if schedule.get(key) != expected:
            raise ValueError(f"Unexpected schedule {key}: {schedule.get(key)!r}")

    group_matches = dataset["groupMatches"]
    knockout_matches = dataset["knockoutMatches"]
    if len(group_matches) != 72:
        raise ValueError(f"Expected 72 group matches, found {len(group_matches)}")
    if len(knockout_matches) != 32:
        raise ValueError(f"Expected 32 knockout matches, found {len(knockout_matches)}")
    if len(dataset["groups"]) != 12:
        raise ValueError(f"Expected 12 groups, found {len(dataset['groups'])}")
    if not dataset["teams"]:
        raise ValueError("Dataset must include team metadata")

    all_ids = [match["id"] for match in group_matches + knockout_matches]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Match IDs must be unique")

    source_prefix = f"{repository_url}/tree/main/{dataset['snapshotPath']}/"
    for match in group_matches:
        if not match["sourceUrl"].startswith(source_prefix):
            raise ValueError(
                f"Match {match['id']} source URL does not use the canonical repository"
            )
        probabilities = match["prediction"]["probabilities"]
        values = [
            probabilities["homeWin"],
            probabilities["draw"],
            probabilities["awayWin"],
        ]
        if any(value < 0 or value > 1 for value in values):
            raise ValueError(f"Match {match['id']} has an invalid probability")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-7):
            raise ValueError(f"Match {match['id']} probabilities do not sum to one")


def main() -> None:
    """Compile and write the frontend dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = compile_dataset()
    validate_dataset(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(dataset['groupMatches'])} group predictions and "
        f"{len(dataset['knockoutMatches'])} knockout fixtures to {args.output}"
    )


if __name__ == "__main__":
    main()
