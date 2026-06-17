# ruff: noqa: D101,D103,E402
"""Backtest result-probability predictions as a scaled Kelly betting strategy.

The prediction files contain ``probs_results`` in the order:
draw, home win, away win. This script ignores scoreline probabilities.

Results are downloaded from the same international-results source used by the
model. Odds are scraped from FotMob's displayed 1X2 match odds.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotnine as pn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cuthberto_carlos.data import download_data


DEFAULT_PREDICTION_ROOT = Path("outputs/predictions/2026-06-11")
DEFAULT_OUTPUT_ROOT = Path("outputs/kelly_backtest/2026-06-11")
DEFAULT_FOTMOB_CCODE3 = "GBR"
FOTMOB_KNOWN_MATCH_IDS = (
    4667751,
    4667752,
    4667757,
    4667758,
    4667764,
    4667765,
    4667771,
    4667772,
    4667777,
    4667778,
    4667783,
    4667784,
    4667790,
    4667791,
    4667798,
    4667799,
    4667804,
    4667805,
    4667812,
    4667813,
)
RESULT_ORDER = ("draw", "home", "away")


@dataclass(frozen=True)
class Prediction:
    date: str
    home_team: str
    away_team: str
    prob_draw: float
    prob_home: float
    prob_away: float
    prediction_path: Path


@dataclass(frozen=True)
class MatchResult:
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    result: str
    source: str


@dataclass(frozen=True)
class Odds:
    date: str
    home_team: str
    away_team: str
    odds_home: float
    odds_draw: float
    odds_away: float
    source: str


@dataclass(frozen=True)
class BacktestMatch:
    prediction: Prediction
    result: MatchResult
    odds: Odds | None


def canonical_team(name: str) -> str:
    """Return a stable team key across common data-source naming variants."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\bd\.?\s*r\.?\b", "dr", text)
    text = re.sub(r"\bu\.?\s*s\.?\s*a\.?\b", "united states", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = re.sub(r"\s+", " ", text)
    aliases = {
        "bosnia and herzegovina": "bosnia and herzegovina",
        "bosnia herzegovina": "bosnia and herzegovina",
        "bosnia": "bosnia and herzegovina",
        "cabo verde": "cape verde",
        "cote d ivoire": "ivory coast",
        "curacao": "curacao",
        "czechia": "czech republic",
        "d r congo": "dr congo",
        "democratic republic of congo": "dr congo",
        "dr congo": "dr congo",
        "ivory coast": "ivory coast",
        "korea republic": "south korea",
        "holland": "netherlands",
        "south korea": "south korea",
        "turkiye": "turkey",
        "usa": "united states",
        "united states": "united states",
        "united states of america": "united states",
    }
    return aliases.get(text, text)


def match_key(date: str, home_team: str, away_team: str) -> tuple[str, str, str]:
    return (date, canonical_team(home_team), canonical_team(away_team))


def date_str(value: Any) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_scales(raw: str) -> list[float]:
    """Parse either comma-separated scales or start:stop:step."""
    raw = raw.strip()
    if ":" in raw:
        start_s, stop_s, step_s = raw.split(":")
        start, stop, step = float(start_s), float(stop_s), float(step_s)
        if step <= 0:
            raise ValueError("Scale step must be positive.")
        values: list[float] = []
        current = start
        while current <= stop + step / 2:
            values.append(round(current, 10))
            current += step
        return values
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def load_predictions(prediction_root: Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    for predictions_path in sorted(prediction_root.glob("*/predictions.json")):
        match_data_path = predictions_path.parent / "match_data.json"
        if not match_data_path.exists():
            continue
        with match_data_path.open() as f:
            match_data = json.load(f)
        with predictions_path.open() as f:
            prediction_data = json.load(f)

        probs = [float(value) for value in prediction_data["probs_results"]]
        if len(probs) != 3:
            raise ValueError(f"Expected 3 probs_results values in {predictions_path}")
        predictions.append(
            Prediction(
                date=date_str(match_data["date"]),
                home_team=str(match_data["home_team"]),
                away_team=str(match_data["away_team"]),
                prob_draw=probs[0],
                prob_home=probs[1],
                prob_away=probs[2],
                prediction_path=predictions_path,
            )
        )
    return predictions


def load_results() -> list[MatchResult]:
    data, _, _, _ = download_data(future_matches=True)
    completed = data[(data["home_score"] >= 0) & (data["away_score"] >= 0)].copy()
    results: list[MatchResult] = []
    for row in completed.itertuples(index=False):
        home_score = int(row.home_score)
        away_score = int(row.away_score)
        if home_score == away_score:
            result = "draw"
        elif home_score > away_score:
            result = "home"
        else:
            result = "away"
        results.append(
            MatchResult(
                date=date_str(row.date),
                home_team=str(row.home_team),
                away_team=str(row.away_team),
                home_score=home_score,
                away_score=away_score,
                result=result,
                source="martj42/international_results",
            )
        )
    return results


def build_result_index(
    results: Iterable[MatchResult],
) -> dict[tuple[str, str, str], MatchResult]:
    index: dict[tuple[str, str, str], MatchResult] = {}
    for result in results:
        index[match_key(result.date, result.home_team, result.away_team)] = result
    return index


def result_for_prediction(
    prediction: Prediction, result_index: dict[tuple[str, str, str], MatchResult]
) -> MatchResult | None:
    exact = result_index.get(
        match_key(prediction.date, prediction.home_team, prediction.away_team)
    )
    if exact is not None:
        return exact

    reversed_result = result_index.get(
        match_key(prediction.date, prediction.away_team, prediction.home_team)
    )
    if reversed_result is None:
        return None

    if reversed_result.result == "home":
        result = "away"
    elif reversed_result.result == "away":
        result = "home"
    else:
        result = "draw"
    return MatchResult(
        date=reversed_result.date,
        home_team=prediction.home_team,
        away_team=prediction.away_team,
        home_score=reversed_result.away_score,
        away_score=reversed_result.home_score,
        result=result,
        source=f"{reversed_result.source}; reversed fixture orientation",
    )


def cached_json_request(url: str, cache_path: Path, timeout: float = 5.0) -> Any:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "cuthberto-carlos/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def next_data_from_html(page: str) -> dict[str, Any]:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        page,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("missing __NEXT_DATA__ script")
    payload = json.loads(html.unescape(match.group(1)))
    if not isinstance(payload, dict):
        raise ValueError("unexpected __NEXT_DATA__ payload")
    return payload


def sports_event_from_html(page: str) -> dict[str, Any]:
    def candidates(value: Any) -> Iterable[dict[str, Any]]:
        if isinstance(value, dict):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item

    for match in re.finditer(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
        page,
        flags=re.DOTALL,
    ):
        try:
            payload = json.loads(html.unescape(match.group(1)))
        except json.JSONDecodeError:
            continue
        for candidate in candidates(payload):
            if candidate.get("@type") == "SportsEvent":
                return candidate
    return {}


def fotmob_match_page(match_id: int, cache_dir: Path) -> tuple[str, str]:
    url = f"https://www.fotmob.com/match/{match_id}"
    cache_path = cache_dir / "fotmob" / "matches" / f"{match_id}.html"
    if cache_path.exists():
        return cache_path.read_text(), url
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=20.0) as response:
        page = response.read().decode()
        final_url = response.geturl()
    cache_path.write_text(page)
    return page, final_url


def fotmob_match_metadata(match_id: int, cache_dir: Path) -> dict[str, str]:
    page, final_url = fotmob_match_page(match_id, cache_dir)
    try:
        payload = next_data_from_html(page)
        event = (
            payload.get("props", {})
            .get("pageProps", {})
            .get("content", {})
            .get("seo", {})
            .get("eventJSONLD", {})
        )
        if not isinstance(event, dict):
            event = {}
    except ValueError:
        event = {}
    if not event:
        event = sports_event_from_html(page)

    home_team = event.get("homeTeam", {}).get("name")
    away_team = event.get("awayTeam", {}).get("name")
    start_date = event.get("startDate")
    if not home_team or not away_team or not start_date:
        raise ValueError(f"Could not parse FotMob metadata for match {match_id}")
    return {
        "date": date_str(start_date),
        "home_team": str(home_team),
        "away_team": str(away_team),
        "url": final_url,
    }


def fotmob_odds_payload(
    match_id: int, cache_dir: Path, ccode3: str
) -> dict[str, Any] | None:
    params = urllib.parse.urlencode(
        {"matchId": match_id, "ccode3": ccode3, "oddsFormat": 2}
    )
    url = f"https://www.fotmob.com/api/data/matchOdds?{params}"
    cache_path = cache_dir / "fotmob" / "odds" / f"{match_id}_{ccode3}.json"
    try:
        payload = cached_json_request(url, cache_path, timeout=20.0)
    except Exception as exc:
        print(f"Skipping FotMob odds for match {match_id}: {exc}")
        return None
    return payload if isinstance(payload, dict) else None


def odds_from_fotmob(
    predictions: Iterable[Prediction], cache_dir: Path, ccode3: str
) -> list[Odds]:
    prediction_rows = list(predictions)

    def matching_prediction(metadata: dict[str, str]) -> Prediction | None:
        metadata_home = canonical_team(metadata["home_team"])
        metadata_away = canonical_team(metadata["away_team"])
        metadata_date = pd.to_datetime(metadata["date"]).date()
        for prediction in prediction_rows:
            prediction_home = canonical_team(prediction.home_team)
            prediction_away = canonical_team(prediction.away_team)
            same_teams = (
                prediction_home == metadata_home and prediction_away == metadata_away
            ) or (prediction_home == metadata_away and prediction_away == metadata_home)
            if not same_teams:
                continue
            prediction_date = pd.to_datetime(prediction.date).date()
            if abs((prediction_date - metadata_date).days) <= 1:
                return prediction
        return None

    odds_rows: list[Odds] = []
    for match_id in FOTMOB_KNOWN_MATCH_IDS:
        try:
            metadata = fotmob_match_metadata(match_id, cache_dir)
        except Exception as exc:
            print(f"Skipping FotMob match {match_id}: {exc}")
            continue

        prediction = matching_prediction(metadata)
        if prediction is None:
            continue

        payload = fotmob_odds_payload(match_id, cache_dir, ccode3)
        if payload is None:
            continue
        selections = (
            payload.get("odds", {}).get("resolvedOddsMarket", {}).get("selections", [])
        )
        if not isinstance(selections, list):
            continue
        prices: dict[str, float] = {}
        for selection in selections:
            if not isinstance(selection, dict):
                continue
            name = str(selection.get("name", "")).upper()
            odds_decimal = selection.get("oddsDecimal")
            if not is_number(odds_decimal):
                continue
            if name == "1":
                prices["home"] = float(odds_decimal)
            elif name == "X":
                prices["draw"] = float(odds_decimal)
            elif name == "2":
                prices["away"] = float(odds_decimal)
        if not all(outcome in prices for outcome in RESULT_ORDER):
            continue

        odds_rows.append(
            Odds(
                date=prediction.date,
                home_team=metadata["home_team"],
                away_team=metadata["away_team"],
                odds_home=prices["home"],
                odds_draw=prices["draw"],
                odds_away=prices["away"],
                source=(
                    f"fotmob:{match_id}:{payload.get('persistentKey', 'unknown')}:"
                    f"{metadata['url']}"
                ),
            )
        )
    return odds_rows


def build_odds_index(odds_rows: Iterable[Odds]) -> dict[tuple[str, str, str], Odds]:
    index: dict[tuple[str, str, str], Odds] = {}
    for odds in odds_rows:
        index[match_key(odds.date, odds.home_team, odds.away_team)] = odds
    return index


def odds_for_prediction(
    prediction: Prediction, odds_index: dict[tuple[str, str, str], Odds]
) -> Odds | None:
    exact = odds_index.get(
        match_key(prediction.date, prediction.home_team, prediction.away_team)
    )
    if exact is not None:
        return exact

    reversed_odds = odds_index.get(
        match_key(prediction.date, prediction.away_team, prediction.home_team)
    )
    if reversed_odds is None:
        return None
    return Odds(
        date=reversed_odds.date,
        home_team=prediction.home_team,
        away_team=prediction.away_team,
        odds_home=reversed_odds.odds_away,
        odds_draw=reversed_odds.odds_draw,
        odds_away=reversed_odds.odds_home,
        source=f"{reversed_odds.source}; reversed fixture orientation",
    )


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        return 0.0
    fraction = (probability * decimal_odds - 1.0) / (decimal_odds - 1.0)
    return max(0.0, fraction)


def prediction_probabilities(prediction: Prediction) -> dict[str, float]:
    return {
        "draw": prediction.prob_draw,
        "home": prediction.prob_home,
        "away": prediction.prob_away,
    }


def match_odds(odds: Odds) -> dict[str, float]:
    return {
        "draw": odds.odds_draw,
        "home": odds.odds_home,
        "away": odds.odds_away,
    }


def backtest(
    matches: list[BacktestMatch],
    scales: list[float],
    starting_bankroll: float,
    max_total_stake_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    bet_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    matches = sorted(
        matches,
        key=lambda match: (
            match.prediction.date,
            match.prediction.home_team,
            match.prediction.away_team,
        ),
    )

    for scale in scales:
        bankroll = starting_bankroll
        peak_bankroll = starting_bankroll
        max_drawdown = 0.0
        total_staked = 0.0
        total_profit = 0.0
        num_bets = 0
        matches_with_odds = 0
        matches_with_bets = 0
        trajectory_rows.append(
            {
                "scale": scale,
                "match_index": 0,
                "date": None,
                "home_team": None,
                "away_team": None,
                "bankroll": bankroll,
            }
        )

        for match_index, match in enumerate(matches, start=1):
            if match.odds is None:
                trajectory_rows.append(
                    {
                        "scale": scale,
                        "match_index": match_index,
                        "date": match.prediction.date,
                        "home_team": match.prediction.home_team,
                        "away_team": match.prediction.away_team,
                        "bankroll": bankroll,
                    }
                )
                continue

            matches_with_odds += 1
            probabilities = prediction_probabilities(match.prediction)
            odds_values = match_odds(match.odds)
            raw_fractions = {
                outcome: scale
                * kelly_fraction(probabilities[outcome], odds_values[outcome])
                for outcome in RESULT_ORDER
            }
            total_fraction = sum(raw_fractions.values())
            if total_fraction > max_total_stake_fraction and total_fraction > 0:
                shrink = max_total_stake_fraction / total_fraction
                fractions = {
                    outcome: raw_fractions[outcome] * shrink for outcome in RESULT_ORDER
                }
            else:
                fractions = raw_fractions

            stakes = {
                outcome: bankroll * fractions[outcome] for outcome in RESULT_ORDER
            }
            match_staked = sum(stakes.values())
            if match_staked <= 0:
                trajectory_rows.append(
                    {
                        "scale": scale,
                        "match_index": match_index,
                        "date": match.prediction.date,
                        "home_team": match.prediction.home_team,
                        "away_team": match.prediction.away_team,
                        "bankroll": bankroll,
                    }
                )
                continue

            before_bankroll = bankroll
            match_profit = 0.0
            matches_with_bets += 1
            for outcome in RESULT_ORDER:
                stake = stakes[outcome]
                if stake <= 0:
                    continue
                num_bets += 1
                won = outcome == match.result.result
                profit = stake * (odds_values[outcome] - 1.0) if won else -stake
                match_profit += profit
                total_staked += stake
                total_profit += profit
                bet_rows.append(
                    {
                        "scale": scale,
                        "date": match.prediction.date,
                        "home_team": match.prediction.home_team,
                        "away_team": match.prediction.away_team,
                        "match_index": match_index,
                        "outcome": outcome,
                        "probability": probabilities[outcome],
                        "decimal_odds": odds_values[outcome],
                        "full_kelly_fraction": kelly_fraction(
                            probabilities[outcome], odds_values[outcome]
                        ),
                        "scaled_fraction": fractions[outcome],
                        "stake": stake,
                        "won": won,
                        "profit": profit,
                        "bankroll_before_match": before_bankroll,
                    }
                )

            bankroll += match_profit
            peak_bankroll = max(peak_bankroll, bankroll)
            if peak_bankroll > 0:
                max_drawdown = max(
                    max_drawdown, (peak_bankroll - bankroll) / peak_bankroll
                )
            trajectory_rows.append(
                {
                    "scale": scale,
                    "match_index": match_index,
                    "date": match.prediction.date,
                    "home_team": match.prediction.home_team,
                    "away_team": match.prediction.away_team,
                    "bankroll": bankroll,
                }
            )

        roi = total_profit / total_staked if total_staked else 0.0
        summary_rows.append(
            {
                "scale": scale,
                "starting_bankroll": starting_bankroll,
                "final_bankroll": bankroll,
                "profit": bankroll - starting_bankroll,
                "total_staked": total_staked,
                "roi_on_staked": roi,
                "return_on_starting_bankroll": bankroll / starting_bankroll - 1.0,
                "num_bets": num_bets,
                "completed_matches": len(matches),
                "matches_with_odds": matches_with_odds,
                "matches_with_bets": matches_with_bets,
                "max_drawdown": max_drawdown,
            }
        )
    return summary_rows, bet_rows, trajectory_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_bankroll_plot(path: Path, trajectory_rows: list[dict[str, Any]]) -> None:
    plot_data = pd.DataFrame(trajectory_rows)
    if plot_data.empty:
        return
    plot_data = plot_data[plot_data["scale"] != 0.0].copy()
    if plot_data.empty:
        return
    plot_data["scale_label"] = plot_data["scale"].map(lambda scale: f"{scale:g}x")

    plot = (
        pn.ggplot(
            plot_data,
            pn.aes(
                x="match_index",
                y="bankroll",
                color="scale_label",
                group="scale_label",
            ),
        )
        + pn.geom_line(size=1.1)
        + pn.geom_point(size=2.0)
        + pn.labs(
            x="Match index",
            y="Bankroll",
            color="Kelly scale",
            title="Scaled Kelly bankroll trajectory",
        )
        + pn.theme_minimal()
        + pn.theme(
            figure_size=(9, 5),
            plot_title=pn.element_text(size=14, weight="bold"),
            legend_position="right",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    plot.save(path, dpi=160, verbose=False)


def load_odds(args: argparse.Namespace, predictions: list[Prediction]) -> list[Odds]:
    cache_dir = args.output_dir / "cache"
    return odds_from_fotmob(predictions, cache_dir, args.fotmob_ccode3)


def build_backtest_matches(
    predictions: list[Prediction],
    result_index: dict[tuple[str, str, str], MatchResult],
    odds_index: dict[tuple[str, str, str], Odds],
) -> tuple[list[BacktestMatch], list[dict[str, Any]]]:
    matches: list[BacktestMatch] = []
    diagnostics: list[dict[str, Any]] = []
    for prediction in predictions:
        result = result_for_prediction(prediction, result_index)
        odds = odds_for_prediction(prediction, odds_index)
        status = []
        if result is None:
            status.append("missing_result")
        if odds is None:
            status.append("missing_odds")
        diagnostics.append(
            {
                "date": prediction.date,
                "home_team": prediction.home_team,
                "away_team": prediction.away_team,
                "prob_draw": prediction.prob_draw,
                "prob_home": prediction.prob_home,
                "prob_away": prediction.prob_away,
                "home_score": None if result is None else result.home_score,
                "away_score": None if result is None else result.away_score,
                "actual_result": None if result is None else result.result,
                "odds_home": None if odds is None else odds.odds_home,
                "odds_draw": None if odds is None else odds.odds_draw,
                "odds_away": None if odds is None else odds.odds_away,
                "status": "ok" if not status else ";".join(status),
                "result_source": None if result is None else result.source,
                "odds_source": None if odds is None else odds.source,
                "prediction_path": str(prediction.prediction_path),
            }
        )
        if result is not None:
            matches.append(
                BacktestMatch(prediction=prediction, result=result, odds=odds)
            )
    return matches, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--scales",
        default="0,0.05,0.1,0.2,0.3,0.5,0.75,1.0",
        help="Comma-separated Kelly scales or start:stop:step.",
    )
    parser.add_argument("--starting-bankroll", type=float, default=100.0)
    parser.add_argument(
        "--max-total-stake-fraction",
        type=float,
        default=1.0,
        help="Caps total stake per match as a fraction of current bankroll.",
    )
    parser.add_argument("--fotmob-ccode3", default=DEFAULT_FOTMOB_CCODE3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scales = parse_scales(args.scales)
    if args.starting_bankroll <= 0:
        raise ValueError("--starting-bankroll must be positive.")
    if not (0 < args.max_total_stake_fraction <= 1):
        raise ValueError("--max-total-stake-fraction must be in (0, 1].")

    predictions = load_predictions(args.prediction_root)
    results = load_results()
    result_index = build_result_index(results)
    completed_predictions = [
        prediction
        for prediction in predictions
        if result_for_prediction(prediction, result_index) is not None
    ]
    odds_rows = load_odds(args, completed_predictions)
    odds_index = build_odds_index(odds_rows)

    matches, match_rows = build_backtest_matches(predictions, result_index, odds_index)
    summary_rows, bet_rows, trajectory_rows = backtest(
        matches=matches,
        scales=scales,
        starting_bankroll=args.starting_bankroll,
        max_total_stake_fraction=args.max_total_stake_fraction,
    )

    write_csv(args.output_dir / "matches.csv", match_rows)
    write_csv(args.output_dir / "bets.csv", bet_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "bankroll_trajectory.csv", trajectory_rows)
    write_bankroll_plot(args.output_dir / "bankroll_plot.png", trajectory_rows)

    ok_count = sum(1 for row in match_rows if row["status"] == "ok")
    completed_count = sum("missing_result" not in row["status"] for row in match_rows)
    missing_result_count = sum("missing_result" in row["status"] for row in match_rows)
    missing_odds_count = sum("missing_odds" in row["status"] for row in match_rows)
    print(f"Loaded predictions: {len(predictions)}")
    print(f"Loaded completed results: {len(results)}")
    print(f"Loaded odds rows: {len(odds_rows)}")
    print(f"Completed predicted matches: {completed_count}")
    print(f"Completed matches with odds: {ok_count}")
    print(f"Missing results: {missing_result_count}")
    print(f"Missing odds: {missing_odds_count}")
    print(f"Wrote: {args.output_dir / 'matches.csv'}")
    print(f"Wrote: {args.output_dir / 'bets.csv'}")
    print(f"Wrote: {args.output_dir / 'summary.csv'}")
    print(f"Wrote: {args.output_dir / 'bankroll_trajectory.csv'}")
    print(f"Wrote: {args.output_dir / 'bankroll_plot.png'}")

    if summary_rows:
        best = max(summary_rows, key=lambda row: row["final_bankroll"])
        print(
            "Best scale by final bankroll: "
            f"{best['scale']} -> {best['final_bankroll']:.4f} "
            f"({best['return_on_starting_bankroll']:.2%})"
        )


if __name__ == "__main__":
    main()
