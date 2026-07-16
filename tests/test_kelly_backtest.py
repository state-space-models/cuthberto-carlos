"""Tests for Polymarket odds used by the Kelly backtest."""

from pathlib import Path
import tempfile
from typing import cast
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.kelly_backtest import (
    Prediction,
    polymarket_decimal_odds,
    polymarket_event_outcomes,
    polymarket_price_as_of,
)


class PolymarketOddsTests(unittest.TestCase):
    """Validate conversion and matching of Polymarket historical prices."""

    def test_price_as_of_uses_last_valid_trade_before_cutoff(self):
        with patch(
            "scripts.kelly_backtest.polymarket_price_history",
            return_value=[
                {"t": 100, "p": 0.4},
                {"t": 200, "p": 0.6},
                {"t": 300, "p": 0.8},
                {"t": 250, "p": 1.0},
            ],
        ):
            result = polymarket_price_as_of(
                "token",
                cast(pd.Timestamp, pd.Timestamp("1970-01-01T00:03:20Z")),
                Path("cache"),
            )

        self.assertEqual(result, (0.6, pd.Timestamp("1970-01-01T00:03:20Z")))

    def test_decimal_odds_include_taker_fee(self):
        odds = polymarket_decimal_odds(
            0.5,
            {"feesEnabled": True, "feeSchedule": {"rate": 0.05}},
        )

        self.assertAlmostEqual(odds, 1 / (0.5 * (1 + 0.05 * 0.5)))

    def test_event_outcomes_match_reversed_fixture_orientation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch = Path(temporary_directory) / "2026-06-11"
            prediction = Prediction(
                date="2026-06-11",
                home_team="South Africa",
                away_team="Mexico",
                prob_draw=0.2,
                prob_home=0.1,
                prob_away=0.7,
                prediction_path=batch / "fixture" / "predictions.json",
            )
            event = {
                "title": "Mexico vs. South Africa",
                "slug": "fifwc-mex-rsa-2026-06-11",
                "markets": [
                    {
                        "sportsMarketType": "moneyline",
                        "gameStartTime": "2026-06-11T19:00:00Z",
                        "marketMetadata": {"opticOddsSelection": "Mexico"},
                        "clobTokenIds": '["mexico"]',
                    },
                    {
                        "sportsMarketType": "moneyline",
                        "marketMetadata": {"opticOddsSelection": "Draw"},
                        "clobTokenIds": '["draw"]',
                    },
                    {
                        "sportsMarketType": "moneyline",
                        "marketMetadata": {"opticOddsSelection": "South Africa"},
                        "clobTokenIds": '["south-africa"]',
                    },
                ],
            }
            prices = {"mexico": 0.7, "draw": 0.2, "south-africa": 0.1}
            priced_at = pd.Timestamp("2026-06-10T12:00:00Z")
            cutoffs: list[pd.Timestamp] = []

            def price_as_of(token, cutoff, _cache):
                cutoffs.append(cutoff)
                return prices[token], priced_at

            with patch(
                "scripts.kelly_backtest.polymarket_price_as_of",
                side_effect=price_as_of,
            ):
                odds = polymarket_event_outcomes(event, prediction, Path("cache"))

        self.assertIsNotNone(odds)
        assert odds is not None
        self.assertAlmostEqual(odds.odds_home, 10.0)
        self.assertAlmostEqual(odds.odds_draw, 5.0)
        self.assertAlmostEqual(odds.odds_away, 1 / 0.7)
        self.assertEqual(cutoffs, [pd.Timestamp("2026-06-11T17:00:00Z")] * 3)
