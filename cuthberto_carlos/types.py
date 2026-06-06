"""Defines the ResultData NamedTuple for storing JAX arrays of football match data."""

from typing import NamedTuple
from jax import Array


class ResultData(NamedTuple):
    """NamedTuple containing JAX arrays for a football match or set of matches.

    Attributes:
        match_index: Unique integer index for each match, starting from 0.
        home_team_id: Integer ID for the home team.
        away_team_id: Integer ID for the away team.
        home_score: Integer number of goals scored by the home team in the match.
        away_score: Integer number of goals scored by the away team in the match.
        neutral: Boolean indicating whether the match was played on neutral ground.
        timestamp: Integer number of days since the origin date for the match.
        timestamp_previous: Optional timestamp for the previous match of the team(s).
            Used to determine the time between matches for the dynamics log density.
    """

    match_index: Array
    home_team_id: Array
    away_team_id: Array
    home_score: Array
    away_score: Array
    neutral: Array
    timestamp: Array
    timestamp_previous: Array | None = None
