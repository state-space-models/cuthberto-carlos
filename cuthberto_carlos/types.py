"""NamedTuples to be used as model_inputs representing the data required for the SSM."""

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
        home_timestamp_previous: Optional timestamp for the previous home team match.
        away_timestamp_previous: Optional timestamp for the previous away team match.
    """

    match_index: Array
    home_team_id: Array
    away_team_id: Array
    home_score: Array
    away_score: Array
    neutral: Array
    timestamp: Array
    home_timestamp_previous: Array | None = None
    away_timestamp_previous: Array | None = None


class DynamicsOnlyData(NamedTuple):
    """NamedTuple containing JAX arrays for propagating the state through the dynamics.

    Attributes:
        team_id: Integer ID for the team.
        timestamp: Integer number of days since origin for the time to propagate to.
        timestamp_previous: Integer number of days since origin for the previous state.
    """

    team_id: Array
    timestamp: Array
    timestamp_previous: Array
