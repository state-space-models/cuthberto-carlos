"""Download historical international football data."""

from typing import NamedTuple
import pandas as pd
from jax import Array, numpy as jnp

DATA_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
# I'm not sure exactly how often this data is updated, but it seems pretty frequently
# Also note it can contain matches in the future (with NA/NaN for scores)


class ResultData(NamedTuple):
    """NamedTuple containing JAX arrays for a football match or set of matches."""

    home_team_id: Array
    away_team_id: Array
    home_score: Array
    away_score: Array
    timestamp_days: Array
    neutral: Array


# TODO: we might want to have the list of teams be static rather than inferred from the
# data, in case a new team appears
def download_data(
    origin_date: str | None = None,
    future_matches: bool = False,
) -> tuple[pd.DataFrame, ResultData, dict[int, str], dict[str, int]]:
    """Download and mildly process historical international football data.

    Args:
        origin_date: The date to use as the origin for the time data.
            Defaults to "1872-11-30", the date of the first international football match.
        future_matches: Whether to include matches with dates in the future.
            Defaults to False.

    Returns:
        A tuple containing:
        - A DataFrame with columns
            (date, home_team, away_team, home_score, away_score, tournament, city,
            country, neutral (bool), timestamp_days, home_team_id, away_team_id)
        - A ResultData NamedTuple containing JAX arrays for the match data
        - A dictionary mapping team IDs to names
        - A dictionary mapping team names to IDs
    """
    if origin_date is None:
        origin_date = "1872-11-30"  # Date of the first international football match

    origin_timestamp = pd.to_datetime(origin_date)

    data_all = pd.read_csv(DATA_URL)

    # Process time data into days since origin date
    data_all["date"] = pd.to_datetime(data_all["date"])
    data_all["timestamp_days"] = (data_all["date"] - origin_timestamp).dt.days

    # Remove future matches if requested
    if not future_matches:
        data_all = data_all[
            data_all["home_score"].notna() & data_all["away_score"].notna()
        ]

    # home_score and away_score are floats because of the NaNs,
    # but we want them to be ints. Fill NaNs with -1 if they exist (int doesn't support NaN)
    data_all["home_score"] = data_all["home_score"].fillna(-1).astype(int)
    data_all["away_score"] = data_all["away_score"].fillna(-1).astype(int)

    # Extract unique teams
    home_counts: pd.Series = data_all["home_team"].value_counts()
    away_counts: pd.Series = data_all["away_team"].value_counts()
    total_counts = home_counts.add(away_counts, fill_value=0)

    # Build team dictionaries and IDs
    team_names = sorted(set(total_counts.index))
    teams_name_to_id_dict = {a: i for i, a in enumerate(team_names)}
    teams_id_to_name_dict = {i: a for i, a in enumerate(team_names)}
    data_all["home_team_id"] = data_all["home_team"].apply(
        lambda s: teams_name_to_id_dict[s]
    )
    data_all["away_team_id"] = data_all["away_team"].apply(
        lambda s: teams_name_to_id_dict[s]
    )

    jax_data = ResultData(
        home_team_id=jnp.array(data_all["home_team_id"].values),
        away_team_id=jnp.array(data_all["away_team_id"].values),
        home_score=jnp.array(data_all["home_score"].values),
        away_score=jnp.array(data_all["away_score"].values),
        timestamp_days=jnp.array(data_all["timestamp_days"].values),
        neutral=jnp.array(data_all["neutral"].values),
    )

    return data_all, jax_data, teams_id_to_name_dict, teams_name_to_id_dict


if __name__ == "__main__":
    data_all, jax_data, teams_id_to_name_dict, teams_name_to_id_dict = download_data()
    print(data_all.tail())
    print(jax_data)
