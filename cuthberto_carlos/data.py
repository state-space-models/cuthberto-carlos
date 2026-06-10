"""Download historical international football data."""

import pandas as pd
import numpy as np
from jax import numpy as jnp

from cuthberto_carlos.types import ResultData

DATA_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
# I'm not sure exactly how often this data is updated, but it seems pretty frequently
# Also note it can contain matches in the future (with NA/NaN for home_score and away_score)


# TODO: we might want to have the list of teams be static rather than inferred from the
# data, in case a new team appears
def download_data(
    origin_date: str | None = None,
    future_matches: bool = False,
    max_goals: int = int(1e6),
) -> tuple[pd.DataFrame, ResultData, dict[int, str], dict[str, int]]:
    """Download and mildly process historical international football data.

    Args:
        origin_date: The date to use as the origin for the time data.
            Defaults to "1872-11-30", the date of the first international football match.
        future_matches: Whether to include matches with dates in the future.
            Defaults to False.
        max_goals: Removes matches where either team scored more than this many goals.
            Defaults to a very large number.

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

    # Remove matches with too many goals
    data_all = data_all[
        (data_all["home_score"] <= max_goals) & (data_all["away_score"] <= max_goals)
    ]

    # home_score and away_score are floats because of the NaNs,
    # but we want them to be ints. Fill NaNs with -1 if they exist (int doesn't support NaN)
    data_all["home_score"] = data_all["home_score"].fillna(-1).astype(int)
    data_all["away_score"] = data_all["away_score"].fillna(-1).astype(int)

    # The filter is sequential, so match updates must be applied chronologically.
    data_all = data_all.sort_values("timestamp_days", kind="stable").reset_index(
        drop=True
    )
    data_all["is_friendly"] = data_all["tournament"] == "Friendly"

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

    # Extract previous timestamps for home and away teams
    num_matches = len(data_all)
    match_positions = np.arange(num_matches)
    timestamps = data_all["timestamp_days"].to_numpy()
    team_ids = np.concatenate(
        [
            data_all["home_team_id"].to_numpy(),
            data_all["away_team_id"].to_numpy(),
        ]
    )
    match_positions_by_team = np.concatenate([match_positions, match_positions])
    timestamps_by_team = np.concatenate([timestamps, timestamps])
    is_home_team = np.concatenate(
        [np.ones(num_matches, dtype=bool), np.zeros(num_matches, dtype=bool)]
    )
    order = np.lexsort((match_positions_by_team, timestamps_by_team, team_ids))
    previous_timestamps = np.zeros(2 * num_matches, dtype=timestamps.dtype)
    same_team_as_previous = team_ids[order][1:] == team_ids[order][:-1]
    previous_timestamps[order[1:]] = np.where(
        same_team_as_previous,
        timestamps_by_team[order[:-1]],
        0,
    )
    data_all["home_timestamp_previous"] = previous_timestamps[is_home_team]
    data_all["away_timestamp_previous"] = previous_timestamps[~is_home_team]

    jax_data = ResultData(
        match_index=jnp.array(data_all.index.values),
        home_team_id=jnp.array(data_all["home_team_id"].values),
        away_team_id=jnp.array(data_all["away_team_id"].values),
        home_score=jnp.array(data_all["home_score"].values),
        away_score=jnp.array(data_all["away_score"].values),
        timestamp=jnp.array(data_all["timestamp_days"].values),
        neutral=jnp.array(data_all["neutral"].values),
        home_timestamp_previous=jnp.array(data_all["home_timestamp_previous"].values),
        away_timestamp_previous=jnp.array(data_all["away_timestamp_previous"].values),
        is_friendly=jnp.array(data_all["is_friendly"].values),
    )

    return data_all, jax_data, teams_id_to_name_dict, teams_name_to_id_dict


if __name__ == "__main__":
    data_all, jax_data, teams_id_to_name_dict, teams_name_to_id_dict = download_data()
    print("Pandas dataframe: \n")
    print(data_all)
    print("\n\n JAX arrays:\n")
    print(jax_data)
