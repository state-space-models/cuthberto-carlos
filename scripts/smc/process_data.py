# from datetime import datetime

# import numpy as np
# import pandas as pd
# import polars as pl
# from jax import numpy as jnp

# from cuthberto_carlos.data import DATA_URL
from cuthberto_carlos.data_types import ResultData

from datetime import datetime
import polars as pl
import jax.numpy as jnp

ORIGIN_DATE = "1872-11-30"
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def process_data_pl(
    start_date: str = "2000-01-01",
    end_date: str = datetime.now().strftime("%Y-%m-%d"),
    max_goals: int = 8,
    future_matches: bool = False,
):
    data = pl.read_csv(DATA_URL, null_values=["NA"])

    # Processing
    # Drop dates with missing scores

    data = data.filter(
        (pl.col("date") >= start_date),
        (pl.col("date") <= end_date),
        (pl.col("home_score") <= max_goals),
        (pl.col("away_score") <= max_goals),
    ).with_columns(
        pl.col("home_score").fill_null(-1),
        pl.col("away_score").fill_null(-1),
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"),
        pl.col("tournament").str.contains("Friendly").alias("friendly")
    ).with_columns(
        (pl.col("date") - pl.col("date").min()).dt.total_days().alias("timestamp_days"),
    )

    # Fix specific matches
    if future_matches:
        data = data.filter(
            (pl.col("home_score") == -1) & (pl.col("away_score") == -1)
        )

    # Build lookup of previous match date per team
    prev_dates = (
        pl.concat([
            data.select(pl.col("home_team").alias("team"), pl.col("date")),
            data.select(pl.col("away_team").alias("team"), pl.col("date")),
        ])
        .unique()
        .sort(["team", "date"])
        .with_columns(pl.col("date").shift(1).over("team").alias("prev_date"))
    )

    # Left join previous dates for home and away.
    # Store the ABSOLUTE timestamp (days since origin) of the previous match,
    # not the delta, so that the model can compute dt = timestamp - timestamp_previous.
    # A value of 0 means "no previous match" (sentinel), since the origin is 1872-11-30
    # and no real match has timestamp 0.
    data = (
        data
        .join(prev_dates, left_on=["home_team", "date"], right_on=["team", "date"], how="left")
        .rename({"prev_date": "home_prev_date"})
        .join(prev_dates, left_on=["away_team", "date"], right_on=["team", "date"], how="left")
        .rename({"prev_date": "away_prev_date"})
        .with_columns(
            (pl.col("home_prev_date") - pl.col("date").min()).dt.total_days().fill_null(0).alias("home_timestamp_previous"),
            (pl.col("away_prev_date") - pl.col("date").min()).dt.total_days().fill_null(0).alias("away_timestamp_previous"),
        )
        .drop(["home_prev_date", "away_prev_date"])
    )
    # convert teams to IDs
    team_names = sorted(set(data.select(pl.col("home_team")).to_numpy().flatten()) | set(data.select(pl.col("away_team")).to_numpy().flatten()))
    teams_name_to_id = {name: i for i, name in enumerate(team_names)}
    teams_id_to_name = {i: name for i, name in enumerate(team_names)}
    data = data.with_columns(
        pl.col("home_team").replace(teams_name_to_id).cast(pl.Int64).alias("home_team_id"),
        pl.col("away_team").replace(teams_name_to_id).cast(pl.Int64).alias("away_team_id"),
    )
    # create jax data
    jax_data = ResultData(
        match_index=jnp.arange(data.height),
        home_team_id=jnp.array(data.select(pl.col("home_team_id")).to_numpy().flatten()),
        away_team_id=jnp.array(data.select(pl.col("away_team_id")).to_numpy().flatten()),
        home_score=jnp.array(data.select(pl.col("home_score")).to_numpy().flatten()),
        away_score=jnp.array(data.select(pl.col("away_score")).to_numpy().flatten()),
        neutral=jnp.array(data.select(pl.col("neutral")).to_numpy().flatten()),
        friendly=jnp.array(data.select(pl.col("friendly")).to_numpy().flatten()),
        timestamp=jnp.array(data.select(pl.col("timestamp_days")).to_numpy().flatten()),
        home_timestamp_previous=jnp.array(data.select(pl.col("home_timestamp_previous")).to_numpy().flatten()),
        away_timestamp_previous=jnp.array(data.select(pl.col("away_timestamp_previous")).to_numpy().flatten()),
    )

    
    return data, jax_data, teams_id_to_name

def main():
    pd_data, jax_data, id_to_name = process_data_pl(
        start_date="2025-01-01",
        end_date="2026-06-11",
        future_matches=False,
        max_goals=8,
    )
    print(pd_data)
    # print(jax_data)

if __name__ == "__main__":
    main()