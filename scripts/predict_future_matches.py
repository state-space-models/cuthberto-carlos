"""Predict the scores and results of future matches using the model trained on past data.

Generates a new folder in outputs/predictions/ with the current date, and saves a subfolder
for each match (if there are new matches to predict).
"""

from jax import numpy as jnp
import jax
import pandas as pd
import os
import json

from cuthberto_carlos.data import download_data, to_jax_data
from cuthberto_carlos.data_types import ResultData, DynamicsOnlyData
from cuthberto_carlos.graphics import make_graphic
from cuthberto_carlos.json_io import save_arraytree, load_arraytree
from cuthberto_carlos import model_moments


FACTORIAL_STATE_PATH = "outputs/live_factorial.json"
PARAMS_FILE = "outputs/moments_params.json"
GAUSS_HERMITE_DEGREE = 32
COLORS_FILE = "assets/team_colors.json"

OUTPUT_FOLDER = f"outputs/predictions/{pd.Timestamp.now().strftime('%Y-%m-%d')}/"


# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


with open(PARAMS_FILE, "r") as f:
    params = json.load(f)["params"]
params = {k: jnp.asarray(v) for k, v in params.items()}

with open(COLORS_FILE, "r") as f:
    team_name_to_colors = json.load(f)

init_chol_cov = jnp.array(params["init_chol_cov"])
max_goals = 8

pd_data, _, teams_id_to_name_dict, teams_name_to_id_dict = download_data(
    max_goals=max_goals, future_matches=True
)
# Future data labelled with home_score=away_score=-1
pd_data_future = pd_data[pd_data["home_score"] == -1].copy()
jax_data_future = to_jax_data(pd_data_future)

factorial_state_and_timestamp = load_arraytree(FACTORIAL_STATE_PATH)
factorial_state = factorial_state_and_timestamp["factorial_state"]
factorial_timestamp = factorial_state_and_timestamp["timestamp"]

_, factorializer, single_team_filter = model_moments.build(**params)


def propagate_and_predict(
    factorial_state, factorial_timestamp: float, match_data: ResultData
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Propagate the state to the time of the match and predict the outcome."""
    dynamics_data = DynamicsOnlyData(
        team_id=jnp.array([match_data.home_team_id, match_data.away_team_id]),
        timestamp=jnp.array([match_data.timestamp, match_data.timestamp]),
        timestamp_previous=jnp.array([factorial_timestamp, factorial_timestamp]),
    )

    factorial_state_two_teams = jax.vmap(factorializer.extract, in_axes=(None, 0))(
        factorial_state,
        dynamics_data.team_id,
    )
    state_prep = jax.vmap(single_team_filter.filter_prepare)(dynamics_data)
    factorial_state_two_teams_match_time = jax.vmap(single_team_filter.filter_combine)(
        factorial_state_two_teams, state_prep
    )

    skills_mean = factorial_state_two_teams_match_time.mean
    skills_chol_cov = factorial_state_two_teams_match_time.chol_cov

    skills_cov = skills_chol_cov @ skills_chol_cov.transpose((0, 2, 1))

    probs_grid, probs_results = model_moments.predict_match(
        skills_mean,
        skills_cov,
        params["alpha"],
        params["beta"],
        scale=1.0,
        max_goals=max_goals,
        gauss_hermite_degree=GAUSS_HERMITE_DEGREE,
    )
    return skills_mean, skills_cov, probs_grid, probs_results


all_predictions = jax.vmap(propagate_and_predict, in_axes=(None, None, 0))(
    factorial_state, factorial_timestamp, jax_data_future
)


# Save the predictions to a JSON file
for i in range(len(jax_data_future.match_index)):
    match_data = pd_data_future.iloc[i]
    skills_mean, skills_cov, probs_grid, probs_results = jax.tree.map(
        lambda x: x[i], all_predictions
    )
    match_date_str = pd.to_datetime(match_data["date"]).strftime("%Y-%m-%d")

    # Output folder = outputs/predictions/today's date/
    match_folder = f"{OUTPUT_FOLDER}{match_date_str}_{match_data.home_team}_{match_data.away_team}/"
    os.makedirs(match_folder, exist_ok=True)

    match_data_file = f"{match_folder}match_data.json"
    predictions_file = f"{match_folder}predictions.json"

    match_data_dict = match_data.to_dict()
    match_data_dict["date"] = (
        match_date_str  # Convert date to string for JSON serialization
    )
    json.dump(match_data_dict, open(match_data_file, "w"))

    save_data = {
        "skills_mean": skills_mean,
        "skills_cov": skills_cov,
        "probs_grid": probs_grid,
        "probs_results": probs_results,
    }
    save_arraytree(save_data, predictions_file)
    make_graphic(
        match_data,
        skills_mean,
        skills_cov,
        probs_grid,
        probs_results,
        team_name_to_colors,
        f"{match_folder}graphic.png",
    )
