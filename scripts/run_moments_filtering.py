"""Run moments filtering on the data, for arbitrary (not learnt) static parameters."""

from jax import numpy as jnp
import jax
import json
from cuthbert.factorial import filter as factorial_filter

from cuthberto_carlos.data import download_data, most_recent_timestamp_by_team
from cuthberto_carlos.data_types import DynamicsOnlyData
from cuthberto_carlos.json_io import save_arraytree
from cuthberto_carlos import model_moments
from cuthberto_carlos.graphics import plot_team_strengths

FACTORIAL_STATE_FILE = "outputs/live_factorial.json"
TEAM_STRENGTH_PLOT_FILE = "outputs/team_strength_plot.png"

max_goals = 8
pd_data, jax_data, teams_id_to_name_dict, teams_name_to_id_dict = download_data(
    max_goals=max_goals
)

init_mean = jnp.array([0.0, 0.0])
params_file = "outputs/moments_params.json"
with open(params_file, "r") as f:
    params = json.load(f)["params"]

params = {k: jnp.asarray(v) for k, v in params.items()}

num_teams = len(teams_id_to_name_dict)

init_chol_cov = jnp.linalg.cholesky(params["init_cov"])

# Add dummy element at index 0 for each leaf
model_inputs = jax.tree.map(
    lambda x: jnp.concatenate([jnp.zeros_like(x[:1]), x], axis=0), jax_data
)

filter_obj, factorializer, single_team_filter = model_moments.build(init_mean, **params)

_, _, out_factorial_final = factorial_filter(
    filter_obj, factorializer, model_inputs, output_factorial=False
)

## Synchronize to the most recent timestamp for each team
# Extract the most recent timestamp for each team
most_recent_timestamp = most_recent_timestamp_by_team(pd_data, num_teams)
current_time = model_inputs.timestamp[-1]
# Create the DynamicsOnlyData for each team
sync_data = DynamicsOnlyData(
    team_id=jnp.arange(num_teams),
    timestamp=jnp.broadcast_to(current_time, (num_teams,)),
    timestamp_previous=most_recent_timestamp,
)

sync_factorial_final = model_moments.synchronize(
    out_factorial_final, factorializer, single_team_filter, sync_data
)

# Save as json sync_factorial_final and current timestamp to be used later
save_data = {
    "factorial_state": sync_factorial_final,
    "match_index_final": model_inputs.match_index[-1],
    "timestamp": current_time,
}
save_arraytree(save_data, FACTORIAL_STATE_FILE)


### Plot best teams
plot_team_strengths(
    sync_factorial_final, teams_id_to_name_dict, TEAM_STRENGTH_PLOT_FILE
)
