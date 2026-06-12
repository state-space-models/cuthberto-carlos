"""Update moments factorial state."""

from jax import numpy as jnp
import jax
import json

from cuthberto_carlos.data_types import DynamicsOnlyData
from cuthberto_carlos.data import download_data, most_recent_timestamp_by_team
from cuthberto_carlos.json_io import load_arraytree, save_arraytree
from cuthberto_carlos import model_moments
from cuthberto_carlos.graphics import plot_team_strengths


FACTORIAL_STATE_FILE = "outputs/live_factorial.json"
TEAM_STRENGTH_PLOT_FILE = "outputs/team_strength_plot.png"

max_goals = 8
pd_data, jax_data, teams_id_to_name_dict, teams_name_to_id_dict = download_data(
    max_goals=max_goals
)
num_teams = len(teams_id_to_name_dict)
init_mean = jnp.array([0.0, 0.0])
params_file = "outputs/moments_params.json"
with open(params_file, "r") as f:
    params = json.load(f)["params"]

params = {k: jnp.asarray(v) for k, v in params.items()}

# Load previously save factorial state
load_data = load_arraytree("outputs/live_factorial.json")
previous_factorial_state = load_data["factorial_state"]
previous_time = load_data["timestamp"]
previous_match_index = load_data["match_index_final"]

# Extract new data
new_pd_data = pd_data[pd_data.index > int(previous_match_index)]
new_jax_data = jax.tree.map(
    lambda x: x[jax_data.match_index > previous_match_index], jax_data
)

# Load filter(s) and factorializer
filter_obj, factorializer, single_team_filter = model_moments.build(init_mean, **params)


# From https://github.com/state-space-models/cuthbert/blob/89bf19036ba8879ed63c91e88059f9be89cf3af2/cuthbert/factorial/filtering.py#L72
# We probably want to modularise this out in cuthbert
def update_factorial(prev_factorial_state, model_inputs):
    """Update the factorial state with the result of a single match."""
    k = None  # No random key for moments filter
    prep_inp = model_inputs
    factorial_inds = factorializer.get_factorial_indices(prep_inp)
    factorial_inds = jnp.asarray(factorial_inds)

    # Extract and join local factors into joint local state
    local_state = factorializer.extract_and_join(prev_factorial_state, prep_inp)

    # Filter the joint local state
    prep_state = filter_obj.filter_prepare(prep_inp, key=k)
    filtered_joint_state = filter_obj.filter_combine(local_state, prep_state)

    # Marginalize and insert filtered joint local state into factorial state
    local_factorial_filtered_state = factorializer.marginalize(
        filtered_joint_state, len(factorial_inds)
    )
    factorial_state = factorializer.insert(
        local_factorial_filtered_state, prev_factorial_state, factorial_inds
    )
    return factorial_state  # , local_factorial_filtered_state


# Update the factorial state with the new matches
live_factorial_state = previous_factorial_state

for i in range(len(new_jax_data.match_index)):
    model_inputs = jax.tree.map(lambda x: x[i], new_jax_data)
    live_factorial_state = update_factorial(live_factorial_state, model_inputs)


# Synchronize other teams to the most recent timestamp
most_recent_timestamp = most_recent_timestamp_by_team(
    pd_data, num_teams, default=previous_time
)
current_time = most_recent_timestamp.max()

sync_data = DynamicsOnlyData(
    team_id=jnp.arange(num_teams),
    timestamp=jnp.broadcast_to(current_time, (num_teams,)),
    timestamp_previous=most_recent_timestamp,
)
live_factorial_state = model_moments.synchronize(
    live_factorial_state, factorializer, single_team_filter, sync_data
)

# Save the factorial state
save_data = {
    "factorial_state": live_factorial_state,
    "match_index_final": model_inputs.match_index[-1],
    "timestamp": current_time,
}
save_arraytree(save_data, FACTORIAL_STATE_FILE)

# Plot the team strengths
plot_team_strengths(
    live_factorial_state, teams_id_to_name_dict, TEAM_STRENGTH_PLOT_FILE
)
