"""Run moments filtering on the data, for arbitrary (not learnt) static parameters."""

from functools import partial
from jax import numpy as jnp
import jax
import pandas as pd
import plotnine as pn
import json
from cuthbert.gaussian import moments
from cuthbert.factorial.gaussian import build_factorializer
from cuthbert.factorial import filter as factorial_filter

from cuthberto_carlos.data import download_data
from cuthberto_carlos.data_types import DynamicsOnlyData
from cuthberto_carlos.json_io import save_arraytree
from cuthberto_carlos import model_moments


max_goals = 8


pd_data, jax_data, teams_id_to_name_dict, teams_name_to_id_dict = download_data(
    max_goals=max_goals
)

average_goals_per_team_in_a_draw = (
    pd_data[pd_data["home_score"] == pd_data["away_score"]][
        ["home_score", "away_score"]
    ]
    .to_numpy()
    .mean()
)

init_mean = jnp.array([0.0, 0.0])

# init_cov = jnp.array([[1.0, 0.2], [0.2, 1.0]])
# kappa = 1e-4
# friendly_scale = 1.0
# alpha = jnp.log(
#     average_goals_per_team_in_a_draw
# )  # exp(alpha) is expected goals for an evenly matched game
# beta = -2.0


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

filter = moments.build_filter(
    get_init_params=partial(
        model_moments.get_init_params,
        init_mean=init_mean,
        init_chol_cov=init_chol_cov,
        num_teams=num_teams,
    ),
    get_dynamics_params=partial(
        model_moments.get_dynamics_params,
        init_mean=init_mean,
        init_chol_cov=init_chol_cov,
        kappa=params["kappa"],
    ),
    get_observation_params=partial(
        model_moments.get_observation_params,
        alpha=params["alpha"],
        beta=params["beta"],
        friendly_scale=params["friendly_scale"],
    ),
)
factorializer = build_factorializer(
    get_factorial_indices=model_moments.get_factorial_inds
)


# init_factorial_state, local_states, final_factorial_state  = factorial_filter(
#     filter, factorializer, model_inputs
# )

_, _, out_factorial_final = factorial_filter(
    filter, factorializer, model_inputs, output_factorial=False
)


## Synchronize to the most recent timestamp for each team
# Extract the most recent timestamp for each team
timestamps = jnp.array(pd_data["timestamp_days"].to_numpy())
most_recent_timestamp_by_team = jnp.zeros(num_teams, dtype=timestamps.dtype)
most_recent_timestamp_by_team = most_recent_timestamp_by_team.at[
    jnp.array(pd_data["home_team_id"].to_numpy())
].max(timestamps)
most_recent_timestamp_by_team = most_recent_timestamp_by_team.at[
    jnp.array(pd_data["away_team_id"].to_numpy())
].max(timestamps)
current_time = model_inputs.timestamp[-1]
# Create the DynamicsOnlyData for each team
sync_data = DynamicsOnlyData(
    team_id=jnp.arange(num_teams),
    timestamp=jnp.broadcast_to(current_time, (num_teams,)),
    timestamp_previous=most_recent_timestamp_by_team,
)

single_team_filter = moments.build_filter(
    get_init_params=partial(
        model_moments.get_init_params,
        init_mean=init_mean,
        init_chol_cov=init_chol_cov,
    ),
    get_dynamics_params=partial(
        model_moments.get_dynamics_params,
        init_mean=init_mean,
        init_chol_cov=init_chol_cov,
        kappa=params["kappa"],
    ),
    get_observation_params=model_moments.get_observation_params_noop,
)
out_factorial_final = jax.vmap(factorializer.extract, in_axes=(None, 0))(
    out_factorial_final, jnp.arange(num_teams)
)
state_prep = jax.vmap(single_team_filter.filter_prepare)(sync_data)
sync_factorial_final = jax.vmap(single_team_filter.filter_combine)(
    out_factorial_final, state_prep
)

# Save as json sync_factorial_final and current timestamp to be used later
save_data = {
    "sync_factorial_final": sync_factorial_final,
    "timestamp": current_time,
}
save_arraytree(save_data, "outputs/sync_factorial_final.json")

# # Commented code to reload later
# load_data = load_arraytree("outputs/sync_factorial_final.json")
# sync_factorial_final = load_data["sync_factorial_final"]
# current_time = load_data["timestamp"]

### Plot best teams

# Sorted by total strength (attack + defence), strongest at the top. Only plot top 20 teams
means = sync_factorial_final.mean
covs = sync_factorial_final.chol_cov @ sync_factorial_final.chol_cov.transpose(0, 2, 1)
stds = jax.vmap(lambda cov: jnp.sqrt(jnp.diag(cov)))(covs)
names = [teams_id_to_name_dict[i] for i in range(num_teams)]

top_team_ids = jnp.argsort(means.sum(axis=1))[-20:][::-1]
top_team_ids_list = [int(team_id) for team_id in top_team_ids]
team_order = [names[team_id] for team_id in reversed(top_team_ids_list)]
metrics = ("attack", "defence")
plot_data = pd.DataFrame(
    [
        {
            "team": names[team_id],
            "metric": metric,
            "mean": float(means[team_id, metric_id]),
            "std": float(stds[team_id, metric_id]),
        }
        for team_id in top_team_ids_list
        for metric_id, metric in enumerate(metrics)
    ]
)
plot_data["team"] = pd.Categorical(
    plot_data["team"], categories=team_order, ordered=True
)
plot_data["metric"] = pd.Categorical(
    plot_data["metric"], categories=("attack", "defence"), ordered=True
)
plot_data["metric_order"] = pd.Categorical(
    plot_data["metric"], categories=("defence", "attack"), ordered=True
)

team_strength_plot = (
    pn.ggplot(plot_data, pn.aes("team", "mean", fill="metric", group="metric_order"))
    + pn.geom_col(position=pn.position_dodge(width=0.8), width=0.7)
    + pn.geom_errorbar(
        pn.aes(ymin="mean - std", ymax="mean + std"),
        position=pn.position_dodge(width=0.8),
        width=0.25,
    )
    + pn.coord_flip()
    + pn.labs(
        x="Team",
        y="Mean strength",
        fill="Metric",
        title="Top 20 Teams by Total Strength",
    )
    + pn.theme_minimal()
    + pn.theme(figure_size=(10, 7))
)
team_strength_plot.save("outputs/team_strength_plot.png", dpi=300, verbose=False)
