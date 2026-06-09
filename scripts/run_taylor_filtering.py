"""Run taylor filtering on the data, for arbitrary (not learnt) static parameters."""

from functools import partial
from jax import numpy as jnp
import jax
import pandas as pd
import plotnine as pn
from cuthbert.gaussian import taylor
from cuthbert.factorial.gaussian import build_factorializer
from cuthbert.factorial import filter as factorial_filter

from cuthberto_carlos.data import download_data
from cuthberto_carlos.types import DynamicsOnlyData
from cuthberto_carlos import model


max_goals = 8

pd_data, jax_data, teams_id_to_name_dict, teams_name_to_id_dict = download_data(
    max_goals=max_goals
)

average_goals_in_a_draw = (
    pd_data[pd_data["home_score"] == pd_data["away_score"]][
        ["home_score", "away_score"]
    ]
    .to_numpy()
    .mean()
)

init_mean = jnp.array([0.0, 0.0])
init_sd = jnp.array([1.0, 1.0])
tau = 0.01
kappa = 1e-4
alpha = jnp.log(
    average_goals_in_a_draw
)  # exp(alpha) is expected goals for an evenly matched game
beta = -4.0

num_teams = len(teams_id_to_name_dict)

# Add dummy element at index 0 for each leaf
model_inputs = jax.tree.map(
    lambda x: jnp.concatenate([jnp.zeros_like(x[:1]), x], axis=0), jax_data
)

filter = taylor.build_filter(
    get_init_log_density=partial(
        model.get_init_log_density,
        init_mean=init_mean,
        init_sd=init_sd,
        num_teams=num_teams,
    ),
    get_dynamics_log_density=partial(
        model.get_dynamics_log_density, tau=tau, init_mean=init_mean, kappa=kappa
    ),
    get_observation_func=partial(
        model.get_observation_log_potential, alpha=alpha, beta=beta, max_goals=max_goals
    ),
    rtol=1e-7,
)
factorializer = build_factorializer(get_factorial_indices=model.get_factorial_inds)


# init_factorial_state, local_states, final_factorial_state  = factorial_filter(
#     filter, factorializer, model_inputs
# )

out_factorial_all = factorial_filter(
    filter, factorializer, model_inputs, output_factorial=True
)
out_factorial_final = jax.tree.map(lambda x: x[-1], out_factorial_all)


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

single_team_filter = taylor.build_filter(
    get_init_log_density=partial(
        model.get_init_log_density,
        init_mean=init_mean,
        init_sd=init_sd,
    ),
    get_dynamics_log_density=partial(
        model.get_dynamics_log_density, tau=tau, init_mean=init_mean, kappa=kappa
    ),
    get_observation_func=model.get_observation_log_potential_noop,
)
out_factorial_final = jax.vmap(factorializer.extract, in_axes=(None, 0))(
    out_factorial_final, jnp.arange(num_teams)
)
state_prep = jax.vmap(single_team_filter.filter_prepare)(sync_data)
sync_factorial_final = jax.vmap(single_team_filter.filter_combine)(
    out_factorial_final, state_prep
)

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
# team_strength_plot.save("taylor_filtering_team_strengths.png", dpi=300, verbose=False)
team_strength_plot.show()
