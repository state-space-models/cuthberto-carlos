"""Run taylor filtering on the data."""

from functools import partial
from jax import numpy as jnp
import jax
import cuthbert
from cuthbert.factorial.gaussian import build_factorializer
from cuthbert.factorial import filter as factorial_filter
from cuthberto_carlos.data import download_data
from cuthberto_carlos import model

init_mean = jnp.array([0.0, 0.0])
init_sd = jnp.array([1.0, 1.0])
tau = 0.1
kappa = 1e-3
alpha = 1.0
beta = -4.0
max_goals = 6


pd_data, jax_data, teams_id_to_name_dict, teams_name_to_id_dict = download_data(
    max_goals=max_goals
)
num_teams = len(teams_id_to_name_dict)

# Add dummy element at index 0 for each leaf
model_inputs = jax.tree.map(
    lambda x: jnp.concatenate([jnp.zeros_like(x[:1]), x], axis=0), jax_data
)

filter = cuthbert.gaussian.taylor.build_filter(
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


init_factorial_state, local_states = factorial_filter(
    filter, factorializer, model_inputs
)

print(local_states.log_normalizing_constant[:100])
print(local_states.log_normalizing_constant[-1])
