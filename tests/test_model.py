from functools import partial
from jax import numpy as jnp, vmap
from cuthbertlib.linearize import linearize_log_density
from cuthbert.gaussian import taylor

from cuthberto_carlos.types import ResultData
from cuthberto_carlos.model import (
    get_init_log_density,
    get_dynamics_log_density,
    get_observation_log_potential,
)

num_teams = 5
init_mean = jnp.array([1.2, 3.2])
init_sd = jnp.array([0.3, 1.1])
tau = jnp.array([0.1, 0.2])
alpha = 0.5
beta = 0.1


filter = taylor.build_filter(
    get_init_log_density=partial(
        get_init_log_density, init_mean=init_mean, init_sd=init_sd, num_teams=num_teams
    ),
    get_dynamics_log_density=partial(get_dynamics_log_density, tau=tau),
    get_observation_func=partial(get_observation_log_potential, alpha=alpha, beta=beta),
)


def test_init():
    init_log_density, lin_point = get_init_log_density(
        None, init_mean, init_sd, num_teams=num_teams
    )

    init_log_density_noneteams, lin_point_noneteams = get_init_log_density(
        None, init_mean, init_sd, num_teams=None
    )

    _, m0, chol_P0 = vmap(linearize_log_density, in_axes=(None, 0, 0))(
        lambda _, x: init_log_density(x),
        lin_point,
        lin_point,
    )

    _, m0_noneteams, chol_P0_noneteams = linearize_log_density(
        lambda _, x: init_log_density_noneteams(x),
        lin_point_noneteams,
        lin_point_noneteams,
    )

    assert jnp.array_equal(jnp.broadcast_to(init_mean, (num_teams, 2)), m0)
    assert jnp.array_equal(
        jnp.broadcast_to(jnp.diag(init_sd), (num_teams, 2, 2)), chol_P0
    )

    assert jnp.array_equal(init_mean, m0_noneteams)
    assert jnp.array_equal(jnp.diag(init_sd), chol_P0_noneteams)


def test_init_filter():
    init_state = filter.init_prepare(None)
    assert jnp.array_equal(jnp.broadcast_to(init_mean, (num_teams, 2)), init_state.mean)
    assert jnp.array_equal(
        jnp.broadcast_to(jnp.diag(init_sd) ** 2, (num_teams, 2, 2)),
        init_state.chol_cov @ init_state.chol_cov.transpose((0, 2, 1)),
    )
    assert jnp.array_equal(0.0, init_state.log_normalizing_constant)


def test_dynamics():
    state = filter.init_prepare(None)
    model_inputs = ResultData(
        match_index=jnp.array([0]),
        home_team_id=jnp.array([0]),
        away_team_id=jnp.array([1]),
        home_score=jnp.array([2]),
        away_score=jnp.array([1]),
        neutral=jnp.array([False]),
        timestamp=jnp.array([10, 10]),
        timestamp_previous=jnp.array([5, 1]),
    )
    log_density, lin_point_prev, lin_point_curr = get_dynamics_log_density(
        state, model_inputs, tau
    )

    mat, shift, chol_cov = linearize_log_density(
        log_density,
        lin_point_prev,
        lin_point_curr,
    )

    assert jnp.allclose(jnp.eye(4), mat)
    assert jnp.array_equal(jnp.zeros(4), shift)
    assert model_inputs.timestamp_previous is not None
    time_diffs = model_inputs.timestamp - model_inputs.timestamp_previous
    time_diffs_repeated = jnp.repeat(time_diffs, 2)  # Repeat for attack and defence
    tau_repeated = jnp.tile(tau, 2)  # Repeat for attack and defence
    desired_cov = jnp.diag((tau_repeated**2) * time_diffs_repeated)
    assert jnp.allclose(desired_cov, chol_cov @ chol_cov.T)
