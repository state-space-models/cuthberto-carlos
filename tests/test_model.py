from functools import partial
from jax import numpy as jnp, vmap
from cuthbertlib.linearize import linearize_log_density
from cuthbert.gaussian import taylor

from cuthberto_carlos import bivariate_poisson
from cuthberto_carlos.types import ResultData
from cuthberto_carlos.model import (
    get_init_log_density,
    get_dynamics_log_density,
    get_observation_log_potential,
)

num_teams = 5
init_mean = jnp.array([1.2, 3.2])
init_cov = jnp.array([[0.09, 0.12], [0.12, 1.21]])
init_chol_cov = jnp.linalg.cholesky(init_cov)
tau = jnp.array([0.1, 0.2])
kappa = jnp.array([0.0, 0.0])
alpha = 1.0
beta = -4.0


filter = taylor.build_filter(
    get_init_log_density=partial(
        get_init_log_density,
        init_mean=init_mean,
        init_chol_cov=init_chol_cov,
        num_teams=num_teams,
    ),
    get_dynamics_log_density=partial(
        get_dynamics_log_density, tau=tau, init_mean=init_mean, kappa=kappa
    ),
    get_observation_func=partial(get_observation_log_potential, alpha=alpha, beta=beta),
)


def test_init():
    init_log_density, lin_point = get_init_log_density(
        None, init_mean, init_cov=init_cov, num_teams=num_teams
    )

    init_log_density_noneteams, lin_point_noneteams = get_init_log_density(
        None, init_mean, init_chol_cov=init_chol_cov, num_teams=None
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

    assert jnp.allclose(jnp.broadcast_to(init_mean, (num_teams, 2)), m0)
    assert jnp.allclose(
        jnp.broadcast_to(init_cov, (num_teams, 2, 2)),
        chol_P0 @ chol_P0.transpose((0, 2, 1)),
    )

    assert jnp.allclose(init_mean, m0_noneteams)
    assert jnp.allclose(init_cov, chol_P0_noneteams @ chol_P0_noneteams.T)


def test_init_filter():
    init_state = filter.init_prepare(None)
    assert jnp.allclose(jnp.broadcast_to(init_mean, (num_teams, 2)), init_state.mean)
    assert jnp.allclose(
        jnp.broadcast_to(init_cov, (num_teams, 2, 2)),
        init_state.chol_cov @ init_state.chol_cov.transpose((0, 2, 1)),
    )
    assert jnp.array_equal(0.0, init_state.log_normalizing_constant)


def test_dynamics():
    state = filter.init_prepare(None)
    model_inputs = ResultData(
        match_index=jnp.array(0),
        home_team_id=jnp.array(0),
        away_team_id=jnp.array(1),
        home_score=jnp.array(2),
        away_score=jnp.array(1),
        neutral=jnp.array(False),
        timestamp=jnp.array([10, 10]),
        home_timestamp_previous=jnp.array(5),
        away_timestamp_previous=jnp.array(1),
    )
    log_density, lin_point_prev, lin_point_curr = get_dynamics_log_density(
        state, model_inputs, tau, init_mean, kappa
    )

    mat, shift, chol_cov = linearize_log_density(
        log_density,
        lin_point_prev,
        lin_point_curr,
    )

    assert jnp.allclose(jnp.eye(4), mat)
    assert jnp.array_equal(jnp.zeros(4), shift)
    assert model_inputs.home_timestamp_previous is not None
    assert model_inputs.away_timestamp_previous is not None
    time_diffs = model_inputs.timestamp - jnp.array(
        [
            model_inputs.home_timestamp_previous,
            model_inputs.away_timestamp_previous,
        ]
    )
    time_diffs_repeated = jnp.repeat(time_diffs, 2)  # Repeat for attack and defence
    tau_repeated = jnp.tile(tau, 2)  # Repeat for attack and defence
    desired_cov = jnp.diag((tau_repeated**2) * time_diffs_repeated)
    assert jnp.allclose(desired_cov, chol_cov @ chol_cov.T)


def test_dynamics_uses_std_floor_for_zero_elapsed_time():
    state = filter.init_prepare(None)
    model_inputs = ResultData(
        match_index=jnp.array(0),
        home_team_id=jnp.array(0),
        away_team_id=jnp.array(1),
        home_score=jnp.array(2),
        away_score=jnp.array(1),
        neutral=jnp.array(False),
        timestamp=jnp.array([10, 10]),
        home_timestamp_previous=jnp.array(10),
        away_timestamp_previous=jnp.array(10),
    )
    std_floor = jnp.array([0.002, 0.003])
    log_density, lin_point_prev, lin_point_curr = get_dynamics_log_density(
        state, model_inputs, tau, init_mean, kappa, std_floor=std_floor
    )

    _, _, chol_cov = linearize_log_density(
        log_density,
        lin_point_prev,
        lin_point_curr,
    )

    desired_cov = jnp.diag(jnp.tile(std_floor, 2) ** 2)
    assert jnp.allclose(desired_cov, chol_cov @ chol_cov.T)


def test_observation_uses_friendly_ability_scale():
    state = filter.init_prepare(None)
    x = jnp.array([0.4, 0.2, -0.1, 0.3])
    friendly_scale = 2.0
    competitive_scale = 1.0
    base_inputs = {
        "match_index": jnp.array(0),
        "home_team_id": jnp.array(0),
        "away_team_id": jnp.array(1),
        "home_score": jnp.array(2),
        "away_score": jnp.array(1),
        "neutral": jnp.array(False),
        "timestamp": jnp.array(10),
        "home_timestamp_previous": jnp.array(5),
        "away_timestamp_previous": jnp.array(1),
    }
    friendly_inputs = ResultData(**base_inputs, is_friendly=jnp.array(True))
    competitive_inputs = ResultData(**base_inputs, is_friendly=jnp.array(False))

    friendly_potential, _ = get_observation_log_potential(
        state,
        friendly_inputs,
        alpha=alpha,
        beta=beta,
        friendly_ability_scale=friendly_scale,
        competitive_ability_scale=competitive_scale,
    )
    competitive_potential, _ = get_observation_log_potential(
        state,
        competitive_inputs,
        alpha=alpha,
        beta=beta,
        friendly_ability_scale=friendly_scale,
        competitive_ability_scale=competitive_scale,
    )
    y = jnp.array([base_inputs["home_score"], base_inputs["away_score"]])

    assert jnp.allclose(
        friendly_potential(x),
        bivariate_poisson.loglik(
            y, x[:2], x[2:], alpha, beta, 8, ability_scale=friendly_scale
        ),
    )
    assert jnp.allclose(
        competitive_potential(x),
        bivariate_poisson.loglik(
            y, x[:2], x[2:], alpha, beta, 8, ability_scale=competitive_scale
        ),
    )
