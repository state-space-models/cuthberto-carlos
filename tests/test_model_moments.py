from functools import partial

from jax import numpy as jnp
from cuthbert.factorial.gaussian import build_factorializer
from cuthbert.gaussian import moments
from cuthbertlib.linearize import linearize_moments

from cuthberto_carlos.data_types import DynamicsOnlyData, ResultData
from cuthberto_carlos.model_moments import (
    get_dynamics_params,
    get_factorial_inds,
    get_init_params,
    get_observation_params,
    get_observation_params_noop,
)

num_teams = 5
init_mean = jnp.array([1.2, 3.2])
init_cov = jnp.array([[0.09, -0.12], [-0.12, 1.21]])
init_chol_cov = jnp.linalg.cholesky(init_cov)
kappa = jnp.array([0.1, 0.2])
alpha = 1.0
beta = -4.0
friendly_scale = 1.5


filter = moments.build_filter(
    get_init_params=partial(
        get_init_params,
        init_mean=init_mean,
        init_chol_cov=init_chol_cov,
        num_teams=num_teams,
    ),
    get_dynamics_params=partial(
        get_dynamics_params,
        init_mean=init_mean,
        init_chol_cov=init_chol_cov,
        kappa=kappa,
    ),
    get_observation_params=partial(
        get_observation_params,
        alpha=alpha,
        beta=beta,
        friendly_scale=friendly_scale,
    ),
)
factorializer = build_factorializer(get_factorial_indices=get_factorial_inds)


def _result_data() -> ResultData:
    return ResultData(
        match_index=jnp.array(0),
        home_team_id=jnp.array(0),
        away_team_id=jnp.array(1),
        home_score=jnp.array(2),
        away_score=jnp.array(1),
        neutral=jnp.array(False),
        friendly=jnp.array(False),
        timestamp=jnp.array([10, 10]),
        home_timestamp_previous=jnp.array(5),
        away_timestamp_previous=jnp.array(1),
    )


def test_init_params():
    m0, chol_P0 = get_init_params(None, init_mean, init_chol_cov, num_teams=num_teams)
    m0_noneteams, chol_P0_noneteams = get_init_params(
        None, init_mean, init_chol_cov, num_teams=None
    )

    assert jnp.array_equal(jnp.broadcast_to(init_mean, (num_teams, 2)), m0)
    assert jnp.allclose(
        jnp.broadcast_to(init_cov, (num_teams, 2, 2)),
        chol_P0 @ chol_P0.transpose((0, 2, 1)),
    )
    assert jnp.array_equal(init_mean, m0_noneteams)
    assert jnp.allclose(init_cov, chol_P0_noneteams @ chol_P0_noneteams.T)


def test_init_filter():
    init_state = filter.init_prepare(None)
    assert jnp.array_equal(jnp.broadcast_to(init_mean, (num_teams, 2)), init_state.mean)
    assert jnp.allclose(
        jnp.broadcast_to(init_cov, (num_teams, 2, 2)),
        init_state.chol_cov @ init_state.chol_cov.transpose((0, 2, 1)),
    )
    assert jnp.array_equal(0.0, init_state.log_normalizing_constant)


def test_dynamics_params():
    state = filter.init_prepare(None)
    model_inputs = _result_data()
    mean_func, lin_point = get_dynamics_params(
        state, model_inputs, init_mean, init_chol_cov, kappa
    )

    mat, shift, chol_cov = linearize_moments(mean_func, lin_point)

    assert model_inputs.home_timestamp_previous is not None
    assert model_inputs.away_timestamp_previous is not None
    time_diffs = model_inputs.timestamp - jnp.array(
        [
            model_inputs.home_timestamp_previous,
            model_inputs.away_timestamp_previous,
        ]
    )
    time_diffs_repeated = jnp.repeat(time_diffs, 2)
    kappa_repeated = jnp.tile(kappa, 2)
    phi = jnp.exp(-kappa_repeated * time_diffs_repeated)
    init_mean_repeated = jnp.tile(init_mean, 2)
    stationary_cov = jnp.kron(jnp.eye(2), init_cov)
    desired_cov = stationary_cov * (1.0 - phi[:, None] * phi[None, :])

    assert jnp.allclose(jnp.diag(phi), mat)
    assert jnp.allclose(init_mean_repeated * (1.0 - phi), shift)
    assert jnp.allclose(
        desired_cov,
        chol_cov @ chol_cov.T,
        atol=1e-6,
    )


def test_dynamics_stationary_covariance_is_init_cov():
    state = filter.init_prepare(None)
    model_inputs = _result_data()
    mean_func, _ = get_dynamics_params(
        state, model_inputs, init_mean, init_chol_cov, kappa
    )

    assert model_inputs.home_timestamp_previous is not None
    assert model_inputs.away_timestamp_previous is not None
    time_diffs = model_inputs.timestamp - jnp.array(
        [
            model_inputs.home_timestamp_previous,
            model_inputs.away_timestamp_previous,
        ]
    )
    time_diffs_repeated = jnp.repeat(time_diffs, 2)
    phi = jnp.exp(-jnp.tile(kappa, 2) * time_diffs_repeated)
    stationary_cov = jnp.kron(jnp.eye(2), init_cov)
    transition_mean, transition_chol_cov = mean_func(jnp.tile(init_mean, 2))
    propagated_cov = (
        jnp.diag(phi) @ stationary_cov @ jnp.diag(phi)
        + transition_chol_cov @ transition_chol_cov.T
    )

    assert jnp.allclose(jnp.tile(init_mean, 2), transition_mean)
    assert jnp.allclose(stationary_cov, propagated_cov, atol=1e-6)


def test_observation_params_match_bivariate_poisson_moments():
    init_state = filter.init_prepare(None)
    local_state = factorializer.join(
        factorializer.extract(init_state, jnp.array([0, 1]))
    )
    model_inputs = _result_data()
    obs_func, lin_point, y = get_observation_params(
        local_state, model_inputs, alpha, beta, friendly_scale
    )

    x = jnp.array([0.2, -0.1, -0.3, 0.4])
    mean, chol_cov = obs_func(x)
    lambda_1 = jnp.exp(alpha + x[0] - x[3])
    lambda_2 = jnp.exp(alpha + x[2] - x[1])
    lambda_3 = jnp.exp(beta)
    expected_mean = jnp.array([lambda_1 + lambda_3, lambda_2 + lambda_3])
    expected_cov = jnp.array(
        [
            [lambda_1 + lambda_3, lambda_3],
            [lambda_3, lambda_2 + lambda_3],
        ]
    )

    assert jnp.array_equal(local_state.mean, lin_point)
    assert jnp.array_equal(jnp.array([2, 1]), y)
    assert jnp.allclose(expected_mean, mean)
    assert jnp.allclose(expected_cov, chol_cov @ chol_cov.T)


def test_noop_observation_leaves_dynamics_only_update_at_prediction():
    single_team_filter = moments.build_filter(
        get_init_params=partial(
            get_init_params,
            init_mean=init_mean,
            init_chol_cov=init_chol_cov,
            num_teams=None,
        ),
        get_dynamics_params=partial(
            get_dynamics_params,
            init_mean=init_mean,
            init_chol_cov=init_chol_cov,
            kappa=kappa,
        ),
        get_observation_params=get_observation_params_noop,
    )
    init_state = single_team_filter.init_prepare(None)
    model_inputs = DynamicsOnlyData(
        team_id=jnp.array(0),
        timestamp=jnp.array(10),
        timestamp_previous=jnp.array(0),
    )

    prep_state = single_team_filter.filter_prepare(model_inputs)
    update_state = single_team_filter.filter_combine(init_state, prep_state)

    assert jnp.allclose(init_mean, update_state.mean)
    assert jnp.allclose(
        init_cov,
        update_state.chol_cov @ update_state.chol_cov.T,
        atol=1e-6,
    )
    assert jnp.allclose(0.0, update_state.log_normalizing_constant)
