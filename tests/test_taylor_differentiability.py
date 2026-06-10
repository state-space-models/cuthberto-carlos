"""Differentiability checks for non-factorial Taylor filtering."""

from functools import partial
from typing import Any, NamedTuple, cast

import jax
from jax import Array, numpy as jnp
from jax.scipy.stats import norm
import pytest
from cuthbert.factorial import filter as factorial_filter
from cuthbert.factorial.gaussian import build_factorializer
from cuthbert.filtering import filter as run_filter
from cuthbert.gaussian import taylor

from cuthberto_carlos import model
from cuthberto_carlos.types import ResultData


class GaussianData(NamedTuple):
    """Observations for a tiny one-dimensional Gaussian SSM."""

    y: Array


class ObservationState(NamedTuple):
    """Minimal state object for directly testing observation potentials."""

    mean: Array


def _gaussian_init(model_inputs, init_sd):
    def log_density(x):
        return norm.logpdf(x[0], 0.0, init_sd)

    return log_density, jnp.array([0.0])


def _gaussian_dynamics(state, model_inputs, tau):
    def log_density(x_prev, x):
        return norm.logpdf(x[0], x_prev[0], tau)

    return log_density, jnp.array([0.0]), jnp.array([0.0])


def _gaussian_observation(state, model_inputs, obs_sd):
    def log_potential(x):
        return norm.logpdf(model_inputs.y, x[0], obs_sd)

    return log_potential, state.mean


def test_nonfactorial_taylor_filter_is_differentiable_for_gaussian_model():
    """A full-rank Gaussian Taylor-filter objective has finite gradients."""
    model_inputs = GaussianData(y=jnp.array([0.0, 0.2, -0.1, 0.3]))

    def objective(raw_params):
        init_sd, tau, obs_sd = jax.nn.softplus(raw_params) + 1e-6
        filter_obj = taylor.build_filter(
            get_init_log_density=partial(_gaussian_init, init_sd=init_sd),
            get_dynamics_log_density=partial(_gaussian_dynamics, tau=tau),
            get_observation_func=partial(_gaussian_observation, obs_sd=obs_sd),
            rtol=1e-7,
        )
        states = run_filter(filter_obj, model_inputs)
        return states.log_normalizing_constant[-1]

    value, grad = jax.value_and_grad(objective)(jnp.array([0.0, -2.0, -1.0]))

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))


def _flat_init(model_inputs, init_sd):
    def log_density(x):
        return norm.logpdf(x, 0.0, init_sd).sum()

    return log_density, jnp.zeros(4)


def _flat_random_walk_dynamics(state, model_inputs, tau):
    def log_density(x_prev, x):
        return norm.logpdf(x, x_prev, tau).sum()

    return log_density, jnp.zeros(4), jnp.zeros(4)


def _weighted_init(model_inputs, init_sd):
    weights = jnp.array([1.0, 1.01, 1.03, 1.07])

    def log_density(x):
        return norm.logpdf(x, 0.0, init_sd * weights).sum()

    return log_density, jnp.zeros(4)


def _weighted_random_walk_dynamics(state, model_inputs, tau):
    weights = jnp.array([1.0, 1.01, 1.03, 1.07])

    def log_density(x_prev, x):
        return norm.logpdf(x, x_prev, tau * weights).sum()

    return log_density, jnp.zeros(4), jnp.zeros(4)


def _team_init(model_inputs, init_sd):
    init_sd_repeated = jnp.tile(init_sd, 2)

    def log_density(x):
        return norm.logpdf(x, 0.0, init_sd_repeated).sum()

    return log_density, jnp.zeros(4)


def test_bivariate_poisson_observation_potential_derivatives_are_finite():
    """The model observation potential itself has finite parameter derivatives."""
    state = cast(
        taylor.LinearizedKalmanFilterState, ObservationState(mean=jnp.zeros(4))
    )
    model_inputs = ResultData(
        match_index=jnp.array(1),
        home_team_id=jnp.array(0),
        away_team_id=jnp.array(1),
        home_score=jnp.array(2),
        away_score=jnp.array(1),
        neutral=jnp.array(False),
        timestamp=jnp.array(10),
        home_timestamp_previous=jnp.array(0),
        away_timestamp_previous=jnp.array(0),
    )
    x = jnp.array([0.1, -0.2, 0.3, 0.05])

    def potential_value(params):
        log_potential, _ = model.get_observation_log_potential(
            state,
            model_inputs,
            alpha=params[0],
            beta=params[1],
            max_goals=8,
        )
        return log_potential(x)

    def potential_hessian(params):
        log_potential, _ = model.get_observation_log_potential(
            state,
            model_inputs,
            alpha=params[0],
            beta=params[1],
            max_goals=8,
        )
        return jax.hessian(log_potential)(x)

    params = jnp.array([-0.1, -4.0])

    assert jnp.all(jnp.isfinite(jax.grad(potential_value)(params)))
    assert jnp.all(jnp.isfinite(jax.jacrev(potential_hessian)(params)))


def _two_step_football_inputs():
    return ResultData(
        match_index=jnp.array([0, 1]),
        home_team_id=jnp.array([0, 0]),
        away_team_id=jnp.array([0, 1]),
        home_score=jnp.array([0, 2]),
        away_score=jnp.array([0, 1]),
        neutral=jnp.array([False, False]),
        timestamp=jnp.array([0, 10]),
        home_timestamp_previous=jnp.array([0, 0]),
        away_timestamp_previous=jnp.array([0, 0]),
    )


def test_factorial_taylor_filter_gradients_are_finite_with_tiny_anisotropy():
    """The scalar-parameter factorial path is differentiable with the opt-in ridge."""
    model_inputs = ResultData(
        match_index=jnp.array([0, 1]),
        home_team_id=jnp.array([0, 0]),
        away_team_id=jnp.array([0, 1]),
        home_score=jnp.array([0, 0]),
        away_score=jnp.array([0, 0]),
        neutral=jnp.array([False, False]),
        timestamp=jnp.array([0, 0]),
        home_timestamp_previous=jnp.array([0, 0]),
        away_timestamp_previous=jnp.array([0, 0]),
    )
    factorializer = build_factorializer(get_factorial_indices=model.get_factorial_inds)
    init_mean = jnp.array([0.0, 0.0])
    anisotropy_epsilon = 1e-2

    def objective(raw_params):
        init_sd, tau, kappa = jax.nn.softplus(raw_params[:3]) + 1e-6
        alpha, beta = raw_params[3], raw_params[4]
        filter_obj = taylor.build_filter(
            get_init_log_density=partial(
                model.get_init_log_density,
                init_mean=init_mean,
                init_chol_cov=jnp.diag(jnp.broadcast_to(init_sd, (2,))),
                num_teams=2,
                anisotropy_epsilon=anisotropy_epsilon,
            ),
            get_dynamics_log_density=partial(
                model.get_dynamics_log_density,
                tau=tau,
                init_mean=init_mean,
                kappa=kappa,
                anisotropy_epsilon=anisotropy_epsilon,
            ),
            get_observation_func=partial(
                model.get_observation_log_potential,
                alpha=alpha,
                beta=beta,
                max_goals=8,
                anisotropy_epsilon=anisotropy_epsilon,
                ridge_precision=1e-6,
            ),
            rtol=1e-7,
        )
        states = cast(
            Any,
            factorial_filter(
                filter_obj,
                factorializer,
                model_inputs,
                output_factorial=True,
            ),
        )
        return states.log_normalizing_constant[-1]

    raw_params = jnp.array([0.0, -4.6, -9.2, -0.1, -4.0])
    value, grad = jax.value_and_grad(objective)(raw_params)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))


def test_bivariate_poisson_taylor_filter_gradients_are_finite_with_distinct_scales():
    """The same observation model is differentiable when covariance scales differ."""
    model_inputs = _two_step_football_inputs()

    def objective(raw_params):
        init_sd, tau = jax.nn.softplus(raw_params[:2]) + 1e-6
        alpha, beta = raw_params[2], raw_params[3]
        filter_obj = taylor.build_filter(
            get_init_log_density=partial(_weighted_init, init_sd=init_sd),
            get_dynamics_log_density=partial(_weighted_random_walk_dynamics, tau=tau),
            get_observation_func=partial(
                model.get_observation_log_potential,
                alpha=alpha,
                beta=beta,
                max_goals=8,
            ),
            rtol=1e-7,
        )
        states = run_filter(filter_obj, model_inputs)
        return states.log_normalizing_constant[-1]

    value, grad = jax.value_and_grad(objective)(jnp.array([0.0, -4.6, -0.1, -4.0]))

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Attack/defence-specific scales are still repeated across the two teams "
        "in the local joint state."
    ),
)
def test_bivariate_poisson_taylor_filter_gradients_are_finite_with_attack_defence_scales():
    """Training attack and defence scales separately is not enough."""
    model_inputs = _two_step_football_inputs()
    init_mean = jnp.array([0.0, 0.0])

    def objective(raw_params):
        init_sd = jax.nn.softplus(raw_params[:2]) + 1e-6
        tau = jax.nn.softplus(raw_params[2:4]) + 1e-6
        kappa = jax.nn.softplus(raw_params[4:6]) + 1e-6
        alpha, beta = raw_params[6], raw_params[7]
        filter_obj = taylor.build_filter(
            get_init_log_density=partial(_team_init, init_sd=init_sd),
            get_dynamics_log_density=partial(
                model.get_dynamics_log_density,
                tau=tau,
                init_mean=init_mean,
                kappa=kappa,
            ),
            get_observation_func=partial(
                model.get_observation_log_potential,
                alpha=alpha,
                beta=beta,
                max_goals=8,
            ),
            rtol=1e-7,
        )
        states = run_filter(filter_obj, model_inputs)
        return states.log_normalizing_constant[-1]

    value, grad = jax.value_and_grad(objective)(
        jnp.array([0.0, 0.1, -4.6, -4.5, -9.2, -9.1, -0.1, -4.0])
    )

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Taylor filtering with repeated covariance eigenvalues has NaN gradients "
        "for prior/dynamics parameters."
    ),
)
def test_nonfactorial_taylor_filter_is_differentiable_for_bivariate_poisson_model():
    """The Taylor-filtered football observation path fails without factorial code."""
    model_inputs = _two_step_football_inputs()

    def objective(raw_params):
        init_sd, tau = jax.nn.softplus(raw_params[:2]) + 1e-6
        alpha, beta = raw_params[2], raw_params[3]
        filter_obj = taylor.build_filter(
            get_init_log_density=partial(_flat_init, init_sd=init_sd),
            get_dynamics_log_density=partial(_flat_random_walk_dynamics, tau=tau),
            get_observation_func=partial(
                model.get_observation_log_potential,
                alpha=alpha,
                beta=beta,
                max_goals=8,
            ),
            rtol=1e-7,
        )
        states = run_filter(filter_obj, model_inputs)
        return states.log_normalizing_constant[-1]

    value, grad = jax.value_and_grad(objective)(jnp.array([0.0, -4.6, -0.1, -4.0]))

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))
