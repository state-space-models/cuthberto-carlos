"""Moment-based Gaussian approximation for football match scores."""

from typing import Any

from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

from cuthbert.gaussian.types import LinearizedKalmanFilterState
from cuthbertlib.linearize.moments import MeanAndCholCovFunc

from cuthberto_carlos.data_types import DynamicsOnlyData, ResultData


def get_factorial_inds(model_inputs: ResultData | DynamicsOnlyData) -> Array:
    """Get indices of the factors corresponding to the match score.

    Args:
        model_inputs: The match data, used to extract the team indices.

    Returns:
        An array of either shape (2,) for two teams involved in a match
            or shape (,) for a single team in the case of DynamicsOnlyData.
    """
    if isinstance(model_inputs, ResultData):
        return jnp.array([model_inputs.home_team_id, model_inputs.away_team_id])
    elif isinstance(model_inputs, DynamicsOnlyData):
        return jnp.array(model_inputs.team_id)
    raise TypeError(f"Unsupported model_inputs type: {type(model_inputs)!r}")


def get_init_params(
    model_inputs: Any,
    init_mean: ArrayLike,
    init_chol_cov: ArrayLike,
    num_teams: int | None = None,
) -> tuple[Array, Array]:
    """Get initial Gaussian mean and Cholesky covariance.

    Args:
        model_inputs: The match data. Not used.
        init_mean: The mean of the initial distribution for each state dimension.
        init_chol_cov: The Cholesky covariance matrix of the initial distribution.
            Shape (2, 2), allowing correlation between attack and defence.
        num_teams: The number of teams, used to determine the shape of the initial
            state.

    Returns:
        A tuple containing initial mean and Cholesky covariance.
    """
    init_mean = jnp.asarray(init_mean)
    init_chol_cov = jnp.asarray(init_chol_cov)
    if num_teams is None:
        return init_mean, init_chol_cov

    return (
        jnp.broadcast_to(init_mean, (num_teams, 2)),
        jnp.broadcast_to(init_chol_cov, (num_teams, 2, 2)),
    )


def _process_timestamp(t: ArrayLike, num_joined_factors: int) -> Array:
    t = jnp.broadcast_to(t, (num_joined_factors,))
    return jnp.repeat(t, 2)  # Repeat for attack and defence.


def get_dynamics_params(
    state: LinearizedKalmanFilterState,
    model_inputs: ResultData | DynamicsOnlyData,
    init_mean: ArrayLike,
    init_chol_cov: ArrayLike,
    kappa: ArrayLike,
) -> tuple[MeanAndCholCovFunc, Array]:
    """Get conditional moments for the OU state dynamics.

    We have Ornstein-Uhlenbeck dynamics:
    x_t | x_{t-1} ~ N(mu + phi * (x_{t-1} - mu), Q_t),
    where phi = exp(-kappa * (timestamp_t - timestamp_{t-1})) and Q_t is chosen
    so that the long-term marginal distribution is N(init_mean, init_cov).

    Args:
        state: The current state of the Kalman filter.
            Only used to determine the number of joined factors.
        model_inputs: The match data, used to determine the time between matches.
        init_mean: Long-run mean of the OU dynamics.
            Shape (2,) for attack and defence.
        init_chol_cov: Cholesky factor of the long-run covariance of the OU dynamics.
            Shape (2, 2) for attack and defence.
        kappa: Mean-reversion rate per day.
            Shape (,), (1,) or (2,) for attack and defence.

    Returns:
        A tuple containing a function that maps x_{t-1} to the conditional mean and
        Cholesky covariance of x_t, and a linearization point for x_{t-1}.
    """
    num_joined_factors = state.mean.shape[0] // 2  # Can be 1
    init_mean = jnp.broadcast_to(init_mean, (2,))
    init_chol_cov = jnp.asarray(init_chol_cov)
    kappa = jnp.broadcast_to(kappa, (2,))

    init_mean_repeated = jnp.tile(init_mean, num_joined_factors)
    kappa_repeated = jnp.tile(kappa, num_joined_factors)

    if isinstance(model_inputs, ResultData):
        assert (
            model_inputs.home_timestamp_previous is not None
            and model_inputs.away_timestamp_previous is not None
        ), (
            "model_inputs.home_timestamp_previous and "
            "model_inputs.away_timestamp_previous are required for dynamics moments"
        )
        timestamp_previous = jnp.array(
            [
                model_inputs.home_timestamp_previous,
                model_inputs.away_timestamp_previous,
            ]
        )
    else:
        timestamp_previous = model_inputs.timestamp_previous

    timestamp = _process_timestamp(model_inputs.timestamp, num_joined_factors)
    timestamp_previous = _process_timestamp(timestamp_previous, num_joined_factors)

    elapsed_days = jnp.maximum(timestamp - timestamp_previous, 0)
    phi = jnp.exp(-kappa_repeated * elapsed_days)
    init_cov = init_chol_cov @ init_chol_cov.T
    stationary_cov = jnp.kron(jnp.eye(num_joined_factors), init_cov)
    transition_cov = stationary_cov * (1.0 - phi[:, None] * phi[None, :])
    jitter = 1e-8
    chol_cov = jnp.linalg.cholesky(
        transition_cov + jitter * jnp.eye(transition_cov.shape[0])
    )

    def dynamics_mean_and_chol_cov(x_prev: ArrayLike) -> tuple[Array, Array]:
        x_prev = jnp.asarray(x_prev)
        mean = init_mean_repeated + phi * (x_prev - init_mean_repeated)
        return mean, chol_cov

    linearization_point = jnp.zeros(num_joined_factors * 2)
    return dynamics_mean_and_chol_cov, linearization_point


def _bivariate_poisson_mean_and_chol_cov(
    x_i: ArrayLike,
    x_j: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
    scale: ArrayLike,
) -> tuple[Array, Array]:
    """Get exact mean and Cholesky covariance for the bivariate Poisson model."""
    x_i = jnp.asarray(x_i)
    x_j = jnp.asarray(x_j)

    lambda_1 = jnp.exp(alpha + (x_i[0] - x_j[1]) / scale)
    lambda_2 = jnp.exp(alpha + (x_j[0] - x_i[1]) / scale)
    lambda_3 = jnp.exp(beta)

    mean = jnp.array([lambda_1 + lambda_3, lambda_2 + lambda_3])
    cov = jnp.array(
        [
            [lambda_1 + lambda_3, lambda_3],
            [lambda_3, lambda_2 + lambda_3],
        ]
    )
    return mean, jnp.linalg.cholesky(cov)


def get_observation_params(
    state: LinearizedKalmanFilterState,
    model_inputs: ResultData,
    alpha: ArrayLike,
    beta: ArrayLike,
    friendly_scale: ArrayLike,
) -> tuple[MeanAndCholCovFunc, Array, Array]:
    """Get conditional moments for football scores under a bivariate Poisson model.

    Args:
        state: The current state of the Kalman filter.
            The observation moments are linearized around ``state.mean``.
        model_inputs: The match data, used to extract the match score.
        alpha: Scalar baseline scoring parameter.
        beta: Scalar covariance/shared-scoring parameter.
        friendly_scale: Scalar parameter that controls the influence of the team strength
            parameters on the expected score for friendly matches. Larger values correspond
            to less influence (and thus higher variance in the observation model).

    Returns:
        A tuple containing a function that maps x_t to the conditional mean and
        Cholesky covariance of y_t, a linearization point for x_t, and the observed
        score vector.
    """
    scale = jnp.where(model_inputs.friendly, friendly_scale, 1.0)

    def observation_mean_and_chol_cov(x: ArrayLike) -> tuple[Array, Array]:
        x = jnp.asarray(x)
        x_i = x[:2]
        x_j = x[2:]
        return _bivariate_poisson_mean_and_chol_cov(x_i, x_j, alpha, beta, scale)

    y = jnp.array([model_inputs.home_score, model_inputs.away_score], dtype=jnp.float32)
    return observation_mean_and_chol_cov, state.mean, y


def get_observation_params_noop(
    state: LinearizedKalmanFilterState,
    model_inputs: ResultData | DynamicsOnlyData,
) -> tuple[MeanAndCholCovFunc, Array, Array]:
    """Get a zero-information observation for dynamics-only filtering steps."""
    del model_inputs

    def observation_mean_and_chol_cov(x: ArrayLike) -> tuple[Array, Array]:
        dtype = jnp.asarray(x).dtype
        return jnp.zeros(1, dtype=dtype), jnp.ones((1, 1), dtype=dtype)

    return observation_mean_and_chol_cov, state.mean, jnp.array([jnp.nan])
