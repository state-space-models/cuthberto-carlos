"""Define the model for football match scores."""

from typing import Any
from jax import Array, numpy as jnp
from jax.typing import ArrayLike
from jax.scipy.stats import norm
from cuthbertlib.types import LogConditionalDensity, LogDensity
from cuthbert.gaussian import taylor

from cuthberto_carlos.types import ResultData, DynamicsOnlyData
from cuthberto_carlos import bivariate_poisson


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


def get_init_log_density(
    model_inputs: Any,
    init_mean: ArrayLike,
    init_sd: ArrayLike,
    num_teams: int | None = None,
) -> tuple[LogDensity, Array]:
    """Get log p(x_0) and linearization point.

    Args:
        model_inputs: The match data, used to determine the number of teams.
            Not used.
        init_mean: The mean of the initial distribution for each state dimension.
        init_sd: The standard deviation of the initial distribution for each state
            dimension.
        num_teams: The number of teams, used to determine the shape of the
            linearization point.

    Returns:
        A tuple containing log density function and linearization point
    """

    def init_log_density(x):
        # x.shape = (2,) or (num_teams, 2)
        # init_mean.shape = (2,), init_sd.shape = (2,)
        return norm.logpdf(x, init_mean, init_sd).sum()

    linearization_point_shape = (num_teams, 2) if num_teams is not None else (2,)
    return init_log_density, jnp.zeros(linearization_point_shape)


def get_dynamics_log_density(
    state: taylor.LinearizedKalmanFilterState,
    model_inputs: ResultData | DynamicsOnlyData,
    tau: ArrayLike,
    init_mean: ArrayLike,
    kappa: ArrayLike,
) -> tuple[LogConditionalDensity, Array, Array]:
    """Get log p(x_t | x_{t-1}) for a single factor or vector of stacked factors.

    We have Ornstein-Uhlenbeck dynamics:
    p(x_t | x_{t-1}) = N(mu + phi * (x_{t-1} - mu),
    tau^2 * (1 - phi^2) / (2 * kappa)),
    where phi = exp(-kappa * (timestamp_t - timestamp_{t-1})).
    When kappa = 0, this recovers Brownian dynamics with variance
    tau^2 * (timestamp_t - timestamp_{t-1}).

    Args:
        state: The current state of the Kalman filter.
            Only used to determine the number of factors to be processed.
        model_inputs: The match data, used to determine the time between matches.
        tau: Diffusion standard deviation per sqrt day.
            Shape (,), (1,) or (2,) for attack and defence.
        init_mean: Long-run mean of the OU dynamics.
            Shape (2,) for attack and defence.
        kappa: Mean-reversion rate per day. kappa = 0 gives Brownian dynamics.
            Shape (,), (1,) or (2,) for attack and defence.

    Returns:
        A tuple containing log density function, linearization point for x_{t-1}, and
        linearization point for x_t.
        Linearization points are set to zeros since the dynamics are linear and
            Gaussian so cuthbert.taylor.gaussian linearization is exact.
    """
    # TODO: Generalise to arbitrary state dim rather than hardcoding 2?
    num_joined_factors = state.mean.shape[0] // 2  # Can be 1
    tau = jnp.broadcast_to(tau, (2,))
    init_mean = jnp.broadcast_to(init_mean, (2,))
    kappa = jnp.broadcast_to(kappa, (2,))

    # x_prev.shape = x.shape = (2 * num_joined_factors,)
    # Repeat parameters num_joined_factors times
    tau_repeated = jnp.tile(tau, num_joined_factors)
    init_mean_repeated = jnp.tile(init_mean, num_joined_factors)
    kappa_repeated = jnp.tile(kappa, num_joined_factors)

    def process_timestamp(t):
        t = jnp.broadcast_to(t, (num_joined_factors,))
        return jnp.repeat(t, 2)  # Repeat for attack and defence.

    if isinstance(model_inputs, ResultData):
        assert (
            model_inputs.home_timestamp_previous is not None
            and model_inputs.away_timestamp_previous is not None
        ), (
            "model_inputs.home_timestamp_previous and "
            "model_inputs.away_timestamp_previous are required for dynamics log density"
        )
        timestamp_previous = jnp.array(
            [
                model_inputs.home_timestamp_previous,
                model_inputs.away_timestamp_previous,
            ]
        )
    else:
        timestamp_previous = model_inputs.timestamp_previous

    timestamp = process_timestamp(model_inputs.timestamp)
    timestamp_previous = process_timestamp(timestamp_previous)

    elapsed_days = jnp.maximum(timestamp - timestamp_previous, 0)
    phi = jnp.exp(-kappa_repeated * elapsed_days)
    safe_kappa = jnp.where(kappa_repeated > 0, kappa_repeated, 1.0)
    one_minus_phi_squared = -jnp.expm1(-2.0 * kappa_repeated * elapsed_days)
    ou_variance = tau_repeated**2 * one_minus_phi_squared / (2.0 * safe_kappa)
    brownian_variance = tau_repeated**2 * elapsed_days
    variance = jnp.where(kappa_repeated > 0, ou_variance, brownian_variance)
    std_devs = jnp.sqrt(variance)
    std_floor = 1e-3
    std_devs = jnp.where(std_devs > std_floor, std_devs, std_floor)

    def dynamics_log_density(x_prev, x):
        mean = init_mean_repeated + phi * (x_prev - init_mean_repeated)
        return norm.logpdf(x, mean, std_devs).sum()

    linearization_point = jnp.zeros(num_joined_factors * 2)
    # Shape (2,) or (num_teams, 2) flattened to (2 * num_teams,)
    return dynamics_log_density, linearization_point, linearization_point


def get_observation_log_potential(
    state: taylor.LinearizedKalmanFilterState,
    model_inputs: ResultData,
    alpha: float,
    beta: float,
    max_goals: int = 8,
) -> tuple[taylor.LogPotential, Array]:
    """Get log p(y_t | x_t) as a function of x_t for Bivariate Poisson.

    Args:
        state: The current state of the Kalman filter.
            Only used to determine the number of factors to be processed.
        model_inputs: The match data, used to extract the match score.
        alpha: Scalar baseline scoring parameter.
        beta: Scalar covariance/shared-scoring parameter.
        max_goals: Static upper bound for the finite sum in the bivariate Poisson
            log likelihood. Must be >= min(home_score, away_score).

    Returns:
        A tuple containing log potential function and linearization point for x_t.
    """

    def log_potential(x: Array) -> Array:
        y = jnp.array([model_inputs.home_score, model_inputs.away_score])
        x_i = x[:2]
        x_j = x[2:]
        loglik = bivariate_poisson.loglik(y, x_i, x_j, alpha, beta, max_goals)
        return loglik

    return log_potential, state.mean
