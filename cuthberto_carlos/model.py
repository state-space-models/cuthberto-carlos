"""Define the model for football match scores."""

from typing import Any

from jax import Array, numpy as jnp
from jax.typing import ArrayLike
from jax.scipy.stats import norm
from cuthbertlib.types import LogConditionalDensity, LogDensity
from cuthbert.gaussian import taylor

from cuthberto_carlos.types import ResultData, DynamicsOnlyData


def get_init_log_density(
    model_inputs: Any,
    init_mean: ArrayLike,
    init_sd: ArrayLike,
    num_teams: int | None,
) -> tuple[LogDensity, Array]:
    """Get log p(x_0).

    x_0 is of shape (2,) or (num_teams, 2), where the 2 state dimensions correspond to
    strength of defence and attack.

    Args:
        model_inputs: The match data, used to determine the number of teams.
            Not used.
        init_mean: The mean of the initial distribution for each state dimension.
        init_sd: The standard deviation of the initial distribution for each state dimension.
        num_teams: The number of teams in the model.
            Integer or None. If None, the linearization point is of shape (2,),
            otherwise  (num_teams, 2).

    Returns:
        A tuple containing log density function and linearization point
    """

    def init_log_density(x):
        # x.shape = (2,) or (num_teams, 2)
        # init_mean.shape = (2,), init_sd.shape = (2,)
        return norm.logpdf(x, init_mean, init_sd).sum()

    linearization_point_shape = (2,) if num_teams is None else (num_teams, 2)

    return init_log_density, jnp.zeros(linearization_point_shape)


def get_dynamics_log_density(
    state: taylor.LinearizedKalmanFilterState,
    model_inputs: ResultData | DynamicsOnlyData,
    tau: ArrayLike,
) -> tuple[LogConditionalDensity, Array, Array]:
    """Get log p(x_t | x_{t-1}) for a single factor or vector of stacked factors.

    We have Brownian dynamics:
    p(x_t | x_{t-1}) = N(x_t | x_{t-1}, tau^2 * (timestamp_t - timestamp_{t-1})).

    Args:
        state: The current state of the Kalman filter.
            Only used to determine the number of factors to be processed.
        model_inputs: The match data, used to determine the time between matches.
        tau: Brownian standard deviation per day.
            Shape (,), (1,) or (2,) for defence and attack.

    Returns:
        A tuple containing log density function, linearization point for x_{t-1}, and
        linearization point for x_t.
        Linearization points are set to zeros since the dynamics are linear and
            Gaussian so cuthbert.taylor.gaussian linearization is exact.
    """
    # TODO: Generalise to arbitrary state dim rather than hardcoding 2?
    num_joined_factors = state.mean.shape[0] // 2  # Can be 1
    tau = jnp.broadcast_to(tau, (2,))

    # x_prev.shape = x.shape = (2 * num_joined_factors,)
    # Repeat tau num_joined_factors times
    tau_repeated = jnp.tile(tau, num_joined_factors)
    assert model_inputs.timestamp_previous is not None, (
        "model_inputs.timestamp_previous is required for dynamics log density"
    )

    # timestamp.shape = timestamp_previous.shape = (,) or (1,) or (num_joined_factors,)
    # process into guaranteed shape (2 * num_joined_factors,)
    def process_timestamp(t):
        t = jnp.broadcast_to(t, (num_joined_factors,))
        return jnp.repeat(t, 2)  # Repeat for defence and attack

    timestamp = process_timestamp(model_inputs.timestamp)
    timestamp_previous = process_timestamp(model_inputs.timestamp_previous)

    std_devs = tau_repeated * jnp.sqrt(timestamp - timestamp_previous) + 1e-8
    # Add small nugget to avoid numerical issues when timestamp = timestamp_previous

    def dynamics_log_density(x_prev, x):
        return norm.logpdf(x, x_prev, std_devs).sum()

    linearization_point = jnp.zeros(num_joined_factors * 2)
    # Shape (2,) or (num_teams, 2) flattened to (2 * num_teams,)
    return dynamics_log_density, linearization_point, linearization_point


def get_observation_func(
    state: taylor.LinearizedKalmanFilterState, model_inputs: ResultData
) -> tuple[taylor.LogPotential, Array]:
    """Get log p(y_t | x_t) as a function of x_t for Bivariate Poisson."""

    def log_potential(
        x,
    ): ...  # TODO: Bivariate Poisson log potential from https://github.com/SamDuffield/abile/blob/main/abile/models/bivariate_poisson/extended_kalman.py

    return log_potential, state.mean
