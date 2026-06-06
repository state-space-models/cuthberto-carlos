"""Define the model for football match scores."""

from jax import Array, numpy as jnp
from jax.typing import ArrayLike
from jax.scipy.stats import norm
from cuthbertlib.types import LogConditionalDensity, LogDensity
from cuthbert.gaussian import taylor

from cuthberto_carlos.types import ResultData


# def build_taylor_model(init_sd: float) -> tuple:
#     return (
#         partial(get_init_log_density, init_sd=init_sd),
#         get_dynamics_log_density,
#         get_observation_func,
#     )


def get_init_log_density(
    model_inputs: ResultData, init_mean: ArrayLike, init_sd: ArrayLike, num_teams: int
) -> tuple[LogDensity, Array]:
    """Get log p(x_0).

    x_0 is of shape (num_teams, 2), where the 2 state dimensions correspond to
    strength of defence and attack.

    Args:
        model_inputs: The match data, used to determine the number of teams.
            Not used.
        init_mean: The mean of the initial distribution for each state dimension.
        init_sd: The standard deviation of the initial distribution for each state dimension.
        num_teams: The number of teams in the model.

    Returns:
        A tuple containing log density function and linearization point
    """
    init_mean = jnp.broadcast_to(init_mean, (2,))
    init_sd = jnp.broadcast_to(init_sd, (2,))

    def init_log_density(x):
        # x.shape = (num_teams, 2), init_mean.shape = (2,), init_sd.shape = (2,)
        return norm.logpdf(x, init_mean, init_sd).sum()

    return init_log_density, jnp.zeros((num_teams, 2))


def get_dynamics_log_density_single_factor(
    state: taylor.LinearizedKalmanFilterState, model_inputs: ResultData, tau: ArrayLike
) -> tuple[LogConditionalDensity, Array, Array]:
    """Get log p(x_t | x_{t-1}) for a single factor.

    Args:
        state: The current state of the Kalman filter.
            Not used.
        model_inputs: The match data, used to determine the time between matches.
            Not used.
        tau: Brownian standard deviation per day.
            Shape (2,) for defence and attack independently.

    Returns:
        A tuple containing log density function, linearization point for x_{t-1}, and
        linearization point for x_t.
        Linearization points are set to zeros since the dynamics are linear and
            Gaussian so cuthbert.taylor.gaussian linearization is exact.
    """

    def dynamics_log_density(x_prev, x):
        # x_prev.shape = x.shape = (2,)
        # tau.shape = (2,)
        return norm.logpdf(
            x,
            x_prev,
            jnp.sqrt(
                (tau**2) * (model_inputs.timestamp_days - model_inputs.timestamp_days)
            )
            + 1e-8,  # Add small nugget to avoid numerical issues when x = x_prev
        ).sum()

    return dynamics_log_density, jnp.zeros(2), jnp.zeros(2)


def get_dynamics_log_density_joint_factor(
    state: taylor.LinearizedKalmanFilterState,
    model_inputs: ResultData,
    tau: ArrayLike,
) -> tuple[LogConditionalDensity, Array, Array]:
    """Get log p(x_t | x_{t-1}) for a joint stacked factor.

    Args:
        state: The current state of the Kalman filter.
            Not used.
        model_inputs: The match data, used to determine the time between matches.
            Not used.
        tau: Brownian standard deviation per day.
            Shape (2 * num_joined_factors,) for factors, defence and attack all
            independently.

    Returns:
        A tuple containing log density function, linearization point for x_{t-1}, and
        linearization point for x_t.
        Linearization points are set to zeros since the dynamics are linear and
            Gaussian so cuthbert.taylor.gaussian linearization is exact.
    """
    tau = jnp.broadcast_to(tau, (2,))
    num_joined_factors = state.mean.shape[0] // 2

    def dynamics_log_density(x_prev, x):
        # x_prev.shape = x.shape = (2 * num_joined_factors,)
        # Repeat tau num_joined_factors times
        tau_repeated = jnp.tile(tau, num_joined_factors)
        return norm.logpdf(
            x,
            x_prev,
            jnp.sqrt(
                (tau_repeated**2)
                * (model_inputs.timestamp_days - model_inputs.timestamp_days)
            )
            + 1e-8,  # Add small nugget to avoid numerical issues when x = x_prev
        ).sum()

    return (
        dynamics_log_density,
        jnp.zeros(num_joined_factors * 2),
        jnp.zeros(num_joined_factors * 2),
    )


def get_observation_func(
    state: taylor.LinearizedKalmanFilterState, model_inputs: ResultData
) -> tuple[taylor.LogPotential, Array]:
    """Get log p(y_t | x_t) as a function of x_t for Bivariate Poisson."""

    def log_potential(
        x,
    ): ...  # TODO: Bivariate Poisson log potential from https://github.com/SamDuffield/abile/blob/main/abile/models/bivariate_poisson/extended_kalman.py

    return log_potential, state.mean
