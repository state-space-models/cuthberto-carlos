"""Define the model for football match scores."""

from typing import Any
from jax import Array, numpy as jnp
from jax.typing import ArrayLike
from jax.scipy.stats import norm
from cuthbertlib.types import LogConditionalDensity, LogDensity
from cuthbert.gaussian import taylor

from cuthberto_carlos.types import ResultData, DynamicsOnlyData
from cuthberto_carlos import bivariate_poisson


def _anisotropy_weights(shape: tuple[int, ...], epsilon: float) -> Array:
    size = 1
    for dim in shape:
        size *= dim
    return 1.0 + epsilon * jnp.arange(size).reshape(shape)


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
    init_cov: ArrayLike | None = None,
    init_chol_cov: ArrayLike | None = None,
    num_teams: int | None = None,
    anisotropy_epsilon: float = 0.0,
) -> tuple[LogDensity, Array]:
    """Get log p(x_0) and linearization point.

    Args:
        model_inputs: The match data, used to determine the number of teams.
            Not used.
        init_mean: The mean of the initial distribution for each state dimension.
        init_cov: The initial covariance matrix for attack and defence within a
            team. Shape (2, 2), or (num_teams, 2, 2) for team-specific covariances.
            Exactly one of init_cov and init_chol_cov must be provided.
        init_chol_cov: Lower Cholesky factor of the initial covariance matrix.
            Shape (2, 2), or (num_teams, 2, 2) for team-specific covariances.
            Exactly one of init_cov and init_chol_cov must be provided.
        num_teams: The number of teams, used to determine the shape of the
            linearization point.
        anisotropy_epsilon: Tiny deterministic multiplier used to break repeated
            covariance eigenvalues for differentiating through square-root linear
            algebra. Defaults to 0 for the exact symmetric model.

    Returns:
        A tuple containing log density function and linearization point
    """
    if (init_cov is None) == (init_chol_cov is None):
        raise ValueError("Exactly one of init_cov and init_chol_cov must be provided.")

    if init_chol_cov is None:
        base_chol_cov = jnp.linalg.cholesky(jnp.asarray(init_cov))
    else:
        base_chol_cov = jnp.asarray(init_chol_cov)

    def broadcast_chol_cov(x):
        if base_chol_cov.ndim == 2:
            return jnp.broadcast_to(base_chol_cov, x.shape[:-1] + (2, 2))
        return jnp.broadcast_to(base_chol_cov, x.shape[:-1] + (2, 2))

    def init_log_density(x):
        # x.shape = (2,) or (num_teams, 2)
        # init_mean.shape = (2,) or (num_teams, 2)
        x = jnp.asarray(x)
        is_single_team = x.ndim == 1
        x_teams = x[jnp.newaxis, :] if is_single_team else x
        init_mean_teams = jnp.broadcast_to(init_mean, x_teams.shape)
        chol_cov = broadcast_chol_cov(x_teams)
        weights = _anisotropy_weights(x_teams.shape, anisotropy_epsilon)
        chol_cov = weights[..., :, jnp.newaxis] * chol_cov

        centered = x_teams - init_mean_teams
        whitened = jnp.linalg.solve(chol_cov, centered[..., jnp.newaxis])[..., 0]
        quad = jnp.sum(whitened**2, axis=-1)
        log_det = jnp.sum(jnp.log(jnp.diagonal(chol_cov, axis1=-2, axis2=-1)), axis=-1)
        return jnp.sum(-0.5 * (2 * jnp.log(2 * jnp.pi) + quad) - log_det)

    linearization_point_shape = (num_teams, 2) if num_teams is not None else (2,)
    return init_log_density, jnp.zeros(linearization_point_shape)


def get_dynamics_log_density(
    state: taylor.LinearizedKalmanFilterState,
    model_inputs: ResultData | DynamicsOnlyData,
    tau: ArrayLike,
    init_mean: ArrayLike,
    kappa: ArrayLike,
    std_floor: ArrayLike = 1e-3,
    anisotropy_epsilon: float = 0.0,
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
        std_floor: Minimum dynamics standard deviation. This avoids singular
            zero elapsed-time transitions, including same-day matches.
        anisotropy_epsilon: Tiny deterministic multiplier used to break repeated
            covariance eigenvalues for differentiating through square-root linear
            algebra. Defaults to 0 for the exact symmetric model.

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
    weights = _anisotropy_weights((num_joined_factors * 2,), anisotropy_epsilon)
    tau_repeated = tau_repeated * weights
    kappa_repeated = kappa_repeated * weights

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
    std_floor = jnp.broadcast_to(std_floor, (2,))
    std_floor = jnp.tile(std_floor, num_joined_factors) * weights
    std_devs = jnp.sqrt(jnp.maximum(variance, std_floor**2))

    def dynamics_log_density(x_prev, x):
        mean = init_mean_repeated + phi * (x_prev - init_mean_repeated)
        return norm.logpdf(x, mean, std_devs).sum()

    linearization_point = jnp.zeros(num_joined_factors * 2)
    # Shape (2,) or (num_teams, 2) flattened to (2 * num_teams,)
    return dynamics_log_density, linearization_point, linearization_point


def get_observation_log_potential(
    state: taylor.LinearizedKalmanFilterState,
    model_inputs: ResultData,
    alpha: ArrayLike,
    beta: ArrayLike,
    max_goals: int = 8,
    friendly_ability_scale: ArrayLike = 1.0,
    competitive_ability_scale: ArrayLike = 1.0,
    anisotropy_epsilon: float = 0.0,
    ridge_precision: float = 0.0,
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
        friendly_ability_scale: Positive temperature applied to attack/defence
            contrasts for friendlies. Larger values reduce information from team
            strength differences in friendly observations.
        competitive_ability_scale: Positive temperature applied to attack/defence
            contrasts for non-friendly matches.
        anisotropy_epsilon: Tiny deterministic multiplier used with ridge_precision
            to break repeated covariance eigenvalues. Defaults to 0 for the exact
            symmetric model.
        ridge_precision: Tiny observation precision ridge added to stabilize
            differentiation through square-root linear algebra. Defaults to 0.

    Returns:
        A tuple containing log potential function and linearization point for x_t.
    """

    def log_potential(x: Array) -> Array:
        y = jnp.array([model_inputs.home_score, model_inputs.away_score])
        x_i = x[:2]
        x_j = x[2:]
        is_friendly = (
            jnp.array(False)
            if model_inputs.is_friendly is None
            else model_inputs.is_friendly
        )
        ability_scale = jnp.where(
            is_friendly,
            friendly_ability_scale,
            competitive_ability_scale,
        )
        loglik = bivariate_poisson.loglik(
            y,
            x_i,
            x_j,
            alpha,
            beta,
            max_goals,
            ability_scale=ability_scale,
        )
        if ridge_precision != 0.0:
            weights = _anisotropy_weights(x.shape, anisotropy_epsilon)
            ridge = 0.5 * ridge_precision * jnp.sum((weights * (x - state.mean)) ** 2)
            loglik = loglik - ridge
        return loglik

    return log_potential, state.mean


def get_observation_log_potential_noop(
    state: taylor.LinearizedKalmanFilterState,
    model_inputs: ResultData,
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
        return jnp.array(jnp.nan)

    return log_potential, jnp.full_like(state.mean, jnp.nan)
