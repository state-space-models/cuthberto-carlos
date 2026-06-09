"""Bivariate Poisson log likelihood utils."""

import functools
import jax
from jax import numpy as jnp, Array
from jax.typing import ArrayLike
from jax.scipy.special import gammaln, logsumexp


def _poisson_logpmf(y: ArrayLike, log_rate: ArrayLike) -> Array:
    y = jnp.asarray(y)
    log_rate = jnp.asarray(log_rate)
    rate = jnp.exp(log_rate)
    return y * log_rate - rate - gammaln(y + 1)


@functools.partial(jax.jit, static_argnames=("max_goals",))
def loglik_grid(
    x_i: ArrayLike,
    x_j: ArrayLike,
    alpha: float,
    beta: float,
    max_goals: int,
) -> Array:
    """Returns grid G where G[a, b] = log p(Y_i=a, Y_j=b).

    Args:
        x_i: shape (2,), [attack_i, defence_i]
        x_j: shape (2,), [attack_j, defence_j]
        alpha: scalar baseline scoring parameter
        beta: scalar shared-scoring/correlation parameter
        max_goals: largest score included in the grid

    Returns:
        loglik_grid: shape (max_goals + 1, max_goals + 1)
    """
    x_i = jnp.asarray(x_i)
    x_j = jnp.asarray(x_j)

    log_lambda1 = alpha + x_i[0] - x_j[1]
    log_lambda2 = alpha + x_j[0] - x_i[1]
    log_lambda3 = beta

    scores = jnp.arange(max_goals + 1)

    y_i = scores[:, None]  # shape (G, 1)
    y_j = scores[None, :]  # shape (1, G)

    k = scores[:, None, None]  # shape (G, 1, 1)

    valid = (k <= y_i) & (k <= y_j)

    # log P(U = y_i - k)
    log_p_u = _poisson_logpmf(y_i - k, log_lambda1)

    # log P(V = y_j - k)
    log_p_v = _poisson_logpmf(y_j - k, log_lambda2)

    # log P(W = k)
    log_p_w = _poisson_logpmf(k, log_lambda3)

    log_terms = log_p_u + log_p_v + log_p_w
    log_terms = jnp.where(valid, log_terms, -jnp.inf)

    return logsumexp(log_terms, axis=0)


def _log_binom(n: ArrayLike, k: ArrayLike) -> Array:
    n = jnp.asarray(n)
    k = jnp.asarray(k)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


@functools.partial(jax.jit, static_argnames=("max_goals",))
def loglik(
    y: ArrayLike,
    x_i: ArrayLike,
    x_j: ArrayLike,
    alpha: float,
    beta: float,
    max_goals: int,
) -> Array:
    """Log likelihood for the bivariate Poisson football-score model.

    Args:
        y: integer array with shape (2,), y[0] = goals for team i,
           y[1] = goals for team j.
        x_i: real array with shape (2,), x_i[0] = attack_i,
             x_i[1] = defence_i.
        x_j: real array with shape (2,), x_j[0] = attack_j,
             x_j[1] = defence_j.
        alpha: scalar baseline scoring parameter.
        beta: scalar covariance/shared-scoring parameter.
        max_goals: static upper bound for the finite sum. Must be >= min(y).

    Returns:
        Scalar log p(y | x_i, x_j, alpha, beta).
    """
    x_i = jnp.asarray(x_i)
    x_j = jnp.asarray(x_j)
    y = jnp.asarray(y)
    y_i = y[0]
    y_j = y[1]

    # Log rates
    log_lambda1 = alpha + x_i[0] - x_j[1]
    log_lambda2 = alpha + x_j[0] - x_i[1]
    log_lambda3 = beta

    lambda1 = jnp.exp(log_lambda1)
    lambda2 = jnp.exp(log_lambda2)
    lambda3 = jnp.exp(log_lambda3)

    # Base independent-Poisson-looking part
    base = (
        -(lambda1 + lambda2 + lambda3)
        + y_i * log_lambda1
        - gammaln(y_i + 1)
        + y_j * log_lambda2
        - gammaln(y_j + 1)
    )

    # Correlation/shared component:
    # sum_{k=0}^{min(y_i, y_j)}
    #   C(y_i,k) C(y_j,k) k! (lambda3 / (lambda1 lambda2))^k
    k = jnp.arange(max_goals + 1)
    min_y = jnp.minimum(y_i, y_j)

    log_terms = (
        _log_binom(y_i, k)
        + _log_binom(y_j, k)
        + gammaln(k + 1)
        + k * (log_lambda3 - log_lambda1 - log_lambda2)
    )

    log_terms = jnp.where(k <= min_y, log_terms, -jnp.inf)

    # Optional guard: return -inf if max_goals is too small.
    log_sum = jnp.where(
        min_y <= max_goals,
        logsumexp(log_terms),
        -jnp.inf,
    )

    return base + log_sum
