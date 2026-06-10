from jax import numpy as jnp

from cuthberto_carlos import bivariate_poisson


def test_bivariate_poisson_loglik_grid_matches_scalar_loglik() -> None:
    max_goals = 4
    x_i = jnp.array([0.2, -0.1])
    x_j = jnp.array([-0.3, 0.4])
    alpha = -0.2
    beta = -1.1

    grid = bivariate_poisson.loglik_grid(x_i, x_j, alpha, beta, max_goals)

    for y_i in range(max_goals + 1):
        for y_j in range(max_goals + 1):
            scalar = bivariate_poisson.loglik(
                jnp.array([y_i, y_j]),
                x_i,
                x_j,
                alpha,
                beta,
                max_goals,
            )
            assert jnp.allclose(grid[y_i, y_j], scalar, rtol=1e-6, atol=1e-6)


def test_bivariate_poisson_loglik_matches_simple_expected_values() -> None:
    max_goals = 2
    x_i = jnp.array([0.0, 0.0])
    x_j = jnp.array([0.0, 0.0])
    alpha = 0.0
    beta = 0.0

    expected = jnp.array(
        [
            [-3.0, -3.0, -3.0 - jnp.log(2.0)],
            [-3.0, -3.0 + jnp.log(2.0), -3.0 + jnp.log(1.5)],
            [-3.0 - jnp.log(2.0), -3.0 + jnp.log(1.5), -3.0 + jnp.log(1.75)],
        ]
    )

    grid = bivariate_poisson.loglik_grid(x_i, x_j, alpha, beta, max_goals)
    assert jnp.allclose(grid, expected, rtol=1e-6, atol=1e-6)

    for y_i in range(max_goals + 1):
        for y_j in range(max_goals + 1):
            scalar = bivariate_poisson.loglik(
                jnp.array([y_i, y_j]),
                x_i,
                x_j,
                alpha,
                beta,
                max_goals,
            )
            assert jnp.allclose(scalar, expected[y_i, y_j], rtol=1e-6, atol=1e-6)


def test_ability_scale_tempers_attack_defence_contrast() -> None:
    max_goals = 4
    x_i = jnp.array([0.8, -0.2])
    x_j = jnp.array([-0.4, 0.6])
    alpha = -0.1
    beta = -1.2
    ability_scale = 2.5
    y = jnp.array([2, 1])

    scaled_state_loglik = bivariate_poisson.loglik(
        y,
        x_i / ability_scale,
        x_j / ability_scale,
        alpha,
        beta,
        max_goals,
    )
    tempered_loglik = bivariate_poisson.loglik(
        y,
        x_i,
        x_j,
        alpha,
        beta,
        max_goals,
        ability_scale=ability_scale,
    )
    tempered_grid = bivariate_poisson.loglik_grid(
        x_i,
        x_j,
        alpha,
        beta,
        max_goals,
        ability_scale=ability_scale,
    )

    assert jnp.allclose(tempered_loglik, scaled_state_loglik, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(tempered_grid[y[0], y[1]], tempered_loglik)
