from jax import numpy as jnp, vmap
from cuthbertlib.linearize import linearize_log_density

from cuthberto_carlos.model import get_init_log_density


def test_init():
    init_mean = jnp.array([1.2, 3.2])
    init_sd = jnp.array([0.3, 1.1])
    num_teams = 5

    init_log_density, lin_point = get_init_log_density(
        None, init_mean, init_sd, num_teams
    )

    init_log_density_noneteams, lin_point_noneteams = get_init_log_density(
        None, init_mean, init_sd, None
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

    assert jnp.array_equal(jnp.broadcast_to(init_mean, (num_teams, 2)), m0)
    assert jnp.array_equal(
        jnp.broadcast_to(jnp.diag(init_sd), (num_teams, 2, 2)), chol_P0
    )

    assert jnp.array_equal(init_mean, m0_noneteams)
    assert jnp.array_equal(jnp.diag(init_sd), chol_P0_noneteams)

    # TODO: Replace this with checking taylor.non_associative_filter.init_prepare
