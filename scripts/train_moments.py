"""Train the moment-based Gaussian filter parameters."""

from functools import partial
import json
from pathlib import Path
import sys
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jnp
import optax
import pandas as pd
import plotnine as pn
from cuthbert.factorial import filter as factorial_filter
from cuthbert.factorial.gaussian import build_factorializer
from cuthbert.gaussian import moments

from cuthberto_carlos import model_moments
from cuthberto_carlos.data import download_data


# Training configuration.
STEPS = 200
LEARNING_RATE = 1e-1
LOG_EVERY = 10
MAX_GOALS = 8
MAX_MATCHES = None  # Set to a small integer, e.g. 100, for quick smoke runs.

# Initial parameter values on the constrained scale.
# INITIAL_KAPPA = 1e-4
# INITIAL_FRIENDLY_SCALE = 1.0
# INITIAL_ALPHA = 0.2
# INITIAL_BETA = -2.0
# INITIAL_INIT_SD = 1.0
# INITIAL_INIT_CORR = 0.0
# INIT_MEAN = jnp.array([0.0, 0.0])

INITIAL_KAPPA = 1e-2
INITIAL_FRIENDLY_SCALE = 2.0
INITIAL_ALPHA = 0.2
INITIAL_BETA = -4.0
INITIAL_INIT_SD = 1.0
INITIAL_INIT_CORR = 0.0
INIT_MEAN = jnp.array([0.0, 0.0])

OUTPUT_DIR = Path("outputs")
PARAMS_JSON_PATH = OUTPUT_DIR / "moments_params.json"
LOSS_PLOT_PATH = OUTPUT_DIR / "moments_loss_curve.png"


def positive(raw: jax.Array, floor: float = 1e-6) -> jax.Array:
    """Map an unconstrained scalar to a positive scalar."""
    return jax.nn.softplus(raw) + floor


def inverse_positive(value: float, floor: float = 1e-6) -> jax.Array:
    """Map a positive scalar back to the unconstrained softplus scale."""
    return jnp.log(jnp.expm1(jnp.asarray(value - floor)))


def constrain_params(raw_params: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Transform unconstrained optimizer parameters to model parameters."""
    init_sd = positive(raw_params["init_sd"])
    init_corr = 0.99 * jnp.tanh(raw_params["init_corr"])
    init_cov = (init_sd**2) * jnp.array(
        [
            [1.0, init_corr],
            [init_corr, 1.0],
        ]
    )
    return {
        "kappa": positive(raw_params["kappa"]),
        "friendly_scale": positive(raw_params["friendly_scale"]),
        "alpha": raw_params["alpha"],
        "beta": raw_params["beta"],
        "init_sd": init_sd,
        "init_corr": init_corr,
        "init_cov": init_cov,
        "init_chol_cov": jnp.linalg.cholesky(init_cov),
    }


def params_to_jsonable(params: dict[str, jax.Array]) -> dict[str, float | list]:
    """Convert constrained JAX params to JSON-serializable values."""
    return {
        "kappa": float(params["kappa"]),
        "friendly_scale": float(params["friendly_scale"]),
        "alpha": float(params["alpha"]),
        "beta": float(params["beta"]),
        "init_chol_cov": jnp.asarray(params["init_chol_cov"]).tolist(),
        "init_mean": jnp.asarray(INIT_MEAN).tolist(),
    }


def format_params(params: dict[str, jax.Array]) -> str:
    """Format constrained scalar parameters for logging."""
    return (
        f"kappa={float(params['kappa']):.6g}, "
        f"friendly_scale={float(params['friendly_scale']):.6g}, "
        f"alpha={float(params['alpha']):.6g}, "
        f"beta={float(params['beta']):.6g}, "
        f"init_sd={float(params['init_sd']):.6g}, "
        f"init_corr={float(params['init_corr']):.6g}"
    )


def add_dummy_initial_input(model_inputs):
    """Add a dummy model input at index 0 for Cuthbert factorial filtering."""
    return jax.tree.map(
        lambda x: jnp.concatenate([jnp.zeros_like(x[:1]), x], axis=0),
        model_inputs,
    )


def maybe_truncate_model_inputs(model_inputs):
    """Optionally keep only the first ``MAX_MATCHES`` matches."""
    if MAX_MATCHES is None:
        return model_inputs
    return jax.tree.map(lambda x: x[:MAX_MATCHES], model_inputs)


pd_data, jax_data, teams_id_to_name_dict, _ = download_data(max_goals=MAX_GOALS)
jax_data = maybe_truncate_model_inputs(jax_data)
model_inputs = add_dummy_initial_input(jax_data)
num_teams = len(teams_id_to_name_dict)


raw_params = {
    "kappa": inverse_positive(INITIAL_KAPPA),
    "friendly_scale": inverse_positive(INITIAL_FRIENDLY_SCALE),
    "alpha": jnp.asarray(INITIAL_ALPHA),
    "beta": jnp.asarray(INITIAL_BETA),
    "init_sd": inverse_positive(INITIAL_INIT_SD),
    "init_corr": jnp.arctanh(jnp.asarray(INITIAL_INIT_CORR) / 0.99),
}

factorializer = build_factorializer(
    get_factorial_indices=model_moments.get_factorial_inds
)


def log_normalizing_constant(raw_params: dict[str, jax.Array]) -> jax.Array:
    """Run the filter and return the final log normalizing constant."""
    params = constrain_params(raw_params)
    filter_obj = moments.build_filter(
        get_init_params=partial(
            model_moments.get_init_params,
            init_mean=INIT_MEAN,
            init_chol_cov=params["init_chol_cov"],
            num_teams=num_teams,
        ),
        get_dynamics_params=partial(
            model_moments.get_dynamics_params,
            init_mean=INIT_MEAN,
            init_chol_cov=params["init_chol_cov"],
            kappa=params["kappa"],
        ),
        get_observation_params=partial(
            model_moments.get_observation_params,
            alpha=params["alpha"],
            beta=params["beta"],
            friendly_scale=params["friendly_scale"],
        ),
    )
    final_factorial_state = factorial_filter(
        filter_obj,
        factorializer,
        model_inputs,
        output_factorial=False,
    )[-1]
    return final_factorial_state.log_normalizing_constant


def loss(raw_params: dict[str, jax.Array]) -> jax.Array:
    """Negative log normalizing constant for minimization."""
    return -log_normalizing_constant(raw_params)


optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(raw_params)
value_and_grad = jax.jit(jax.value_and_grad(loss))
history = []

for step in range(STEPS + 1):
    loss_value, grads = value_and_grad(raw_params)
    logz = -loss_value
    history.append(
        {
            "step": step,
            "loss": float(loss_value),
            "log_normalizing_constant": float(logz),
        }
    )

    if step % max(LOG_EVERY, 1) == 0 or step == STEPS:
        print(
            f"step={step:05d}, "
            f"log_normalizing_constant={float(logz):.6f}, "
            f"loss={float(loss_value):.6f}, "
            f"{format_params(constrain_params(raw_params))}"
        )

    if step == STEPS:
        break

    updates, opt_state = optimizer.update(grads, opt_state, raw_params)
    raw_params = cast(
        dict[str, jax.Array],
        optax.apply_updates(raw_params, updates),
    )

final_params = constrain_params(raw_params)
final_history = history[-1]

OUTPUT_DIR.mkdir(exist_ok=True)

with PARAMS_JSON_PATH.open("w") as f:
    json.dump(
        {
            "final_loss": final_history["loss"],
            "final_log_normalizing_constant": final_history["log_normalizing_constant"],
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "max_goals": MAX_GOALS,
            "max_matches": MAX_MATCHES,
            "params": params_to_jsonable(final_params),
        },
        f,
        indent=2,
    )

loss_plot = (
    pn.ggplot(pd.DataFrame(history), pn.aes("step", "loss"))
    + pn.geom_line()
    + pn.labs(
        x="Step",
        y="Negative log normalizing constant",
        title="Moments filter training loss",
    )
    + pn.theme_minimal()
)
loss_plot.save(LOSS_PLOT_PATH, width=7, height=4, dpi=150, verbose=False)

print("\nFinal")
print(f"log_normalizing_constant={final_history['log_normalizing_constant']:.6f}")
print(format_params(final_params))
print(f"Saved params to {PARAMS_JSON_PATH}")
print(f"Saved loss plot to {LOSS_PLOT_PATH}")
