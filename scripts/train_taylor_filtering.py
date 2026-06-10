"""Train static parameters for Taylor filtering."""

from __future__ import annotations

from functools import partial
import json
import os
from pathlib import Path
import sys
from typing import Any, cast

import jax
from jax import Array, numpy as jnp
import optax
import pandas as pd
from cuthbert.gaussian import taylor
from cuthbert.factorial.gaussian import build_factorializer
from cuthbert.factorial import filter as factorial_filter

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cuthberto_carlos.data import download_data  # noqa: E402
from cuthberto_carlos.types import ResultData  # noqa: E402
from cuthberto_carlos import model  # noqa: E402


MAX_GOALS = 8
NUM_MATCHES = None  # Set to an int for quicker local iterations.

NUM_STEPS = 100
LEARNING_RATE = 1e-5
MAX_GRAD_NORM = 10.0
PRINT_EVERY = 1
MAX_STEP_HALVINGS = 12
OUTPUT_DIR = Path("outputs")
LOSS_PLOT_PATH = OUTPUT_DIR / "taylor_filtering_training_loss.png"
PARAMS_PATH = OUTPUT_DIR / "taylor_filtering_params.json"

INIT_CHOL_COV = jnp.array([[1.0, 0.0], [0.0, 1.0]])
TAU = 0.01
KAPPA = 1e-4
ALPHA = None  # Defaults to log average goals in drawn matches.
BETA = -4.0
COMPETITIVE_ABILITY_SCALE = 1.0
FRIENDLY_ABILITY_SCALE = 2.0

POSITIVE_EPS = 1e-6
ANISOTROPY_EPSILON = 1e-2
OBSERVATION_RIDGE_PRECISION = 1e-6
INIT_MEAN = jnp.array([0.0, 0.0])


def inverse_softplus(x: float | Array) -> Array:
    """Map a positive value back to an unconstrained raw value."""
    return jnp.log(jnp.expm1(jnp.asarray(x)))


def initial_raw_params(alpha: float) -> Array:
    """Pack initial constrained values into the raw vector optimized by Optax."""
    friendly_scale_delta = FRIENDLY_ABILITY_SCALE - COMPETITIVE_ABILITY_SCALE
    positive_params = (
        jnp.array(
            [
                INIT_CHOL_COV[0, 0],
                INIT_CHOL_COV[1, 1],
                TAU,
                KAPPA,
                COMPETITIVE_ABILITY_SCALE,
                friendly_scale_delta,
            ]
        )
        - POSITIVE_EPS
    )
    return jnp.concatenate(
        [
            jnp.array(
                [
                    inverse_softplus(positive_params[0]),
                    INIT_CHOL_COV[1, 0],
                    inverse_softplus(positive_params[1]),
                    inverse_softplus(positive_params[2]),
                    inverse_softplus(positive_params[3]),
                    inverse_softplus(positive_params[4]),
                    inverse_softplus(positive_params[5]),
                ]
            ),
            jnp.array([alpha, BETA]),
        ]
    )


def unpack_params(
    raw_params: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Transform raw params into init_chol_cov, tau, kappa, scales, alpha, beta."""
    init_chol_cov = jnp.array(
        [
            [jax.nn.softplus(raw_params[0]) + POSITIVE_EPS, 0.0],
            [raw_params[1], jax.nn.softplus(raw_params[2]) + POSITIVE_EPS],
        ]
    )
    tau = jax.nn.softplus(raw_params[3]) + POSITIVE_EPS
    kappa = jax.nn.softplus(raw_params[4]) + POSITIVE_EPS
    competitive_ability_scale = jax.nn.softplus(raw_params[5]) + POSITIVE_EPS
    friendly_scale_delta = jax.nn.softplus(raw_params[6]) + POSITIVE_EPS
    friendly_ability_scale = competitive_ability_scale + friendly_scale_delta
    alpha, beta = raw_params[7], raw_params[8]
    return (
        init_chol_cov,
        tau,
        kappa,
        friendly_ability_scale,
        competitive_ability_scale,
        alpha,
        beta,
    )


def init_cov_summary(init_chol_cov: Array) -> dict[str, Array]:
    """Return interpretable scalars from the initial Cholesky factor."""
    init_cov = init_chol_cov @ init_chol_cov.T
    init_std = jnp.sqrt(jnp.diag(init_cov))
    init_corr = init_cov[0, 1] / (init_std[0] * init_std[1])
    return {
        "init_attack_sd": init_std[0],
        "init_defence_sd": init_std[1],
        "init_attack_defence_cov": init_cov[0, 1],
        "init_attack_defence_corr": init_corr,
    }


def add_initial_dummy(jax_data: ResultData) -> ResultData:
    """Add the dummy first element expected by factorial_filter."""
    return jax.tree.map(
        lambda x: jnp.concatenate([jnp.zeros_like(x[:1]), x], axis=0),
        jax_data,
    )


def slice_matches(jax_data: ResultData) -> ResultData:
    """Optionally slice the data for faster experiments."""
    if NUM_MATCHES is None:
        return jax_data
    return jax.tree.map(lambda x: x[:NUM_MATCHES], jax_data)


def format_params(raw_params: Array) -> str:
    """Format constrained parameters for progress logging."""
    (
        init_chol_cov,
        tau,
        kappa,
        friendly_ability_scale,
        competitive_ability_scale,
        alpha,
        beta,
    ) = unpack_params(raw_params)
    init_summary = init_cov_summary(init_chol_cov)
    return (
        f"init_attack_sd={float(init_summary['init_attack_sd']):.6g}, "
        f"init_defence_sd={float(init_summary['init_defence_sd']):.6g}, "
        f"init_corr={float(init_summary['init_attack_defence_corr']):.6g}, "
        f"tau={float(tau):.6g}, "
        f"kappa={float(kappa):.6g}, "
        f"friendly_scale={float(friendly_ability_scale):.6g}, "
        f"competitive_scale={float(competitive_ability_scale):.6g}, "
        f"alpha={float(alpha):.6g}, "
        f"beta={float(beta):.6g}"
    )


def params_dict(raw_params: Array) -> dict[str, float]:
    """Return constrained parameters as plain Python floats."""
    (
        init_chol_cov,
        tau,
        kappa,
        friendly_ability_scale,
        competitive_ability_scale,
        alpha,
        beta,
    ) = unpack_params(raw_params)
    init_cov = init_chol_cov @ init_chol_cov.T
    init_summary = init_cov_summary(init_chol_cov)
    return {
        "init_chol_cov_00": float(init_chol_cov[0, 0]),
        "init_chol_cov_10": float(init_chol_cov[1, 0]),
        "init_chol_cov_11": float(init_chol_cov[1, 1]),
        "init_cov_00": float(init_cov[0, 0]),
        "init_cov_01": float(init_cov[0, 1]),
        "init_cov_10": float(init_cov[1, 0]),
        "init_cov_11": float(init_cov[1, 1]),
        **{key: float(value) for key, value in init_summary.items()},
        "tau": float(tau),
        "kappa": float(kappa),
        "friendly_ability_scale": float(friendly_ability_scale),
        "competitive_ability_scale": float(competitive_ability_scale),
        "alpha": float(alpha),
        "beta": float(beta),
    }


def save_loss_plot(history_data: pd.DataFrame) -> None:
    """Save a plotnine loss curve."""
    (OUTPUT_DIR / "matplotlib").mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "matplotlib"))

    import plotnine as pn

    loss_plot = (
        pn.ggplot(history_data, pn.aes("step", "loss"))
        + pn.geom_line()
        + pn.geom_point(size=0.8)
        + pn.labs(
            x="Step",
            y="Loss (-log normalizing constant)",
            title="Taylor Filtering Training Loss",
        )
        + pn.theme_minimal()
        + pn.theme(figure_size=(8, 4.5))
    )
    loss_plot.save(LOSS_PLOT_PATH, dpi=150, verbose=False)


pd_data, jax_data, teams_id_to_name_dict, _ = download_data(max_goals=MAX_GOALS)
jax_data = slice_matches(jax_data)
model_inputs = add_initial_dummy(jax_data)
num_teams = len(teams_id_to_name_dict)
factorializer = build_factorializer(get_factorial_indices=model.get_factorial_inds)

if ALPHA is None:
    draw_scores = pd_data[pd_data["home_score"] == pd_data["away_score"]][
        ["home_score", "away_score"]
    ]
    alpha = float(jnp.log(draw_scores.to_numpy().mean()))
else:
    alpha = ALPHA


def neg_log_normalizing_constant(raw_params: Array) -> Array:
    """Return the final Taylor-filter log normalizing constant."""
    (
        init_chol_cov,
        tau,
        kappa,
        friendly_ability_scale,
        competitive_ability_scale,
        alpha,
        beta,
    ) = unpack_params(raw_params)

    filter_obj = taylor.build_filter(
        get_init_log_density=partial(
            model.get_init_log_density,
            init_mean=INIT_MEAN,
            init_chol_cov=init_chol_cov,
            num_teams=num_teams,
            anisotropy_epsilon=ANISOTROPY_EPSILON,
        ),
        get_dynamics_log_density=partial(
            model.get_dynamics_log_density,
            tau=tau,
            init_mean=INIT_MEAN,
            kappa=kappa,
            anisotropy_epsilon=ANISOTROPY_EPSILON,
        ),
        get_observation_func=partial(
            model.get_observation_log_potential,
            alpha=alpha,
            beta=beta,
            max_goals=MAX_GOALS,
            friendly_ability_scale=friendly_ability_scale,
            competitive_ability_scale=competitive_ability_scale,
            anisotropy_epsilon=ANISOTROPY_EPSILON,
            ridge_precision=OBSERVATION_RIDGE_PRECISION,
        ),
        rtol=1e-7,
    )
    out = cast(
        Any,
        factorial_filter(
            filter_obj,
            factorializer,
            model_inputs,
            output_factorial=True,
        ),
    )
    return -out.log_normalizing_constant[-1]


value_fn = jax.jit(neg_log_normalizing_constant)
grad_fn = jax.jit(jax.jacfwd(neg_log_normalizing_constant))
optimizer = optax.chain(
    optax.clip_by_global_norm(MAX_GRAD_NORM),
    optax.adam(LEARNING_RATE),
)

raw_params = initial_raw_params(alpha)
opt_state = optimizer.init(raw_params)
history = []

for step in range(NUM_STEPS + 1):
    loss_value = value_fn(raw_params)
    grads = grad_fn(raw_params)
    if not bool(jnp.isfinite(loss_value) & jnp.all(jnp.isfinite(grads))):
        raise FloatingPointError("Non-finite loss or gradient.")

    log_z = -loss_value
    grad_norm = optax.global_norm(grads)
    history.append(
        {
            "step": step,
            "loss": float(loss_value),
            "log_normalizing_constant": float(log_z),
            "grad_norm": float(grad_norm),
            **params_dict(raw_params),
        }
    )

    if step % PRINT_EVERY == 0 or step == NUM_STEPS:
        print(
            f"step={step:05d} "
            f"log_normalizing_constant={float(log_z):.6f} "
            f"grad_norm={float(grad_norm):.6g} "
            f"{format_params(raw_params)}"
        )

    if step == NUM_STEPS:
        break

    updates, opt_state = optimizer.update(grads, opt_state, raw_params)
    updates = cast(Array, updates)
    for step_halvings in range(MAX_STEP_HALVINGS + 1):
        update_scale = 0.5**step_halvings
        candidate_params = cast(
            Array, optax.apply_updates(raw_params, updates * update_scale)
        )
        candidate_loss = value_fn(candidate_params)
        candidate_grads = grad_fn(candidate_params)
        if bool(jnp.isfinite(candidate_loss) & jnp.all(jnp.isfinite(candidate_grads))):
            raw_params = candidate_params
            break
    else:
        print("Stopping: no finite update found from the current parameters.")
        break

OUTPUT_DIR.mkdir(exist_ok=True)
history_data = pd.DataFrame(history)
save_loss_plot(history_data)

with PARAMS_PATH.open("w") as f:
    json.dump(
        {
            "params": params_dict(raw_params),
            "loss": float(history[-1]["loss"]),
            "log_normalizing_constant": float(history[-1]["log_normalizing_constant"]),
            "num_steps": NUM_STEPS,
            "num_matches": NUM_MATCHES,
            "max_goals": MAX_GOALS,
            "learning_rate": LEARNING_RATE,
            "max_grad_norm": MAX_GRAD_NORM,
            "anisotropy_epsilon": ANISOTROPY_EPSILON,
            "observation_ridge_precision": OBSERVATION_RIDGE_PRECISION,
        },
        f,
        indent=2,
        sort_keys=True,
    )

print(f"Saved loss plot to {LOSS_PLOT_PATH}")
print(f"Saved learned parameters to {PARAMS_PATH}")
