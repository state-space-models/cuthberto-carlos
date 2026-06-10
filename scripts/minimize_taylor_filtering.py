"""Gradient-free optimization of static Taylor-filtering parameters."""

from __future__ import annotations

from functools import partial
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, cast

import jax
from jax import Array, numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from cuthbert.gaussian import taylor
from cuthbert.factorial.gaussian import build_factorializer
from cuthbert.factorial import filter as factorial_filter

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cuthberto_carlos.data import download_data  # noqa: E402
from cuthberto_carlos.types import ResultData  # noqa: E402
from cuthberto_carlos import model  # noqa: E402


MAX_GOALS = 8
NUM_MATCHES = None  # Set to an int for quicker local iterations.

METHOD = "Powell"
MAXITER = 20
MAXFEV = 200
PRINT_EVERY = 1
PENALTY_LOSS = 1e30

# INIT_CHOL_COV = jnp.array([[1.0, 0.0], [0.0, 1.0]])
# TAU = 0.01
# KAPPA = 1e-4
# ALPHA = None  # Defaults to log average goals in drawn matches.
# BETA = -4.0

INIT_CHOL_COV = jnp.array([[0.1, 0.0], [0.0, 0.1]])
TAU = 0.01
KAPPA = 1e-4
ALPHA = 6.0
BETA = -5.0
COMPETITIVE_ABILITY_SCALE = 1.0
FRIENDLY_ABILITY_SCALE = 2.0

POSITIVE_EPS = 1e-6
ANISOTROPY_EPSILON = 1e-2
OBSERVATION_RIDGE_PRECISION = 1e-6
INIT_MEAN = jnp.array([0.0, 0.0])

OUTPUT_DIR = Path("outputs")
LOSS_PLOT_PATH = OUTPUT_DIR / "taylor_filtering_minimize_loss.png"
PARAMS_PATH = OUTPUT_DIR / "taylor_filtering_minimize_params.json"


def inverse_softplus(x: float | Array) -> Array:
    """Map a positive value back to an unconstrained raw value."""
    return jnp.log(jnp.expm1(jnp.asarray(x)))


def initial_raw_params(alpha: float) -> Array:
    """Pack initial constrained values into the raw vector optimized by SciPy."""
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


def params_dict(raw_params: Array | np.ndarray) -> dict[str, float]:
    """Return constrained parameters as plain Python floats."""
    raw_params = jnp.asarray(raw_params)
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


def save_loss_plot(history_data: pd.DataFrame) -> None:
    """Save a plotnine objective curve."""
    (OUTPUT_DIR / "matplotlib").mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "matplotlib"))

    import plotnine as pn

    plot = (
        pn.ggplot(history_data, pn.aes("eval", "best_loss"))
        + pn.geom_line()
        + pn.geom_point(size=0.8)
        + pn.labs(
            x="Objective evaluation",
            y="Best loss (-log normalizing constant)",
            title="Taylor Filtering Gradient-Free Optimization",
        )
        + pn.theme_minimal()
        + pn.theme(figure_size=(8, 4.5))
    )
    plot.save(LOSS_PLOT_PATH, dpi=150, verbose=False)


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
    """Return the negative final Taylor-filter log normalizing constant."""
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
history = []
best_loss = math.inf


def objective(raw_params: np.ndarray) -> float:
    """SciPy objective function with finite penalty for invalid proposals."""
    global best_loss

    eval_id = len(history)
    if not np.all(np.isfinite(raw_params)):
        loss = PENALTY_LOSS
    else:
        loss = float(value_fn(jnp.asarray(raw_params, dtype=jnp.float32)))
        if not math.isfinite(loss):
            loss = PENALTY_LOSS

    best_loss = min(best_loss, loss)
    row = {
        "eval": eval_id,
        "loss": loss,
        "best_loss": best_loss,
        "log_normalizing_constant": -loss,
        **params_dict(raw_params),
    }
    history.append(row)

    if eval_id % PRINT_EVERY == 0:
        print(
            f"eval={eval_id:05d} "
            f"loss={loss:.6f} "
            f"best_loss={best_loss:.6f} "
            f"init_attack_sd={row['init_attack_sd']:.6g}, "
            f"init_defence_sd={row['init_defence_sd']:.6g}, "
            f"init_corr={row['init_attack_defence_corr']:.6g}, "
            f"tau={row['tau']:.6g}, "
            f"kappa={row['kappa']:.6g}, "
            f"friendly_scale={row['friendly_ability_scale']:.6g}, "
            f"competitive_scale={row['competitive_ability_scale']:.6g}, "
            f"alpha={row['alpha']:.6g}, "
            f"beta={row['beta']:.6g}"
        )

    return loss


raw_params0 = np.asarray(initial_raw_params(alpha), dtype=np.float64)
result = minimize(
    objective,
    raw_params0,
    method=METHOD,
    options={
        "maxiter": MAXITER,
        "maxfev": MAXFEV,
        "disp": True,
    },
)

OUTPUT_DIR.mkdir(exist_ok=True)
history_data = pd.DataFrame(history)
save_loss_plot(history_data)

with PARAMS_PATH.open("w") as f:
    json.dump(
        {
            "params": params_dict(result.x),
            "raw_params": [float(x) for x in result.x],
            "loss": float(result.fun),
            "log_normalizing_constant": float(-result.fun),
            "success": bool(result.success),
            "message": str(result.message),
            "method": METHOD,
            "num_evaluations": int(result.nfev),
            "num_iterations": int(getattr(result, "nit", -1)),
            "num_matches": NUM_MATCHES,
            "max_goals": MAX_GOALS,
            "anisotropy_epsilon": ANISOTROPY_EPSILON,
            "observation_ridge_precision": OBSERVATION_RIDGE_PRECISION,
        },
        f,
        indent=2,
        sort_keys=True,
    )

print(f"Saved loss plot to {LOSS_PLOT_PATH}")
print(f"Saved learned parameters to {PARAMS_PATH}")
