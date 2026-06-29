"""
Factorial SMC

"""

from functools import partial
import os
import pandas as pd
from tqdm import tqdm

import cuthbert
from cuthbert.factorial.smc import build_factorializer
from cuthbert.smc.particle_filter import build_filter
from cuthbertlib.resampling import no_resampling, systematic

from cuthberto_carlos.data import to_jax_data, download_data
from cuthberto_carlos.data_types import ResultData, DynamicsOnlyData

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree

MAX_GOALS = 8

INIT_MEAN = jnp.array([0.0, 0.0])
INIT_SD = 1.0
INIT_CORR = 0.0
INIT_COV = jnp.array([[INIT_SD**2, INIT_CORR * INIT_SD**2],
                    [INIT_CORR * INIT_SD**2, INIT_SD**2]])
INIT_CHOL_COV = jnp.linalg.cholesky(INIT_COV)
INITIAL_KAPPA = 1e-2
INITIAL_FRIENDLY_SCALE = 2.0
INITIAL_ALPHA = 0.2
INITIAL_BETA = -4.0

def init_sample(
        key: jax.Array,
        model_inputs: ResultData,
        num_teams: int
):
    """
    Init states for the SMC filter.
    """
    return jax.random.multivariate_normal(key, INIT_MEAN, INIT_COV, shape=(num_teams,))

def compute_ou_dynamics(
    state_team: jax.Array,  # (2,) - single team's (attack, defence)
    time_delta: jax.Array,  # scalar - days since last match
) -> tuple[jax.Array, jax.Array]:
    phi = jnp.exp(-INITIAL_KAPPA * time_delta)
    mean = INIT_MEAN + phi * (state_team - INIT_MEAN)
    # phi is a scalar, so phi * INIT_COV @ phi.T = phi^2 * INIT_COV
    cov = INIT_COV - phi**2 * INIT_COV
    # When time_delta == 0, phi == 1 and cov becomes a zero matrix.
    # jax.random.multivariate_normal returns NaN for a zero covariance matrix,
    # so add a tiny jitter to the diagonal to keep it positive-definite.
    cov = cov + jnp.eye(2) * 1e-8
    return mean, cov

def propagate_sample(
    key: jax.Array,
    state: jax.Array,
    model_inputs: ResultData
):
    """
    Propagate sample from a factorized model. tate: (4,) — home(2) + away(2)

    Args:
        key (jax.Array): _description_
        state (jax.Array): 2 x 2 array of attack and defense
        model_inputs (ResultData): _description_
    """
    key_home, key_away = jax.random.split(key)
    dt_home = jnp.where(model_inputs.home_timestamp_previous == 0, 0, model_inputs.timestamp - model_inputs.home_timestamp_previous)
    mean_home, cov_home = compute_ou_dynamics(state[:2], dt_home)
    state_home = jax.random.multivariate_normal(key_home, mean_home, cov_home)

    dt_away = jnp.where(model_inputs.away_timestamp_previous == 0, 0, model_inputs.timestamp - model_inputs.away_timestamp_previous)
    mean_away, cov_away = compute_ou_dynamics(state[2:], dt_away)
    state_away = jax.random.multivariate_normal(key_away, mean_away, cov_away)
    return jnp.concatenate([state_home, state_away])

def log_potential(
    state_prev: jax.Array,
    state: jax.Array,
    model_inputs: ResultData,
):
    """
    log potential for factorialized state. state: (4,) — home(2) + away(2)

    Args:
        state_prev (jax.Array): _description_
        state (jax.Array): _description_
        model_inputs (ResultData): _description_

    Returns:
        _type_: _description_
    """
    # lambda_1 = exp(alpha + attack_home - defence_away)
    lambda_1 = jnp.exp(INITIAL_ALPHA + state[0] - state[3])
    # lambda_2 = exp(alpha + attack_away - defence_home)
    lambda_2 = jnp.exp(INITIAL_ALPHA + state[2] - state[1])
    # lambda_3 = exp(beta)
    lambda_3 = jnp.exp(INITIAL_BETA)

    lambda_terms = - (lambda_1 + lambda_2 + lambda_3)
    home_term = model_inputs.home_score * jnp.log(lambda_1) - jax.scipy.special.gammaln(model_inputs.home_score + 1)
    away_term = model_inputs.away_score * jnp.log(lambda_2) - jax.scipy.special.gammaln(model_inputs.away_score + 1)
    
    # Sum over k = 0, 1, ..., min(home_score, away_score).
    # Use a fixed-size range (MAX_GOALS) with masking so shapes are static
    # under JAX tracing/JIT.
    k_max = jnp.minimum(model_inputs.home_score, model_inputs.away_score)
    k_range = jnp.arange(MAX_GOALS + 1)  # 0, 1, ..., MAX_GOALS
    mask = k_range <= k_max

    # log(C(n, k)) = log(n!) - log(k!) - log((n-k)!)
    log_comb_home = jax.scipy.special.gammaln(model_inputs.home_score + 1) - jax.scipy.special.gammaln(k_range + 1) - jax.scipy.special.gammaln(model_inputs.home_score - k_range + 1)
    log_comb_away = jax.scipy.special.gammaln(model_inputs.away_score + 1) - jax.scipy.special.gammaln(k_range + 1) - jax.scipy.special.gammaln(model_inputs.away_score - k_range + 1)
    log_k_factorial = jax.scipy.special.gammaln(k_range + 1)
    log_ratio = k_range * (jnp.log(lambda_3) - jnp.log(lambda_1) - jnp.log(lambda_2))

    terms = log_comb_home + log_comb_away + log_k_factorial + log_ratio
    terms = jnp.where(mask, terms, -jnp.inf)
    sum_terms = jax.scipy.special.logsumexp(terms)

    log_likelihood = lambda_terms + home_term + away_term + sum_terms

    return log_likelihood

# home_team_id: Array
# away_team_id: Array

def summary(factorial_state, train_data, teams_id_to_name_dict, N, num_teams):
    # Summary
    particles = np.array(factorial_state.particles)  # (F, P, 2)
    weights = jax.nn.softmax(factorial_state.log_weights, axis=-1)  # (F, P)
    weighted_means = np.sum(particles * weights[..., None], axis=1)  # (F, 2)
    weighted_vars = np.sum((particles - weighted_means[:, None, :]) ** 2 * weights[..., None], axis=1)  # (F, 2)

    print(f"\n{'='*60}")
    print(f"Factorial SMC Filter Summary")
    print(f"{'='*60}")
    print(f"Particles per factor:    {N}")
    print(f"Number of teams (factors): {num_teams}")
    print(f"Number of matches:        {len(train_data)}")
    print(f"Log normalizing constant: {float(factorial_state.log_normalizing_constant):.2f}")
    print(f"Training period:          {train_data['date'].min().date()} to {train_data['date'].max().date()}")

    # Top 5 teams by attack
    print(f"\n--- Top 5 Teams by Attack (higher = better) ---")
    attack_means = weighted_means[:, 0]
    valid = ~np.isnan(attack_means)
    top_attack = np.argsort(attack_means[valid])[-5:][::-1]
    valid_indices = np.where(valid)[0]
    for idx in top_attack:
        team_idx = int(valid_indices[idx])
        name = teams_id_to_name_dict.get(team_idx, f"Team {team_idx}")
        print(f"  {name}: attack={attack_means[team_idx]:.3f}, defence={weighted_means[team_idx, 1]:.3f}")

    # Top 5 teams by defence (higher = better, reduces opponent scoring)
    print(f"\n--- Top 5 Teams by Defence (higher = better) ---")
    defence_means = weighted_means[:, 1]
    valid_def = ~np.isnan(defence_means)
    top_defence = np.argsort(defence_means[valid_def])[-5:][::-1]
    valid_def_indices = np.where(valid_def)[0]
    for idx in top_defence:
        team_idx = int(valid_def_indices[idx])
        name = teams_id_to_name_dict.get(team_idx, f"Team {team_idx}")
        print(f"  {name}: attack={weighted_means[team_idx, 0]:.3f}, defence={defence_means[team_idx]:.3f}")

    # Particle diversity (ESS) for a few teams
    print(f"\n--- Particle Diversity (ESS) ---")
    for i in range(min(5, num_teams)):
        ess = 1.0 / np.sum(weights[i] ** 2)
        name = teams_id_to_name_dict.get(i, f"Team {i}")
        print(f"  {name}: ESS = {ess:.1f} / {N}")

    print(f"{'='*60}")

def main():
    N = 10000
    MAX_GOALS = 8

    pd_data, _, _, _ = download_data(max_goals=MAX_GOALS)

    train_data = pd_data[
        (pd_data['date'] > pd.to_datetime("2000-01-01"))
        & (pd_data['date'] < pd.to_datetime("2026-06-11"))
    ].copy()

    # Re-index team IDs to be contiguous (0..N-1) within the filtered subset.
    # download_data assigns IDs globally across all history, so filtering by date
    # leaves gaps. Re-indexing avoids ghost factors in the factorial state.
    teams_in_subset = sorted(set(train_data['home_team_id']) | set(train_data['away_team_id']))
    old_to_new_id = {old: new for new, old in enumerate(teams_in_subset)}
    train_data['home_team_id'] = train_data['home_team_id'].map(old_to_new_id)
    train_data['away_team_id'] = train_data['away_team_id'].map(old_to_new_id)

    teams_id_to_name_dict = {}
    for _, row in train_data.iterrows():
        teams_id_to_name_dict[int(row['home_team_id'])] = row['home_team']
        teams_id_to_name_dict[int(row['away_team_id'])] = row['away_team']

    jax_data = to_jax_data(train_data)
    num_teams = len(teams_id_to_name_dict)

    print(f"Training from {train_data['date'].min().date()} to {train_data['date'].max().date()}")
    print(f"Number of teams: {num_teams}")
    print(f"Number of matches: {len(train_data)}")

    # Build the factorial SMC filter. Same as SMC filter, but propagate sample is only on the 2x2 state.
    smc_filter = build_filter(
        init_sample=partial(
            init_sample,
            num_teams=num_teams
        ),
        propagate_sample=propagate_sample,
        log_potential=log_potential,
        n_filter_particles=N,
        resampling_fn=no_resampling.resampling,
    )
    # factoralizer manages the mapping of the state and the resampling function.
    factorializer = build_factorializer(
        get_factorial_indices=lambda model_inputs: jnp.array(
            [model_inputs.home_team_id, model_inputs.away_team_id]
        ),
        resampling_fn=systematic.resampling,
    )
    # init_state, local_states, final_state = cuthbert.factorial.filter(
    #     filter_obj=smc_filter,
    #     factorializer=factorializer,
    #     model_inputs=jax_data,
    #     output_factorial=False,
    #     key=jax.random.PRNGKey(0)
    # )
    keys = jax.random.split(jax.random.PRNGKey(0), len(jax_data.match_index))
    # map data into
    init_model_inputs = tree.map(lambda x: x[0], jax_data)
    factorial_state = smc_filter.init_prepare(init_model_inputs, key=keys[0])
    factorial_state = factorializer.factorialize_init_state(factorial_state, init_model_inputs)

    for t in tqdm(range(1, len(jax_data.match_index)), desc="Filtering"):
        model_inputs_t = tree.map(lambda x: x[t], jax_data)
        local_state = factorializer.extract_and_join(factorial_state, model_inputs_t)
        prep_state = smc_filter.filter_prepare(model_inputs_t, key=keys[t])
        filtered_joint = smc_filter.filter_combine(local_state, prep_state)
        factorial_state = factorializer.marginalize_and_insert(
            filtered_joint, factorial_state, model_inputs_t
        )

    summary(factorial_state, train_data, teams_id_to_name_dict, N, num_teams)

    # Store the final filter state
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fact_smc_filter_latest.npz")
    np.savez(
        output_path,
        particles=np.array(factorial_state.particles),
        log_weights=np.array(factorial_state.log_weights),
        log_normalizing_constant=np.array(factorial_state.log_normalizing_constant),
    )
    print(f"\nFinal filter state saved to {output_path}")

if __name__ == "__main__":
    main()