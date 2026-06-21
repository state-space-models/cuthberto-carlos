r"""
SMC model for bivariate poisson football match scores.

Prior: bivariate normal for attack and defence for each team.

$$p(x_{i, 0}) = p(x_{i, 0}^A, x_{i, 0}^D) = \mathcal{N}(x_{i, 0}^A, x_{i, 0}^D | \mu_0, \Sigma_0)$$

Transition: bivariate normal for attack and defence which follows a OU process. i.e. the attack and defence exhibit mean-reverting behaviour to initial mean if no matches are played for a long time.

$$p(x^i_k \mid x^i_{k-1}) = \mathrm{N}\left(x^i_k \mid \mu_0 + \phi_k (x^i_{k-1} - \mu_0), Q_k\right)$$

- $\phi_k = \exp(-\kappa (t_k - t_{k-1}))$
- $Q_k = \Sigma_0 - \phi_k \Sigma_0 \phi_k^\top$

Likelihood: goals of a match modelled as a bivariate poisson distribution with attack and defence of both teams.

$$
p(y \mid x^i, x^j, \alpha, \beta) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y^i}}{y^i!} \frac{\lambda_2^{y^j}}{y^j!} \sum_{k=0}^{\min(y^i, y^j)} \binom{y^i}{k} \binom{y^j}{k} k! \left( \frac{\lambda_{3}}{\lambda_1 \lambda_2} \right)^k,
$$

with $\lambda_1 = \exp(\alpha + x^{\text{att}, i} - x^{\text{def}, j})$, $\lambda_2 = \exp(\alpha + x^{\text{att}, j} - x^{\text{def}, i})$ and $\lambda_3 = \exp(\beta)$.

The log potential of the model is also given by the likelihood function (bivariate poisson).

$$\log G_t(x^i, x^j, y) = \log p(y \mid x^i, x^j, \alpha, \beta) = - (\lambda_1 + \lambda_2 + \lambda_3) + y^i \log(\lambda_1) + y^j \log(\lambda_2) - \log(y^i!) - \log(y^j!) + \log \left(\sum_{k=0}^{\min(y^i, y^j)} \binom{y^i}{k} \binom{y^j}{k} k! \left( \frac{\lambda_3}{\lambda_1 \lambda_2} \right)^k \right)$$

Result Data(NamedTuple) is the state of the model and contains the following attributes:
    match_index (Array): Unique integer index for each match, starting from 0.
    home_team_id (Array): Integer ID for the home team.
    away_team_id (Array): Integer ID for the away team.
    home_score (Array): Integer number of goals scored by the home team in the match.
    away_score (Array): Integer number of goals scored by the away team in the match.
    neutral (Array): Boolean indicating whether the match was played on neutral ground.
    friendly (Array): Boolean indicating whether the match was a friendly match.
    timestamp (Array): Integer number of days since the origin date for the match.
    home_timestamp_previous (Array): Optional timestamp for the previous home team match.
    away_timestamp_previous (Array): Optional timestamp for the previous away team match.
"""

import os
import pandas as pd

import jax.numpy as jnp
import jax
import numpy as np
# from jax import Array, random
from functools import partial
import cuthbert
from cuthbert.smc.particle_filter import build_filter
from cuthbertlib.resampling import adaptive, systematic

from cuthberto_carlos.data_types import ResultData
from cuthberto_carlos.data import download_data, to_jax_data

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
    num_teams: int,
) -> jax.Array:
    x = jax.random.multivariate_normal(key, INIT_MEAN, INIT_COV, shape=(num_teams,))
    return x

def compute_ou_dynamics(
    state_team: jax.Array,  # (2,) - single team's (attack, defence)
    time_delta: jax.Array,  # scalar - days since last match
) -> tuple[jax.Array, jax.Array]:
    phi = jnp.exp(-INITIAL_KAPPA * time_delta)
    mean = INIT_MEAN + phi * (state_team - INIT_MEAN)
    # phi is a scalar, so phi * INIT_COV @ phi.T = phi^2 * INIT_COV
    cov = INIT_COV - phi**2 * INIT_COV
    # Add jitter to prevent singular covariance when time_delta == 0 (same-day
    # matches), which would cause multivariate_normal to return NaN.
    jitter = 1e-8
    cov = cov + jitter * jnp.eye(cov.shape[0])
    return mean, cov

def propagate_sample(
    key: jax.Array,
    state: jax.Array,
    model_inputs: ResultData
) -> jax.Array:
    """Compute OU dynamics for home and away and update state with new values. other teams remain unchanged.

    Args:
        key (jax.Array): Random key for JAX random number generation.
        state (jax.Array): Attack and defence state of all teams, shape (num_teams, 2).
        model_inputs (ResultData): Results of current matches.

    Returns:
        jax.Array: Updated state of all teams after propagating the home and away teams' states.
    """
    key_home, key_away = jax.random.split(key)

    # Sentinel for "no previous match" is 0 (set in download_data).
    # When there is no previous match, set dt = 0 so the state is unchanged.
    dt_home = jnp.where(
        model_inputs.home_timestamp_previous == 0,
        0,
        model_inputs.timestamp - model_inputs.home_timestamp_previous,
    )
    mean_home, cov_home = compute_ou_dynamics(state[model_inputs.home_team_id], dt_home)
    new_state_home = jax.random.multivariate_normal(
        key=key_home, 
        mean=mean_home, 
        cov=cov_home,
        method='cholesky')
    state = state.at[model_inputs.home_team_id].set(new_state_home)

    dt_away = jnp.where(
        model_inputs.away_timestamp_previous == 0,
        0,
        model_inputs.timestamp - model_inputs.away_timestamp_previous,
    )
    mean_away, cov_away = compute_ou_dynamics(state[model_inputs.away_team_id], dt_away)
    new_state_away = jax.random.multivariate_normal(
        key=key_away, 
        mean=mean_away, 
        cov=cov_away,
        method='cholesky')
    state = state.at[model_inputs.away_team_id].set(new_state_away)

    return state

def log_potential(
    state_prev: jax.Array,
    state: jax.Array,
    model_inputs: ResultData,
) -> jax.Array:
    """Log likelihood of results. Follows a bivariate poisson distribution.

    Args:
        state_prev (jax.Array): _description_
        state (jax.Array): _description_
        model_inputs (ResultData): _description_

    Returns:
        jax.Array: _description_
    """
    # lambda_1 = exp(alpha + attack_home - defence_away)
    lambda_1 = jnp.exp(INITIAL_ALPHA + state[model_inputs.home_team_id, 0] - state[model_inputs.away_team_id, 1])
    # lambda_2 = exp(alpha + attack_away - defence_home)
    lambda_2 = jnp.exp(INITIAL_ALPHA + state[model_inputs.away_team_id, 0] - state[model_inputs.home_team_id, 1])
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

def check_results(filter_state, N, num_teams, teams_id_to_name_dict):
    # Extract particles from ParticleFilterState
    particles = filter_state.particles  # (N, num_teams, 2)
    
    # 1. Verify shapes
    print(f"Filter state particles shape: {particles.shape}")  # Should be (N, num_teams, 2)
    print(f"Number of particles: {N}")
    print(f"Number of teams: {num_teams}")
    
    # 2. Check particle distribution for a specific team
    team_id = 0  # Change to check different teams
    team_name = teams_id_to_name_dict.get(team_id, f"Team {team_id}")
    team_particles = particles[:, team_id, :]  # (N, 2)
    
    print(f"\n--- Team: {team_name} ---")
    print(f"Attack mean ± std: {team_particles[:, 0].mean():.3f} ± {team_particles[:, 0].std():.3f}")
    print(f"Defence mean ± std: {team_particles[:, 1].mean():.3f} ± {team_particles[:, 1].std():.3f}")
    
    # 3. Check correlation between attack and defence
    print(f"Attack-Defence correlation: {jnp.corrcoef(team_particles.T)[0, 1]:.3f}")
    
    # 4. Compare top teams by attack strength
    print("\n--- Top 5 Teams by Attack (mean) ---")
    attack_means = particles[:, :, 0].mean(axis=0)  # (num_teams,)
    # Convert to numpy for easier NaN handling
    attack_means_np = np.array(attack_means)
    valid_indices = np.where(~np.isnan(attack_means_np))[0]
    valid_attack_means = attack_means_np[valid_indices]
    top_5_idx = np.argsort(valid_attack_means)[-5:][::-1]
    top_attack = valid_indices[top_5_idx]
    for idx in top_attack:
        name = teams_id_to_name_dict.get(int(idx), f"Team {idx}")
        print(f"  {name}: {attack_means[idx]:.3f}")
    
    # 5. Compare top teams by defence strength (lower is better)
    print("\n--- Top 5 Teams by Defence (mean, lower=better) ---")
    defence_means = particles[:, :, 1].mean(axis=0)
    top_defence = jnp.argsort(defence_means)[:5]  # Ascending for defence
    for idx in top_defence:
        name = teams_id_to_name_dict.get(int(idx), f"Team {idx}")
        print(f"  {name}: {defence_means[idx]:.3f}")
    
    # 6. Check particle diversity (should not be too low = collapse)
    print(f"\n--- Particle Diversity ---")
    for i in range(min(3, num_teams)):
        team_particles = particles[:, i, :]
        ess = jnp.sum(team_particles[:, 0])**2 / jnp.sum(team_particles[:, 0]**2)  # Rough ESS proxy
        print(f"Team {i} effective sample size proxy: {ess:.1f} / {N}")

def store_latest_filter_state(filter_state, filename="./scripts/smc/output/smc_filter_latest.npz"):
    """Store the latest filter state to a .npz file for later analysis."""
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez(filename, particles=np.array(filter_state.particles))
    print(f"Latest filter state stored in {filename}")

def main():
    N = 500  # Reduced from 1000 to avoid memory issues
    MAX_GOALS = 8

    pd_data, _, _, _ = download_data(max_goals=MAX_GOALS)
    
    train_data = pd_data[
        (pd_data['date'] > pd.to_datetime("2000-01-01"))
        & (pd_data['date'] < pd.to_datetime("2026-06-11"))
    ]
    teams_id_to_name_dict = {int(row['home_team_id']): row['home_team'] for _, row in train_data.iterrows()}
    jax_data = to_jax_data(train_data)
    num_teams = len(teams_id_to_name_dict)

    print(f"Training from {train_data['date'].min().date()} to {train_data['date'].max().date()}")
    print(f"Number of teams: {num_teams}")
    print(f"Number of matches: {len(train_data)}")

    # resampling function
    resampling_fn = adaptive.ess_decorator(
        func=systematic.resampling, 
        threshold=0.2,
    )
    # build filter
    # pass number of teams to init_sample function using partial
    smc_filter = build_filter(
        init_sample=partial(
            init_sample,
            num_teams=num_teams,
        ),
        propagate_sample=propagate_sample,
        log_potential=log_potential,
        n_filter_particles = N,
        resampling_fn=resampling_fn,
    )
    print(f"Running filter with {N} particles.....")
    # offline filtering (JIT-compiled: fuses the entire scan+vmap into one XLA program).
    # We wrap the filter so only the final state is returned, allowing XLA to
    # avoid materializing the full (T, N, num_teams, 2) history in memory.
    @partial(jax.jit, static_argnames=("filter_obj",))
    def run_filter(filter_obj, model_inputs, key):
        previous_states = cuthbert.filter(
            filter_obj=filter_obj,
            model_inputs=model_inputs,
            key=key,
        )
        return jax.tree.map(lambda x: x[-1], previous_states)

    key, previous_key = jax.random.split(jax.random.PRNGKey(0))
    filter_state = run_filter(
        filter_obj=smc_filter,
        model_inputs=jax_data,
        key=key,
    )
    print("Filter completed.")
    # store latest filter state for analysis
    store_latest_filter_state(filter_state)
    # check results
    check_results(filter_state, N, num_teams, teams_id_to_name_dict)
if __name__ == "__main__":
    main()