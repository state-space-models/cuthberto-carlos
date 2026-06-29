from scripts.smc.process_data import process_data_pl
from scripts.smc.fact_smc.model_factsmc import init_sample, propagate_sample, log_potential
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import cuthbert
import cuthbertlib
from cuthbert.smc import particle_filter
from cuthbert.factorial.smc import build_factorializer

def build_factsmc_model(N: int, num_teams: int) -> tuple[particle_filter.Filter, cuthbert.factorial.smc.Factorializer]:
    """Build factorial SMC model for bivariate Poisson.
    
    - no reasampling for factorial SMC in build_filter. resampling is part of the `build_factorializer` function. Refer to https://state-space-models.github.io/cuthbert/api_cuthbert/factorial/smc/ for more information.
    """

    smc_filter = particle_filter.build_filter(
        init_sample=partial(
            init_sample,
            num_teams=num_teams
        ),
        propagate_sample=propagate_sample,
        log_potential=log_potential,
        n_filter_particles=N,
        resampling_fn=cuthbertlib.resampling.no_resampling.resampling, # no resampling for factorial SMC. refer 
    )
    # 
    factorializer = build_factorializer(
        get_factorial_indices=lambda model_inputs: jnp.array(
            [model_inputs.home_team_id, model_inputs.away_team_id]
        ),
        resampling_fn=cuthbertlib.resampling.systematic.resampling,
    )
    return smc_filter, factorializer

def main():
    N = 10000

    pl_data, jax_data, teams_id_to_name_dict = process_data_pl(
        start_date="2020-01-01",
        end_date="2026-06-11",
        future_matches=False,
        max_goals=8,
    )
    num_teams = len(teams_id_to_name_dict)
    print(f"Matches: {pl_data['date'].min()} to {pl_data['date'].max()}, {len(pl_data)} matches")
    print(f"Number of teams: {num_teams}")
    smc_filter, factorializer = build_factsmc_model(N=N, num_teams=num_teams)

    print("Running factorial SMC...")
    init_state, local_states, final_state = cuthbert.factorial.filter(
        filter_obj=smc_filter,
        factorializer=factorializer,
        model_inputs=jax_data,
        output_factorial=False,
        key=jax.random.PRNGKey(0)
    )
    # particles = np.array(final_state.particles)
    print("Local states shape:", local_states[1].shape)
    print("Final state shape:", final_state[1].shape)
    print(final_state[1][0])
if __name__ == "__main__":
    main()