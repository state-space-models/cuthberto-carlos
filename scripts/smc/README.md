# SMC implementation of the Bivariate Poisson model for football scores

This directory contains a SMC implementation of the Bivariate Poisson model for football scores following the framework in `../../README.md`.

```python
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
```

[X] Implement SMC filter for the Bivariate Poisson model for football scores.
[X] Perform diagnostics on SMC filter results - does not seem that accurate? partly because low particle count (N=250)
[X] Build factorial SMC model for bivariate poisson.
[X] Perform diagnostics on factorial SMC filter results
[ ] Compare factorial SMC filter with Gaussian Moments Filter (compare distributions, whether Gaussian Moments Filter approximates SMC (true) distribution well)
[ ] Compare performance of factorial SMC compared to Gaussian Moments Filter (Log-normalizing constant, predictive accuracy, etc.)
[ ] Implement SQMC
[ ] Perform diagnostics on SQMC results
[ ] Compare performance of SQMC, SMC and Gaussian Moments Filter

**210626** 
- Built SMC filter `model_smc.py`
  - Experiencing weight degeneracy issues. Max number of particles is 250. Used a shorter time frame (2000-01-01 to 06-11-2026) to reduce memory usage.
  - particle filter is sampling 2 x (1 x 2) x N = 4N particles at each time step. Currently, highest number N I tried is 250.

**220626**
- Implement factorialized SMC filter: each team's particle filter is independent of the others
- ran on `N=10000` from `2000-01-01` to `2026-06-11`. computation is more efficient, so then naturally there are results from this. not sure why England / Portugal has nan values? even though they should have played matches.
  - ![alt text](./fact_smc/output/fact_smc_output.png)

# Models

## SMC

## Factorial SMC

## Rao-Blackwellized SMC

## Rao-Blackwellized SQMC