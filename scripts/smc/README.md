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
[ ] Perform diagnostics on SMC filter results
[ ] Compare SMC filter with Gaussian Moments Filter (compare distributions, whether Gaussian Moments Filter approximates SMC (true) distribution well)
[ ] Compare performance of SMC compared to Gaussian Moments Filter (Log-normalizing constant, predictive accuracy, etc.)
[ ] Implement SQMC
[ ] Perform diagnostics on SQMC results
[ ] Compare performance of SQMC, SMC and Gaussian Moments Filter

**260626** 
- Built SMC filter `model_smc.py`
  - Experiencing weight degeneracy issues. Max number of particles is 250. Used a shorter time frame (2000-01-01 to 06-11-2026) to reduce memory usage.
  - particle filter is sampling 2 x (1 x 2) x N = 4N particles at each time step. Currently, highest number N I tried is 250.