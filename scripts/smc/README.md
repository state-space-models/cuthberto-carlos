# SMC implementation of the Bivariate Poisson model for football scores

This directory contains a SMC implementation of the Bivariate Poisson model for football scores following the framework in `../../README.md`.

## Model

Define the latent state of a team $x_k \in \mathbb{R}^{2}$ where $x_k = [x_k^{\text{attack}}, x_k^{\text{defense}}]^\top$ and the observation of a match $y_k \in \mathbb{R}^{2}$ where $y_k = [y_k^{\text{home}}, y_k^{\text{away}}]^\top$. For $F$ unique teams over $K$ observations, the joint distribution of latent states $x_{0:K}^{1:F} \in \mathbb{R}^{F \times K}$ and observations $y_{1:K} \in \mathbb{R}^{K}$ is given by:

$$p(x_{0:K}^{1:F}, y_{1:K}) = p(x_{0}^{1:F}) \prod_{k=1}^{K} \left\{ p(x_k^{1:F} | x_{k-1}^{1:F}) \cdot G_k (y_k | x_k^{1:F}) \right\}$$

where $G_k (y_k | x_k^{1:F})$ is the likelihood of observation $y_k$ given the latent state $x_k^{1:F}$.

Following the works Glicko (Glickman, 1999) and TrueSkill (Dangauthier et al., 2008) by modelling the skill evolution with an Orstein-Uhlenbeck process, we have the following prior and transition distributions:

$$p(x_0^f) = \mathcal{N}(x_0^f | \mu_0, \Sigma_0)$$

$$p(x_k^f | x_{k-1}^f) = \mathcal{N}(x_k^f | \mu_0 + \phi_k(x_{k-1}^f - \mu_0), Q_k)$$

where $\phi_k = \exp(-\kappa \Delta t_k)$ and $Q_k = \Sigma_0 - \Phi_k \Sigma_0 \Phi_k$, which allows the skill evolution to mean revert back to the initial mean $\mu_0$ and covariance $\Sigma_0$. Further, we utilize the bivariate Poisson likelihood model proposed by Karlis and Ntzoufras (2003) to model the number of goals scored by the home and away teams in a match:

$$
G_k(y \mid x^{h(k)}, x^{a(k)}, \alpha, \beta) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y^{h(k)}}}{y^{h(k)}!} \frac{\lambda_2^{y^{a(k)}}}{y^{a(k)}!} \sum_{k=0}^{\min(y^{h(k)}, y^{a(k)})} \binom{y^{h(k)}}{k} \binom{y^{a(k)}}{k} k! \left( \frac{\lambda_{3}}{\lambda_1 \lambda_2} \right)^k,
$$

where $\lambda_1 = \exp(x_k^{h(k), \text{attack}} + x_k^{a(k), \text{defense}} + \alpha)$, $\lambda_2 = \exp(x_k^{a(k), \text{attack}} + x_k^{h(k), \text{defense}})$, $\lambda_3 = \exp(\beta)$, with $h(k)$ and $a(k)$ denoting the indices of home and away teams in match $k$ and $y_k = (y_k^{\text{home}}, y_k^{\text{away}})^\top$ denoting the number of goals scored by the home and away teams in match $k$. The parameters $\alpha$ and $\beta$ are used to model the home advantage and the correlation between the number of goals scored by the two teams, respectively.

### Factorial SSMs

A factorial ssm is a ssm where dynamics distributions factors into a product of independent distributions across factors. By denoting the latent state of factor $f$ at time $k$ as $x_k^f$, we can factorize the dynamics and observation distributions as:

$$p(x_k \mid x_{k-1}) = \prod_{f \in F} p(x_k^f \mid x_{k-1}^f)$$

$$p(y_k \mid x_k) = p (y_k | x_k^{(S_k)})$$

where $S_k \subseteq \{1, \ldots, F\}$ is a subset of factors involved in observation $y_k$. The posterior distribution can also be factorized as:

$$p(x_k \mid y_{1:k}) = \prod_{f \in F} p(x_k^f \mid y_{1:k})$$

Further assuming that 1) initial skills drawn are mutually independent 2) skills evolve independently over time 3) evolutions of skills are markovian, we can factorize the joint distribution as:

$$p(x_{0:K}^{1:F}, y_{1:K}) = \prod_{i \in F} \left\{ p_0^i (x_0^i) \cdot \prod_{k=1}^K p_{k-1, k}^i (x_{k-1}^i, x_k^i) \right\} \cdot \prod_{k=1}^K G_k (y_k | x_k^{h(k)}, x_{k}^{a(k)})$$

where $h(k)$ and $a(k)$ denote the indices of home and away teams in match $k$. This factorization reduces number of terms from $\mathcal{O}(2F \cdot K)$ to $\mathcal{O}(2F + K)$ where $F$ is the number of unique teams and $K$ is the number of observations.

However, this factorization removes any cross-team correlation generated from the likelihood function $G_k (y_k | x_k^{h(k)}, x_{k}^{a(k)})$. i.e. assume that $p(y_k | x_k^{h(k)}, x_{k}^{a(k)}) = p(y_k | x_k^{h(k)}) p(y_k | x_{k}^{a(k)})$.

[`cuthbert` Factorial SSMs](https://state-space-models.github.io/cuthbert/api_cuthbert/factorial/)

#### Gaussian Moments Factorial model

The Gaussian Moments factorial model assumes that the posterior is approximately gaussian. This is a General Gaussian Filtering.

$$p(x_k^f \mid y_{1:k}) \approx \mathbb{N}(x_k^f \mid m_k^f, \Sigma_k^f)$$

As a result, the posterior distribution can be approximated by the kalman filter update equations of the second order moments of latent states. The update equations are given by:

- **Prediction Step:**
  - $m_{k|k-1}^f = A m_{k-1|k-1}^f$
  - $\Sigma_{k|k-1}^f = A \Sigma_{k-1|k-1}^f A^T + Q$
- **Update Step:**
  - $S_k^f = H \Sigma_{k|k-1}^f H^T + R$
  - $K_k^f = \Sigma_{k|k-1}^f H^T S_k^{f,-1}$
  - $m_{k|k}^f = m_{k|k-1}^f + K_k^f (y_k - H m_{k|k-1}^f)$
  - $\Sigma_{k|k}^f = (I - K_k^f H) \Sigma_{k|k-1}^f$
- **Likelihood:**
  - $p(y_k \mid y_{1:k-1}) = \mathbb{N}(y_k \mid H m_{k|k-1}^f, S_k^f)$
  - $\log p(y_k \mid y_{1:k-1}) = -\frac{1}{2} \left( \log |S_k^f| + (y_k - H m_{k|k-1}^f)^T S_k^{f,-1} (y_k - H m_{k|k-1}^f) + d \log 2\pi \right)$
- **Posterior:**
  - $p(x_k^f \mid y_{1:k}) = \mathbb{N}(x_k^f \mid m_{k|k}^f, \Sigma_{k|k}^f)$

#### Factorial SMC

The Factorial SMC model does not assume that the posterior is approximately gaussian. Instead, it uses a particle filter to estimate the posterior distribution of the latent states.

$$p\left(\{x_{\ell^f}^f\}_{f \in F} \mid y_{1:K}\right) \propto \prod_{f \in F} \left\{ p_0 (x_0^f) \cdot \prod_{\ell^f \in \mathcal{L}_f} p_{\ell^{f,-}, \ell^f}^f (x_{\ell^{f,-}}^f, x_{\ell^f}^f) \right\} \cdot \prod_{k=1}^K G_k (y_k | x_k^{h(k)}, x_{k}^{a(k)})$$

where 

- $\mathcal{L}_f$ is the set of time indices for which factor $f$ is involved in an observation and
- $\ell^{f,-}$ is the previous time index for factor $f$ before $\ell^f$.

### SMC

Using SMC to estimate the joint posterior distribution is computationally expensive since it involves sampling from the latent states of all teams at each time step. The complexity is $\mathcal{O}(N \cdot 2F \cdot K)$, where $N$ is the number of particles.




In our problem, we assumed that the states evolved independently over time, so we can use Rao-Blackwellization to reduce the dimensionality of the problem to only involve teams involved in a match. This allows us to retain the cross-team correlation generated from the likelihood function $G_k (y_k | x_k^{h(k)}, x_{k}^{a(k)})$.

Factorializing runs independent particle filters for each team. This removes any cross-team correlation generated from the likelihood function $G_k (y_k | x_k^{h(k)}, x_{k}^{a(k)})$. However, this operation would be $\mathcal{O}(N \cdot F)$ instead of $\mathcal{O}(N + F)$ in Factorial SMC.

In our problem, we assumed that the states evolved independently over time, so we can use Rao-Blackwellization to reduce the dimensionality of the problem to only involve teams involved in a match. This allows us to retain the cross-team correlation generated from the likelihood function $G_k (y_k | x_k^{h(k)}, x_{k}^{a(k)})$. This operation is also $\mathcal{O}(N + F)$ since it only involves sampling from the teams involved in a match.

### Rao-Blackwellized SMC

$$p(x_{0:T}^{1:J} | y_{1:T}) = p(x_0^{1:J}, x_{1:T}^{E_{1:T}} \mid y_{1:T}) \cdot \underbrace{p(x_{1:T}^{-E_{1:T}} | x_0, x_{1:T}^{E_{1:T}})}_{\text{computed analytically}}$$

where 

- $E_i = \{a(i), h(i)\}$ is the set of teams involved in match $i$ and
- $x_{1:T}^{E_{1:T}} = \{x_1^{E_1}, x_2^{E_2} \ldots, x_T^{E_T}\}$ represents the latent state of teams involved in matches and 
- $x_{1:T}^{-E_{1:T}} = \{x_1^{-E_1}, x_2^{-E_2} \ldots, x_T^{-E_T}\}$ represents the latent state of teams not involved in matches. 

The first term can be estimated using SMC and the second term can be computed analytically.

$$p(x_t^{1:F} \mid y_{1:t}) = p(x_t^{E_t} \mid y_{1:t}) \cdot \prod_{f \notin E_t} p(x_t^f \mid y_{1:t})$$

### Rao-Blackwellized SQMC


### Comparison table

- Gaussian Moments and Factorial SMC lose the cross team correlation by decoupling the cross-correlation between teams. 3.4.1 Decoupling approximation
- Rao-Blackwellized SMC retain the cross-team correlation 

## To Do List

- [X] Implement SMC filter for the Bivariate Poisson model for football scores.
- [X] Perform diagnostics on SMC filter results - does not seem that accurate? partly because low particle count (N=250)
- [X] Build factorial SMC model for bivariate poisson.
- [X] Perform diagnostics on factorial SMC filter results
- [X] Implement smoothing for factorial SMC filter
- [ ] Details for Gaussian Moments, Factorial SMC, Rao-Blackwellized SMC, Rao-Blackwellized SQMC
- [ ] Implement SQMC
  - [ ] QMC generator
  - [ ] Hilbert sorting - follow adrien's example
- [ ] Perform diagnostics on SQMC results
- [ ] Compare performance of SQMC, Factorial SMC and Gaussian Moments Filter
- [ ] Compare factorial SMC filter with Gaussian Moments Filter (compare distributions, whether Gaussian Moments Filter approximates SMC (true) distribution well)
- [ ] Compare performance of factorial SMC compared to Gaussian Moments Filter (Log-normalizing constant, predictive accuracy, etc.)

**210626** 
- Built SMC filter `model_smc.py`
  - Experiencing weight degeneracy issues. Max number of particles is 250. Used a shorter time frame (2000-01-01 to 06-11-2026) to reduce memory usage.
  - particle filter is sampling 2 x (1 x 2) x N = 4N particles at each time step. Currently, highest number N I tried is 250.

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

**220626**
- Implement factorialized SMC filter: each team's particle filter is independent of the others
- ran on `N=10000` from `2000-01-01` to `2026-06-11`. computation is more efficient, so then naturally there are results from this. not sure why England / Portugal has nan values? even though they should have played matches.
  - ![alt text](./fact_smc/output/fact_smc_output.png)


**290626**
- Implement prediction and diagnostics for factorial SMC

**300626**
- Implement smoothing for factorial SMC
  - outputs in `scripts/smc/fact_smc/output/factsmc_smoothing_params.json`
  - design choices:
    - `max_transitions` - total transitions across all teams looked back for smoothing. Uses **stratified sampling** to ensure proper representation from all teams. Each team contributes equally (up to their available transitions), with the most recent transitions sampled from each team.
    - `max_matches` - total matches across all teams looked back for smoothing. Supports:
      - Integer (≥1): Use that many matches (e.g., `max_matches=100`)
      - Percentage (0-1): Use fraction of total matches (e.g., `max_matches=0.01` for 1%)
      - `-1`: Use all matches
  - **Implementation Details** (`smoothing_factsmc.py`): The smoothing EM algorithm alternates between:
    1. **E-step**: Run forward factorial SMC filter + backward smoothing for each team independently
    2. **M-step**: Update parameters using smoothed state distributions
      **M-step Parameter Updates:**
      
      | Parameter | Update Method | Sampling Strategy |
      |-----------|---------------|-------------------|
      | `kappa` (dynamics) | Gradient descent on OU transition likelihood | Stratified sampling across teams |
      | `alpha`, `beta` (observation) | Gradient descent on bivariate Poisson likelihood | Most recent matches |
      | `friendly_scale`, `neutral_scale` | Gradient descent with scaled loss | Sampled matches by type |
      | `init_mean`, `init_cov` | Closed-form from smoothed initial states | All teams |
    
    **Loss Function for Scales:**
    $$\text{loss} = -\frac{1}{N}\left(\sum_{\text{regular}} L_i + \frac{\sum_{\text{friendly}} L_j}{s_f} + \frac{\sum_{\text{neutral}} L_k}{s_n}\right)$$
    
    where $s_f$ = friendly_scale, $s_n$ = neutral_scale. Higher scales downweight those match types.
    **Stratified Sampling for Transitions:**
    - Organizes transitions by team_id in a dictionary
    - Distributes `max_transitions` evenly across all teams with transitions
    - Takes most recent transitions from each team
    - Ensures representation from teams with few games
  - Interpretation
    - Kappa $\kappa$ increased dramatically (0.01 to 0.60): Teams' skills revert to mean much faster than initially estimated. This suggests team strengths are more volatile than the initial guess.
    - Alpha $\alpha$ turned negative (0.2 to -0.22): The home advantage parameter became negative, which is interesting - it might indicate that in this dataset, home teams don't have an advantage, or the model is capturing other dynamics.
    - Beta $\beta$ increased (-4.0 to -3.25): Higher baseline for expected goals.
    - Scales increased: Both friendly ($s_{\text{friendly}}$) and neutral ($s_{\text{neutral}}$) scales increased, meaning these match types get downweighted more in the likelihood (higher scale = less weight).

**010726**
- Online particle smoothing: Duffield S., & Singh S. S. (2022). Online particle smoothing with application to map-matching. IEEE Transactions on Signal Processing, 70, 497–508. https://doi.org/10.1109/TSP.2022.3141259
  - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9676428
- Sam's paper mentioned local/blocked particle filter of Rebeschini P., & van Handel R. (2015)
- approximate smoothing and parameter estimation in high-dimensional ssms Finke and Singh (2017)