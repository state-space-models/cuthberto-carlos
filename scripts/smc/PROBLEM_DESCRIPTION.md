# SQMC Problem Description

## Comparison Table

| Aspect | Factorial Moments | Factorial SMC | SMC (Standard) | RB SMC | RBSQMC |
|--------|-------------------|---------------|----------------|--------|--------|
| **Dimensionality** | $2F$ (all teams) | $2F$ (all teams) | $2F \times 2F$ (full joint) | $2 \times 2$ (playing teams only) | $2 \times 2$ (playing teams only) |
| **Sampling Method** | Moment matching / Mean-field | Monte Carlo (random) | Monte Carlo (random) | Monte Carlo (random) | Quasi-Monte Carlo (low-discrepancy) |
| **Non-playing Teams** | Analytical update | Sampled (Point-in-time evaluation) | Sampled | Analytical (RB) | Analytical (RB) |
| **Posterior Approximation** | Gaussian (moments) | Empirical (particles) | Empirical (particles) | Empirical (particles) + analytical covariance | Empirical (particles) + analytical covariance |
| **Resampling** | Not applicable | When ESS < threshold | When ESS < threshold | When ESS < threshold | Every step (implicit via Hilbert sorting) |
| **Key Algorithm** | Kalman-like moment propagation | Bootstrap particle filter | Bootstrap particle filter | RB particle filter | RB SQMC with Hilbert sort |
| **Complexity** | $\mathcal{O}(F \cdot K)$ | $\mathcal{O}(N \cdot 2F \cdot K)$ | $\mathcal{O}(N \cdot 2F \cdot 2F \cdot K)$ | $\mathcal{O}(N \cdot 2F \cdot K)$ | $\mathcal{O}(N \log N \cdot 2F \cdot K)$ |
| **Transition Handling** | Analytical (OU process) | Sampled | Sampled | Analytical for non-playing | Analytical for non-playing |
| **Independence of states between teams** | $p(x_k^{h}, x_k^{a}) \approx p(x_k^{h})p(x_k^{a})$ | Yes | No | Yes | Yes |

## Background

Following the works Glicko (Glickman, 1999) and TrueSkill (Dangauthier et al., 2008) by modelling the skill evolution with an Ornstein-Uhlenbeck process, we have the following prior and transition distributions:

$$p(x_0^f) = \mathcal{N}(x_0^f | \mu_0, \Sigma_0)$$

$$p(x_k^f | x_{k-1}^f) = \mathcal{N}(x_k^f | \mu_0 + \phi_k^f(x_{k-1}^f - \mu_0), Q_k^f)$$

where $\phi_k^f = \exp(-\kappa \Delta t_k^f)$, $Q_k^f = \Sigma_0 - \Phi_k \Sigma_0 \Phi_k$ and $\Phi_k = \text{diag}(\phi_k^f)$, which allows the skill evolution to mean revert back to the initial mean $\mu_0$ and covariance $\Sigma_0$. Further, we utilize the bivariate Poisson likelihood model proposed by Karlis and Ntzoufras (2003) to model the number of goals scored by the home and away teams in a match:

$$
G_k(y \mid x^{h(k)}, x^{a(k)}, \alpha, \beta) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y^{h(k)}}}{y^{h(k)}!} \frac{\lambda_2^{y^{a(k)}}}{y^{a(k)}!} \sum_{k=0}^{\min(y^{h(k)}, y^{a(k)})} \binom{y^{h(k)}}{k} \binom{y^{a(k)}}{k} k! \left( \frac{\lambda_{3}}{\lambda_1 \lambda_2} \right)^k,
$$

where 

- $\lambda_1 = \exp(x_k^{h(k), \text{attack}} + x_k^{a(k), \text{defense}} + \alpha)$, 
- $\lambda_2 = \exp(x_k^{a(k), \text{attack}} + x_k^{h(k), \text{defense}})$, 
- $\lambda_3 = \exp(\beta)$, with $h(k)$ and $a(k)$ denoting the indices of home and away teams in match $k$ and $y_k = (y_k^{\text{home}}, y_k^{\text{away}})^\top$ denoting the number of goals scored by the home and away teams in match $k$.
- $\alpha$ - home advantage
- $\beta$ - correlation between the number of goals scored by the two teams

$$p(x_k^{h(k)}, x_k^{a(k)} | y_{1:k}) \propto G_k(y_k | x_k^{h(k)}, x_k^{a(k)}) \cdot p(x_k^{h(k)} | y_{1:k-1}) \cdot p(x_k^{a(k)} | y_{1:k-1})$$

and assume independent evolution of skills across teams, so the joint posterior is

$$p(x_k^{h(k)}, x_k^{a(k)} | y_{1:k}) = p(x_k^{h(k)} | y_{1:k}) \cdot p(x_k^{a(k)} | y_{1:k})$$

## Factorial SSMs

A factorial ssm is a ssm where dynamics distributions factors into a product of independent distributions across factors.

Assuming that 1) initial skills drawn are mutually independent 2) skills evolve independently over time 3) evolutions of skills are markovian, we can factorize the joint distribution as:

$$p(x_{0:K}^{1:F} \mid y_{1:K}) = \prod_{f \in F} \left\{ p_0^f (x_0^f) \cdot \prod_{k=1}^K p_{k-1, k}^f (x_{k-1}^f, x_k^f) \right\} \cdot \prod_{k=1}^K G_k (y_k | x_k^{h(k)}, x_{k}^{a(k)})$$

where $h(k)$ and $a(k)$ denote the indices of home and away teams in match $k$. This factorization reduces number of terms from $\mathcal{O}(2F \cdot K)$ to $\mathcal{O}(2F + K)$ where $F$ is the number of unique teams and $K$ is the number of observations.

The approximation of the joint posterior is assumed to be independent across teams,

$$p(x_k^h(k), x_k^{a(k)} | y_{1:k}) \approx p(x_k^{h(k)} | y_{1:k}) \cdot p(x_k^{a(k)} | y_{1:k})$$

### Factorial SMC

The Factorial SMC model does not assume that the posterior is approximately gaussian. Instead, it uses a particle filter to estimate the posterior distribution of the latent states.

$$p\left(\{x_{\ell^f}^f\}_{f \in F} \mid y_{1:K}\right) \propto \prod_{f \in F} \left\{ p_0 (x_0^f) \cdot \prod_{\ell^f \in \mathcal{L}_f} p_{\ell^{f,-}, \ell^f}^f (x_{\ell^{f,-}}^f, x_{\ell^f}^f) \right\} \cdot \prod_{k=1}^K G_k (y_k | x_k^{h(k)}, x_{k}^{a(k)})$$

where 

- $\mathcal{L}_f$ is the set of time indices for which factor $f$ is involved in an observation and
- $\ell^{f,-}$ is the previous time index for factor $f$ before $\ell^f$.

[`cuthbert` Factorial SSMs](https://state-space-models.github.io/cuthbert/api_cuthbert/factorial/)

## SMC

### Rao-Blackwellization SMC

Denote $E_k = \{h(k), a(k)\}$ as the set of team indices involved in match $k$. Then, we can factorize the joint posterior as:

$$p(x_{0:K}^{1:F} \mid y_{1:K}) = p(x_{0}^{1:F}, x_{1:K}^{E_{1:K}} \mid y_{1:K}) p(x^{-E_{1:K}}_{1:K} \mid y_{1:K})$$

where $x^{-E_{1:K}}_{1:K}$ denotes the latent states not involved in a game at time $k$. The second term can be computed analytically since the OU-transition is linear gaussian. If $x_{k-1}^f \sim \mathcal{N}(\mu_{k-1}^f, \Sigma_{k-1}^f)$, then:

$$x_k^f \mid x_{k-1}^f \sim \mathcal{N}(\mu_0 + \phi_k^f (x_{k-1}^f - \mu_0), (\phi_k^f)^2\Sigma_{k-1}^f + Q_{k-1}^f)$$

So for the non-playing teams, the posterior distribution is given by:

$$p(x_k^{-E_k} \mid y_{k}) = p(x_k^{-E_k} \mid y_{k - 1}) = \prod_{f \notin E_k} \mathcal{N}(x_k^f | \mu_0 + \phi_k^f (x_{k-1}^f - \mu_0), (\phi_k^f)^2\Sigma_{k-1}^f + Q_{k-1}^f)$$

For the playing teams, we utilize SMC to sample from the joint posterior distribution.

$$p(x_k^{E_k} \mid y_{1:k}) \propto  p(x_k^{E_k} | y_{1:k-1}) \cdot G_k(y_k | x_k^{E_k}) $$

The weights of the particles (from a bootstrap particle filter $q=p$) are then given by

$$w_k^{(i)} \propto w_{k-1}^{(i)} \cdot G_k(y_k | x_k^{E_k, (i)})$$

$$\tilde{w}_k^{(i)} = \frac{w_k^{(i)}}{\sum_{j=1}^N w_k^{(j)}}$$

Then perform resampling if needed. By assuming that the latent states $x_k^f$ of two teams are independent, we can split the joint posterior distribution into two independent distributions for each team:

$$p(x_k^{E_k} \mid y_{1:k}) = p(x_k^{h(k)} \mid y_{1:k}) \cdot p(x_k^{a(k)} \mid y_{1:k})$$

These particles $\{x_k^{E_k, (i)}\}_{i=1}^N$ represent the posterior mean for the playing teams. The posterior covariance is computed analytically using the same OU-transition as above.

Compared to SMC, this method is more efficient since it only involves reducing the sampling problem to a $2 \times 2$ dimensional problem instead of a $2F \times 2F$ dimensional problem. The complexity is $\mathcal{O}(N \cdot 2F \cdot K)$ instead of $\mathcal{O}(N \cdot 2F \cdot 2F \cdot K)$. The SMC algorithm is as follows:

1. Generate $N$ initial particles for latent states $x_0^{1:F} \sim \mathcal{N}(\mu_0, \Sigma_0)$ and compute the initial weights $w_0^{(i)} = 1/N$. Set initial posterior covariance $\Sigma_0^f = \Sigma_0$ for each team $f$.
2. For each observation $k = 1, \ldots, K$:
   1. For each team $f \notin E_k$, compute the posterior mean $\mu_k^f$ and covariance $\Sigma_k^f$ using the OU-transition.
   2. For each team $f \in E_k$, 
      1. sample $N$ particles from the proposal distribution $q(x_k^{E_k} | x_{k-1}^{E_k}) \sim \mathcal{N}(\mu_0 + \phi_k^f (x_{k-1}^f - \mu_0), (\phi_k^f)^2\Sigma_{k-1}^f + Q_{k-1}^f)$ and 
      2. compute the weights $w_k^{(i)} \propto w_{k-1}^{(i)} \cdot G_k(y_k | x_k^{E_k, (i)})$.
      3. Compute the normalized weights $\tilde{w}_k^{(i)} = \frac{w_k^{(i)}}{\sum_{j=1}^N w_k^{(j)}}$.
      4. Resample if $ESS < \text{threshold}$.
   3. Compute posterior covariance $\Sigma_k^f = (\phi_k^f)^2\Sigma_{k-1}^f + Q_{k-1}^f$ for each team $f \in E_k$.
   4. Update the posterior mean $\mu_k^f$ and covariance $\Sigma_k^f$ for each team $f \in E_k$.
3. Return the posterior mean and covariance for each team $f$ at each time step $k$.

### Rao-Blackwellization SQMC

The problem performs a filter for 2 teams for each observation, which means that the effective dimension is $2 \times 2$ instead of a $2F \times 2F$. Chopin and Gerber (2017) suggests that a reduced dimension problem may improve the performance of SQMC compared to SMC.

In this case, we are performing importance sampling at every step $k$ for 2 playing teams $x_k^{E_k} = \{x_k^{h(k)}, x_k^{a(k)}\}$, which is a 4-dimensional problem.

The SQMC algorithm is as follows:

1. Generate $N$ initial particles for latent states $x_0^{1:F} \sim \mathcal{N}(\mu_0, \Sigma_0)$ and compute the initial weights $w_0^{(i)} = 1/N$. Set initial posterior covariance $\Sigma_0^f = \Sigma_0$ for each team $f$.
2. For each observation $k = 1, \ldots, K$:
   1. For each team $f \notin E_k$, compute the posterior mean $\mu_k^f$ and covariance $\Sigma_k^f$ using the OU-transition.
   2. For each team $f \in E_k$, 
      1. Generate a $(2 \times 2) + 1$-dimensional particle from a low-discrepancy sequence $u_k^{(i)} \sim \text{LDS}(N, 2 \times 2 + 1)$.
      2. Perform Hilbert sorting on the particles $u_k^{(i)}$ to obtain the sorted particles $\tilde{u}_k^{(i)}$ and match the sorted particles to the sorted previous particles $\tilde{u}_{k-1}^{(i)}$ to obtain the indices $\sigma(i)$.
      3. Perform inverse transform sampling to obtain the particles $x_k^{E_k, (i)}$ from the proposal distribution $q(x_k^{E_k} | x_{k-1}^{E_k}) \sim \mathcal{N}(\mu_0 + \phi_k^f (x_{k-1}^f - \mu_0), (\phi_k^f)^2\Sigma_{k-1}^f + Q_{k-1}^f)$.
      4. Compute the weights $w_k^{(i)} \propto w_{k-1}^{(i)} \cdot G_k(y_k | x_k^{E_k, (i)})$.
      5. Compute the normalized weights $\tilde{w}_k^{(i)} = \frac{w_k^{(i)}}{\sum_{j=1}^N w_k^{(j)}}$.
   3. Compute posterior covariance $\Sigma_k^f = (\phi_k^f)^2\Sigma_{k-1}^f + Q_{k-1}^f$ for each team $f \in E_k$.
   4. Update the posterior mean $\mu_k^f$ and covariance $\Sigma_k^f$ for each team $f \in E_k$.
3. Return the posterior mean and covariance for each team $f$ at each time step $k$.

The difference is that SQMC performs the resampling at every step of $k$, so resampling is not needed. Given the additional sorting step from SQMC, the complexity is $\mathcal{O}(N \log N \cdot 2F \cdot K)$ instead of $\mathcal{O}(N \cdot 2F \cdot K)$ for SMC. Also has a better convergence rate than SMC ($\mathcal{O}(N^{-1})$ vs $\mathcal{O}(N^{-1/2})$).