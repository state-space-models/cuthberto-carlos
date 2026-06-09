<div align="center">
<img src="https://raw.githubusercontent.com/state-space-models/cuthberto-carlos/main/assets/cuthberto-carlos.png" alt="cuthberto-carlos logo" width="300"></img>
</div>

# cuthberto-carlos

Using [`cuthbert`](https://github.com/state-space-models/cuthbert) to predict the 2026 FIFA World Cup!


## Rough idea

The idea is to use `cuthbert`'s factorial support to use a simple model to predict
scores and results for the 2026 FIFA World Cup.

This follows the framework in [Duffield, Power and Rimella: A state-space perspective on modelling and inference for online skill rating](https://doi.org/10.1093/jrsssc/qlae035), which we can call DPR. Another useful resource is the [non-factorial example in `cuthbert`](https://state-space-models.github.io/cuthbert/quickstart/) which we are extending.

More details on factorial models can be found in DPR and the [factorial documenation in cuthbert](https://state-space-models.github.io/cuthbert/api_cuthbert/factorial/).

### The Model

The simplest model we could use is an Elo-style logistic model for the probability of a win, loss or draw. However, following 4.6 in DPR and [Karlis and Ntzoufras](https://doi.org/10.1111/1467-9884.00366) we can get more granularity by using a bivariate Poisson model to take into account the number of goals scored by each team.

The factorial state-space model  stores a time-varying latent two-dimensional state for each team (or factor) which represents its attacking and defensive strength $x_k^i = (x_k^{\text{att}, i}, x_k^{\text{def}, i})$ for team index $i$ at time $t_k$.

Pairing this with Ornstein-Uhlenbeck dynamics, the full model is then defined as follows:

$$
\begin{aligned}
p(x^i_0) &= \mathrm{N}(x^i_0 \mid \mu_0, \Sigma_0),
\\
p(x^i_k \mid x^i_{k-1}) &= \mathrm{N}\left(x^i_k \mid
\mu_0 + \phi_k (x^i_{k-1} - \mu_0),
\frac{\tau^2(1 - \phi_k^2)}{2\kappa}\right),
\\
p(y_k \mid x^i_k, x^j_k) &= \mathrm{BivariatePoisson}(y_k \mid x^i_k, x^j_k, \alpha, \beta).
\end{aligned}
$$

where $\phi_k = \exp(-\kappa(t_k - t_{k-1}))$ and $y_k = (y_k^i, y_k^j)$ is the observed number of goals scored by teams $i$ and $j$ in match $k$. The Brownian model is recovered in the limit $\kappa \to 0$, where the transition mean becomes $x^i_{k-1}$ and the transition variance becomes $\tau^2(t_k - t_{k-1})$.

The bivariate Poisson distribution is defined as follows (see 4.6 in DPR):

$$
p(y \mid x^i, x^j, \alpha, \beta) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y^i}}{y^i!} \frac{\lambda_2^{y^j}}{y^j!} \sum_{k=0}^{\min(y^i, y^j)} \binom{y^i}{k} \binom{y^j}{k} k! \left( \frac{\lambda_{3}}{\lambda_i \lambda_1} \right)^k,
$$

with $\lambda_1 = \exp(\alpha + x^{\text{att}, i} - x^{\text{def}, j})$, $\lambda_2 = \exp(\alpha + x^{\text{att}, j} - x^{\text{def}, i})$ and $\lambda_3 = \exp(\beta)$.


Overall, the static parameters of the models are $\mu_0 \in \mathbb{R}^2$, $\Sigma_0 \in \mathbb{R}^{2 \times 2}$, $\tau \in \mathbb{R}\_{> 0}^2$, $\kappa \in \mathbb{R}\_{\ge 0}^2$, $\alpha \in \mathbb{R}$ and $\beta \in \mathbb{R}$.


### Inference

Filtering in this model can be done using either the `cuthbert` SMC or `gaussian` filters combined with the `factorial` functionality.

A nice option is `cuthbert.gaussian.taylor` as in the [(non-factorial) example](https://state-space-models.github.io/cuthbert/quickstart/). Since it only requires log densities and we don't expect the true filtering distributions to be too non-Gaussian, it should work well and be faster than SMC.


### Predictions and output

The end goal will be to generate small prediction graphics prior to each match in the World Cup. These can show 3 things:

- The current latent attack and defence strengths for the two teams and their standard deviations.
- A "score grid" showing predicted probabilities for each possible scoreline (e.g. 0-0, 1-0, 0-1, 1-1, etc.) based on the bivariate Poisson likelihood (cut off at 4-4 or so).
- The predicted probabilities of a win, loss or draw for each team.

## Plan

- [x] Code to download historical data
- [ ] Implement the model in `cuthbert` and validate it works on historical data
- [ ] Determine suitable static parameters (fix some, sweep, more fancy parameter estimation?)
- [ ] Code to download/extract future fixtures
- [ ] Code to generate predictions for future fixtures
- [ ] Code to generate graphics for predictions
