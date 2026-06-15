<div align="center">
<img src="https://raw.githubusercontent.com/state-space-models/cuthberto-carlos/main/assets/cuthberto-carlos.png" alt="cuthberto-carlos logo" width="300"></img>
</div>

# cuthberto-carlos

Using [`cuthbert`](https://github.com/state-space-models/cuthbert) to predict the 2026 FIFA World Cup!

Check out the frontend for the predictions here: [https://state-space-models.github.io/cuthberto-carlos/](https://state-space-models.github.io/cuthberto-carlos/)

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
Q_k\right),
\\
p(y_k \mid x^i_k, x^j_k) &= \mathrm{BivariatePoisson}(y_k \mid x^i_k, x^j_k, \alpha, \beta).
\end{aligned}
$$

where $\phi_k = \exp(-\kappa(t_k - t_{k-1}))$ and $y_k = (y_k^i, y_k^j)$ is the observed number of goals scored by teams $i$ and $j$ in match $k$. In the moment-based Gaussian implementation, $Q_k$ is chosen so that the long-run marginal distribution is $\mathrm{N}(\mu_0, \Sigma_0)$. For diagonal mean-reversion rates $\kappa = (\kappa_{\text{att}}, \kappa_{\text{def}})$, writing $\Phi_k = \mathrm{diag}(\phi_k)$ gives

$$
Q_k = \Sigma_0 - \Phi_k \Sigma_0 \Phi_k.
$$

This allows the attack and defence components to revert jointly to the initial covariance, including any attack-defence correlation encoded in $\Sigma_0$.

The bivariate Poisson distribution is defined as follows (see 4.6 in DPR):

$$
p(y \mid x^i, x^j, \alpha, \beta) = e^{-(\lambda_1 + \lambda_2 + \lambda_3)} \frac{\lambda_1^{y^i}}{y^i!} \frac{\lambda_2^{y^j}}{y^j!} \sum_{k=0}^{\min(y^i, y^j)} \binom{y^i}{k} \binom{y^j}{k} k! \left( \frac{\lambda_{3}}{\lambda_i \lambda_1} \right)^k,
$$

with $\lambda_1 = \exp(\alpha + x^{\text{att}, i} - x^{\text{def}, j})$, $\lambda_2 = \exp(\alpha + x^{\text{att}, j} - x^{\text{def}, i})$ and $\lambda_3 = \exp(\beta)$.


Overall, the static parameters of the models are $\mu_0 \in \mathbb{R}^2$, $\Sigma_0 \in \mathbb{R}^{2 \times 2}$, $\kappa \in \mathbb{R}\_{\ge 0}^2$, $\alpha \in \mathbb{R}$ and $\beta \in \mathbb{R}$.


### Inference

Filtering in this model can be done using either the `cuthbert` SMC or `gaussian` filters combined with the `factorial` functionality.

### Predictions and output

The end goal will be to generate small prediction graphics prior to each match in the World Cup. These can show 3 things:

- The current latent attack and defence strengths for the two teams and their standard deviations.
- A "score grid" showing predicted probabilities for each possible scoreline (e.g. 0-0, 1-0, 0-1, 1-1, etc.) based on the bivariate Poisson likelihood (cut off at 4-4 or so).
- The predicted probabilities of a win, loss or draw for each team.

## Plan

- [x] Code to download historical data
- [x] Implement the model in `cuthbert` and validate it works on historical data
- [x] Determine suitable static parameters (fix some, sweep, more fancy parameter estimation?)
- [x] Code to download/extract future fixtures
- [x] Code to generate predictions for future fixtures
- [x] Code to generate graphics for predictions

## Interactive World Cup frontend

The repository includes a responsive React frontend for exploring the latest dated
prediction snapshot. It combines the model output with the CC0
[`openfootball/worldcup.json`](https://github.com/openfootball/worldcup.json)
schedule to provide upcoming matches, model-projected group tables, detailed score
distributions, completed-match scorer details, and an interactive country squad
browser sourced from
[`openfootball/worldcup.squads.json`](https://github.com/openfootball/worldcup.json/blob/master/2026/worldcup.squads.json).
Player names expose accessible squad profiles on hover, keyboard focus, and touch.
Knockout fixture metadata remains in
the generated dataset for future frontend work, but the current UI is limited to the
group stage.

The organization deployment is published at
[state-space-models.github.io/cuthberto-carlos](https://state-space-models.github.io/cuthberto-carlos/).

Generate the latest frontend dataset and run the app locally:

```bash
python scripts/build_frontend_data.py
cd frontend
npm install
npm run dev
```

The GitHub Pages workflow validates pull requests without deploying. Pushes to
`main`, hourly schedules, and manual runs build with a base path derived from the
repository name and deploy the validated artifact. Every scheduled run fetches the
current OpenFootball schedule and squad files with cache bypass headers, replaces
the generated tournament dataset in the workflow workspace, reports squad and
completed-result counts, and publishes that refreshed copy to Pages. The workflow
does not create an hourly commit on `main`.

The production build exposes the exact generated dataset at
`/cuthberto-carlos/data/tournament.json`. The data builder validates the generated
schema, match counts, probabilities, source provenance, and canonical repository
URLs before writing the file. It merges every ISO-dated directory under
`outputs/predictions`, uses the newest available prediction for each fixture, and
retains older prediction folders as source links in that fixture's history:

```bash
python -m unittest discover -s tests -p 'test_*frontend_data.py'
python scripts/build_frontend_data.py
```

The deployment workflow also compares the generated source file with the copy in
`frontend/dist/data/tournament.json` byte-for-byte before publishing.

Repository administrators must configure **Settings > Pages > Build and deployment >
Source** to **GitHub Actions**. The organization must allow Pages publication and the
`github-pages` environment must permit automated deployments from `main` without an
unintended approval gate.
