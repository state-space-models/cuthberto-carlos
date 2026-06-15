# Interactive World Cup frontend

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
`main`, manual runs, and `predictions-updated` repository dispatches build with a
base path derived from the repository name and deploy the validated artifact. The
daily prediction workflow sends that dispatch only after it commits and pushes new
prediction output; unchanged runs do not redeploy the site.

Predictions, fixtures, squads, and deployment-time fallback results are compiled
into the static site. On each page session, the frontend fetches the current
OpenFootball schedule directly from `raw.githubusercontent.com`, using a shared
five-minute cache bucket, and merges completed scores and scorers into the static
predictions. If the request or schema validation fails, the deployed fallback
results remain available. Match-result changes therefore do not rebuild Pages.

The production build exposes the exact generated dataset at
`/cuthberto-carlos/data/tournament.json`. The data builder validates the generated
schema, match counts, probabilities, source provenance, and canonical repository
URLs before writing the file. It merges every ISO-dated directory under
`outputs/predictions`, uses the newest available prediction for each fixture, and
retains older prediction values and source links for interactive comparisons:

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