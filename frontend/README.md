# Interactive World Cup frontend

The repository includes a responsive React frontend for exploring the latest dated
prediction snapshot. It combines the model output with the CC0
[`openfootball/worldcup.json`](https://github.com/openfootball/worldcup.json)
schedule to provide upcoming matches, model-projected group tables, detailed score
distributions, completed-match scorer details, and the full playoff schedule in
interactive bracket and list views. Squad metadata remains sourced from
[`openfootball/worldcup.squads.json`](https://github.com/openfootball/worldcup.json/blob/master/2026/worldcup.squads.json)
for team details used elsewhere in the application.
Upcoming fixtures also compare the model's regulation-time result probabilities
with live three-way moneyline prices from the public Polymarket Gamma API.
Playoff participants, fixture changes, and results refresh from OpenFootball using
the same five-minute request cache as group-stage results, with the generated
dataset as a deployment-time fallback. Playoff forecasts are intentionally not
generated until a future model workflow supports them.

The organization deployment is published at
[state-space-models.github.io/cuthberto-carlos](https://state-space-models.github.io/cuthberto-carlos/).

## Tournament experience

The page is a single responsive tournament explorer with four primary destinations:

- **Upcoming** combines every future group-stage and playoff fixture in chronological
  order. Group fixtures retain model and Polymarket information; unresolved playoff
  fixtures show their official qualification slots, kickoff, and venue without
  presenting a forecast that has not been generated.
- **Completed** archives group-stage forecasts after their scheduled match window and
  compares the original forecast with OpenFootball results and scorer details.
- **Groups** provides projected and actual tables plus every group fixture.
- **Playoffs** provides both a tournament bracket and a filterable list for all 32
  knockout matches from the Round of 32 through the final.

The Countries navigation and squad browser are intentionally hidden. Squad metadata
remains in the dataset because flags, canonical team names, scorer matching, and
existing match details depend on it.

## Playoff data model

The generated `TournamentDataset` uses schema version 7. Each `KnockoutMatch`
contains the stable tournament identity and schedule:

- `matchNumber` and `id` identify matches 73–104.
- `round`, `date`, `kickoffUtc`, and `venue` describe the fixture.
- `team1Slot` and `team2Slot` preserve the official feeder topology, such as `2A`,
  `3A/B/C/D/F`, `W97`, or `L101`.
- Optional `team1` and `team2` values contain resolved participants once OpenFootball
  replaces a feeder slot with a country.
- Optional `score.fullTime`, `score.extraTime`, and `score.penalties` tuples preserve
  every knockout scoring phase available from the source.

The fixed feeder topology is keyed by match number in `scripts/build_frontend_data.py`.
It is deliberately independent of mutable participant strings. This prevents a later
OpenFootball update from erasing the path that explains how a country reached a match.
The builder validates the complete 73–104 sequence, unique IDs, the expected
16/8/4/2/1/1 round counts, known participant names, non-negative score tuples, and
canonical source provenance before writing the dataset.

## Bracket and list behavior

The desktop bracket orders matches by feeder relationships rather than match number.
This ensures each pair of source matches aligns with the match it feeds. An SVG layer
measures the rendered match cards and draws all 30 Round-of-32-to-final connector
paths. Connectors are recalculated when the bracket or viewport resizes. The
third-place fixture remains separate because its losers' path does not advance to the
final.

At mobile widths the full bracket is replaced by a round selector and a compact set
of cards for the selected round. The list view remains available at every width and
supports All rounds, Round of 32, Round of 16, quarter-finals, semi-finals, third
place, and final filters. Its rows use a stable grid for match/time, participants,
score, and venue so optional result content cannot shift columns between fixtures.

## Dynamic schedule and result refresh

The generated JSON is always a complete deployment-time fallback. During a page
session, `useLiveResults` makes one shared request for the OpenFootball schedule with
a five-minute cache-bucket query parameter and browser caching disabled. The parsed
response is used for both group and playoff updates:

- Group results are matched by date and canonicalised team pair.
- Playoff fixtures are matched by stable match number so they continue matching while
  participant placeholders change into country names.
- Playoff participants, dates, kickoff times, venues, full-time scores, extra-time
  scores, and shootout scores can all update without rebuilding GitHub Pages.
- Known source aliases such as `USA` are converted to the names used by the generated
  team dataset.

Remote parsing is defensive. Malformed fixtures or score tuples reject the remote
response, leaving the generated deployment snapshot visible. OpenFootball is a
community-maintained source rather than a real-time score service, so the footer
reports when the schedule was checked without promising instantaneous updates.

## Accessibility and responsive behavior

View and round selectors expose native buttons with `aria-pressed`. Bracket and list
regions have descriptive labels, all kickoff times include the visitor's local
timezone, and status or winner information is not communicated by colour alone.
Desktop bracket overflow stays within its labelled region; the page itself does not
gain horizontal overflow. Mobile cards and list rows collapse to single-column
participant layouts.

The prediction-date control uses a native `<select>`. Tests assert its selected
`<option>` through the DOM `selected` property; `aria-pressed` is intentionally not
added because it is invalid for native option elements.

## Local development and verification

Generate the latest frontend dataset and run the app locally:

```bash
python3 scripts/build_frontend_data.py
cd frontend
npm install
npm run dev
```

Run the complete frontend verification used for this feature:

```bash
python3 -m unittest discover -s tests -p 'test_*frontend_data.py'
python3 scripts/build_frontend_data.py
cd frontend
npm test -- --run
npm run build
```

The tests cover generated topology and score validation, shared OpenFootball request
caching, participant resolution, extra time and penalties, fallback behavior,
combined future fixtures, bracket/list filtering, navigation visibility, and native
prediction-date semantics. Responsive browser checks should additionally cover a
normal desktop viewport and a narrow mobile viewport, including horizontal overflow,
connector count, list-column alignment, and browser console errors.

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

The data builder also pages through Polymarket's World Cup event keyset feed and
stores complete home-win/draw/away-win market groups as deployment-time fallback
data. The frontend refreshes those prices directly from Polymarket in the same
five-minute cache window. Market data is rendered only while a fixture kickoff is
still in the future; ongoing and completed matches never show Polymarket prices.

The production build exposes the exact generated dataset at
`/cuthberto-carlos/data/tournament.json`. The data builder validates the generated
schema, match counts, probabilities, source provenance, and canonical repository
URLs before writing the file. It merges every ISO-dated directory under
`outputs/predictions`, uses the newest available prediction for each fixture, and
retains older prediction values and source links for interactive comparisons:

```bash
python3 -m unittest discover -s tests -p 'test_*frontend_data.py'
python3 scripts/build_frontend_data.py
```

The deployment workflow also compares the generated source file with the copy in
`frontend/dist/data/tournament.json` byte-for-byte before publishing.

Repository administrators must configure **Settings > Pages > Build and deployment >
Source** to **GitHub Actions**. The organization must allow Pages publication and the
`github-pages` environment must permit automated deployments from `main` without an
unintended approval gate.
