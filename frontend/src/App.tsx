import { useCallback, useRef, useState } from "react";
import tournamentData from "./data/tournament.json";
import { CompletedMatches } from "./components/CompletedMatches";
import { Countries } from "./components/Countries";
import { GroupStage } from "./components/GroupStage";
// import { KnockoutBracket } from "./components/KnockoutBracket";
import { MatchDetailDrawer } from "./components/MatchDetailDrawer";
import { UpcomingMatches } from "./components/UpcomingMatches";
import type { MatchPrediction, TournamentDataset } from "./types";
import { useLiveResults } from "./useLiveResults";

const data = tournamentData as unknown as TournamentDataset;

const predictionDates = Array.from(
  new Set(
    data.groupMatches.flatMap((match) => [
      match.predictionDate,
      ...match.predictionHistory.map((prediction) => prediction.predictionDate),
    ]),
  ),
).sort((left, right) => right.localeCompare(left));

const snapshotBaseUrl = data.snapshotUrl.slice(0, -data.snapshotDate.length);

function App() {
  const { matches, status, lastCheckedAt } = useLiveResults(
    data.groupMatches,
    data.sources.schedule.dataUrl,
    data.teams,
  );
  const [selectedMatchId, setSelectedMatchId] = useState<string | null>(null);
  const lastTrigger = useRef<HTMLElement | null>(null);
  const selectedMatch = selectedMatchId
    ? matches.find((match) => match.id === selectedMatchId) ?? null
    : null;

  const openMatch = useCallback((match: MatchPrediction, trigger: HTMLElement) => {
    lastTrigger.current = trigger;
    setSelectedMatchId(match.id);
  }, []);

  const closeMatch = useCallback(() => {
    setSelectedMatchId(null);
    window.requestAnimationFrame(() => lastTrigger.current?.focus());
  }, []);

  return (
    <>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <header className="site-header">
        <a className="site-brand" href="#top" aria-label="Cuthberto Carlos home">
          <span className="site-brand__mark">CC</span>
          <span>Cuthberto Carlos</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#upcoming">Upcoming</a>
          <a href="#completed">Completed</a>
          <a href="#groups">Groups</a>
          <a href="#countries">Countries</a>
          {/* <a href="#playoffs">Playoffs</a> */}
        </nav>
        <div className="header-actions">
          <div className="header-snapshot">
            <a className="header-snapshot__latest" href={data.snapshotUrl} target="_blank" rel="noreferrer">
              <span>Latest model snapshot</span>
              <strong>{data.snapshotDate}</strong>
              <small>Commit {data.sourceCommit}</small>
            </a>
            {predictionDates.length > 1 && (
              <details className="header-snapshot__picker">
                <summary aria-label="Browse previous predictions">⌄</summary>
                <div className="header-snapshot__menu">
                  <strong>Previous predictions</strong>
                  {predictionDates.slice(1).map((predictionDate) => (
                    <a
                      key={predictionDate}
                      href={`${snapshotBaseUrl}${predictionDate}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {predictionDate} <span aria-hidden="true">↗</span>
                    </a>
                  ))}
                </div>
              </details>
            )}
          </div>
          <a className="header-github" href={data.repositoryUrl} target="_blank" rel="noreferrer">
            View model <span aria-hidden="true">↗</span>
          </a>
        </div>
      </header>

      <main id="main-content">
        <section className="hero" id="top">
          <div className="hero__content">
            <span className="hero__kicker">2026 World Cup · model forecast</span>
            <h1>Every group match.<br /><em>One curious caterpillar.</em></h1>
            <p>
              Explore score distributions, result probabilities, and projected group tables for every group-stage fixture.
            </p>
            <div className="hero__actions">
              <a className="button button--primary" href="#upcoming">See upcoming matches</a>
              <a className="button button--secondary" href="#groups">Explore all groups</a>
            </div>
            <dl className="hero__stats">
              <div><dt>72</dt><dd>match forecasts</dd></div>
              <div><dt>12</dt><dd>group projections</dd></div>
              <div><dt>{Object.keys(data.teams).length}</dt><dd>national teams</dd></div>
            </dl>
          </div>
          <div className="hero__art" aria-hidden="true">
            <span className="hero__orbit hero__orbit--one" />
            <span className="hero__orbit hero__orbit--two" />
            <img src={`${import.meta.env.BASE_URL}cuthberto-carlos.png`} alt="" />
          </div>
        </section>

        <UpcomingMatches matches={matches} teams={data.teams} onOpen={openMatch} />
        <CompletedMatches matches={matches} teams={data.teams} onOpen={openMatch} />
        <GroupStage groups={data.groups} matches={matches} teams={data.teams} onOpen={openMatch} />
        <Countries teams={data.teams} />
        {/* <KnockoutBracket matches={data.knockoutMatches} /> */}
      </main>

      <footer className="site-footer">
        <div>
          <strong>Cuthberto Carlos</strong>
          <span>A state-space model predicting the 2026 World Cup.</span>
        </div>
        <div className="site-footer__sources">
          <span>Open data:</span>
          <a href={data.sources.schedule.url} target="_blank" rel="noreferrer">openfootball schedule (CC0)</a>
          <a href={data.sources.squads.url} target="_blank" rel="noreferrer">openfootball squads (CC0)</a>
          <a href={data.sources.historicalResults.url} target="_blank" rel="noreferrer">international results (CC0)</a>
          <a href="https://github.com/lipis/flag-icons" target="_blank" rel="noreferrer">flag-icons (MIT)</a>
        </div>
        <p>Probabilities are model estimates, not betting advice or official FIFA data.</p>
        <p className="site-footer__results-status" role="status">
          {status === "loading" && "Checking OpenFootball for current results…"}
          {status === "current" && `Results checked ${new Date(lastCheckedAt!).toLocaleTimeString()}.`}
          {status === "fallback" && "Showing deployment-time results; the live source is unavailable."}
        </p>
      </footer>

      <MatchDetailDrawer
        match={selectedMatch}
        matches={matches}
        teams={data.teams}
        modelName={data.model.name}
        onClose={closeMatch}
      />
    </>
  );
}

export default App;
