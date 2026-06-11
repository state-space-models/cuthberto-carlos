import { useCallback, useRef, useState } from "react";
import tournamentData from "./data/tournament.json";
import { CompletedMatches } from "./components/CompletedMatches";
import { GroupStage } from "./components/GroupStage";
// import { KnockoutBracket } from "./components/KnockoutBracket";
import { MatchDetailDrawer } from "./components/MatchDetailDrawer";
import { UpcomingMatches } from "./components/UpcomingMatches";
import type { MatchPrediction, TournamentDataset } from "./types";

const data = tournamentData as unknown as TournamentDataset;

function App() {
  const [selectedMatch, setSelectedMatch] = useState<MatchPrediction | null>(null);
  const lastTrigger = useRef<HTMLElement | null>(null);

  const openMatch = useCallback((match: MatchPrediction, trigger: HTMLElement) => {
    lastTrigger.current = trigger;
    setSelectedMatch(match);
  }, []);

  const closeMatch = useCallback(() => {
    setSelectedMatch(null);
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
          {/* <a href="#playoffs">Playoffs</a> */}
        </nav>
        <div className="header-actions">
          <a className="header-snapshot" href={data.snapshotUrl} target="_blank" rel="noreferrer">
            <span>Latest model snapshot</span>
            <strong>{data.snapshotDate}</strong>
            <small>Commit {data.sourceCommit}</small>
          </a>
          <a className="header-github" href="https://github.com/ryantjx/cuthberto-carlos" target="_blank" rel="noreferrer">
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
              Explore score distributions, result probabilities, projected group tables, and the complete path to the final.
            </p>
            <div className="hero__actions">
              <a className="button button--primary" href="#upcoming">See upcoming matches</a>
              <a className="button button--secondary" href="#groups">Explore all groups</a>
            </div>
            <dl className="hero__stats">
              <div><dt>72</dt><dd>match forecasts</dd></div>
              <div><dt>12</dt><dd>group projections</dd></div>
              <div><dt>32</dt><dd>finals fixtures</dd></div>
            </dl>
          </div>
          <div className="hero__art" aria-hidden="true">
            <span className="hero__orbit hero__orbit--one" />
            <span className="hero__orbit hero__orbit--two" />
            <img src={`${import.meta.env.BASE_URL}cuthberto-carlos.png`} alt="" />
          </div>
        </section>

        <UpcomingMatches matches={data.groupMatches} teams={data.teams} onOpen={openMatch} />
        <CompletedMatches matches={data.groupMatches} teams={data.teams} onOpen={openMatch} />
        <GroupStage groups={data.groups} matches={data.groupMatches} teams={data.teams} onOpen={openMatch} />
        {/* <KnockoutBracket matches={data.knockoutMatches} /> */}
      </main>

      <footer className="site-footer">
        <div>
          <strong>Cuthberto Carlos</strong>
          <span>A state-space model predicting the 2026 World Cup.</span>
        </div>
        <div className="site-footer__sources">
          <span>Open data:</span>
          <a href="https://github.com/openfootball/worldcup.json" target="_blank" rel="noreferrer">openfootball schedule (CC0)</a>
          <a href={data.sources.historicalResults.url} target="_blank" rel="noreferrer">international results (CC0)</a>
          <a href="https://github.com/lipis/flag-icons" target="_blank" rel="noreferrer">flag-icons (MIT)</a>
        </div>
        <p>Probabilities are model estimates, not betting advice or official FIFA data.</p>
      </footer>

      <MatchDetailDrawer
        match={selectedMatch}
        teams={data.teams}
        snapshotDate={data.snapshotDate}
        modelName={data.model.name}
        onClose={closeMatch}
      />
    </>
  );
}

export default App;
