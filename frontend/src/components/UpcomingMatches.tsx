import { useState } from "react";
import type { KnockoutMatch, MatchPrediction, Team } from "../types";
import {
  describeBracketSlot,
  formatKickoffParts,
  formatPercent,
  getOngoingMatches,
  getUpcomingMatches,
  mostLikelyOutcome,
  ROUND_LABELS,
} from "../utils";
import { flagClassNames } from "../flags";
import { MatchCard } from "./MatchCard";
import { MatchListRow } from "./MatchListRow";
import { PolymarketCardComparison } from "./PolymarketComparison";

interface UpcomingMatchesProps {
  matches: MatchPrediction[];
  knockoutMatches: KnockoutMatch[];
  teams: Record<string, Team>;
  onOpen: (match: MatchPrediction | KnockoutMatch, trigger: HTMLElement) => void;
}

type UpcomingEntry =
  | { kind: "group"; match: MatchPrediction }
  | { kind: "playoff"; match: KnockoutMatch };

function PlayoffFlag({ name, slot, teams }: {
  name?: string;
  slot: string;
  teams: Record<string, Team>;
}) {
  if (name && teams[name]) {
    const flagClass = flagClassNames[teams[name].flagCode] || "fi fi-unknown";
    return (
      <span className="upcoming-playoff-flag" title={name}>
        <span className={`team-flag ${flagClass}`} aria-hidden="true" />
        <span className="sr-only">{name}</span>
      </span>
    );
  }
  return (
    <span className="upcoming-playoff-participant">
      <strong>{name ?? slot}</strong>
      <small>{describeBracketSlot(slot)}</small>
    </span>
  );
}

function UpcomingPlayoffCard({ match, teams, onOpen }: { match: KnockoutMatch; teams: Record<string, Team>; onOpen: (match: KnockoutMatch, trigger: HTMLElement) => void }) {
  const kickoff = formatKickoffParts(match.kickoffUtc);
  const prediction = match.prediction;
  const team1 = match.team1;
  const team2 = match.team2;
  const clickable = !!prediction;
  return (
    <article
      className={`match-card upcoming-playoff-card${clickable ? " upcoming-playoff-card--clickable" : ""}`}
      {...(clickable ? {
        onClick: (event: React.MouseEvent) => onOpen(match, event.currentTarget as HTMLElement),
        tabIndex: 0,
        onKeyDown: (event: React.KeyboardEvent) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onOpen(match, event.currentTarget as HTMLElement);
          }
        },
        "aria-label": `Explore prediction for ${team1 ?? match.team1Slot} versus ${team2 ?? match.team2Slot}`,
      } : {})}
    >
      <div className="match-card__meta">
        <span className="eyebrow">{ROUND_LABELS[match.round]} · M{match.matchNumber}</span>
        <span>{kickoff.date}</span>
        <strong>{kickoff.time}</strong>
      </div>
      <div className="match-card__teams">
        <div className="match-card__team match-card__team--first">
          <PlayoffFlag name={team1} slot={match.team1Slot} teams={teams} />
        </div>
        {prediction ? (
          <div className="match-card__score-block">
            <span className="match-card__outcome">
              {mostLikelyOutcome(prediction.probabilities, team1 ?? match.team1Slot, team2 ?? match.team2Slot)}
            </span>
            <span className="match-card__score" aria-label="Most likely score">
              {prediction.mostLikelyScore[0]}–{prediction.mostLikelyScore[1]}
            </span>
          </div>
        ) : (
          <strong className="upcoming-playoff-card__versus">vs</strong>
        )}
        <div className="match-card__team match-card__team--second">
          <PlayoffFlag name={team2} slot={match.team2Slot} teams={teams} />
        </div>
      </div>
      <p className="match-card__venue">{match.venue}</p>
      {prediction && (
        <PolymarketCardComparison match={{ ...match, homeTeam: team1 ?? match.team1Slot, awayTeam: team2 ?? match.team2Slot }} />
      )}
    </article>
  );
}

function UpcomingPlayoffRow({ match, teams, onOpen }: { match: KnockoutMatch; teams: Record<string, Team>; onOpen: (match: KnockoutMatch, trigger: HTMLElement) => void }) {
  const kickoff = formatKickoffParts(match.kickoffUtc);
  const prediction = match.prediction;
  const team1 = match.team1;
  const team2 = match.team2;
  const clickable = !!prediction;
  return (
    <article
      className={`match-list-row upcoming-playoff-row${clickable ? " upcoming-playoff-row--clickable" : ""}`}
      {...(clickable ? {
        onClick: (event: React.MouseEvent) => onOpen(match, event.currentTarget as HTMLElement),
        tabIndex: 0,
        onKeyDown: (event: React.KeyboardEvent) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onOpen(match, event.currentTarget as HTMLElement);
          }
        },
        "aria-label": `Explore prediction for ${team1 ?? match.team1Slot} versus ${team2 ?? match.team2Slot}`,
      } : {})}
    >
      <div className="match-list-row__kickoff">
        <span className="eyebrow">{ROUND_LABELS[match.round]} · M{match.matchNumber}</span>
        <strong>{kickoff.date}</strong>
        <span>{kickoff.time}</span>
      </div>
      <div className="upcoming-playoff-row__fixture">
        <PlayoffFlag name={team1} slot={match.team1Slot} teams={teams} />
        {prediction ? (
          <span className="match-list-row__score" aria-label="Most likely score">
            {prediction.mostLikelyScore[0]}–{prediction.mostLikelyScore[1]}
          </span>
        ) : (
          <strong>vs</strong>
        )}
        <PlayoffFlag name={team2} slot={match.team2Slot} teams={teams} />
      </div>
      <span className="match-list-row__venue">{match.venue}</span>
      {prediction && (
        <PolymarketCardComparison match={{ ...match, homeTeam: team1 ?? match.team1Slot, awayTeam: team2 ?? match.team2Slot }} />
      )}
      {!prediction && (
        <strong className="upcoming-playoff-row__stage">Playoff</strong>
      )}
    </article>
  );
}

export function UpcomingMatches({ matches, knockoutMatches, teams, onOpen }: UpcomingMatchesProps) {
  const [view, setView] = useState<"cards" | "list">("cards");
  const entries: UpcomingEntry[] = [
    ...getOngoingMatches(matches).map((match): UpcomingEntry => ({ kind: "group", match })),
    ...getUpcomingMatches(matches).map((match): UpcomingEntry => ({ kind: "group", match })),
    ...knockoutMatches
      .filter((match) => new Date(match.kickoffUtc).getTime() > Date.now())
      .map((match): UpcomingEntry => ({ kind: "playoff", match })),
  ].sort((left, right) =>
    new Date(left.match.kickoffUtc).getTime() - new Date(right.match.kickoffUtc).getTime(),
  );

  return (
    <section className="section section--upcoming" id="upcoming" aria-labelledby="upcoming-title">
      <div className="section-heading">
        <div><span className="eyebrow">Next up</span><h2 id="upcoming-title">Upcoming matches</h2></div>
        <div className="section-heading__tools">
          <p>Every live and future group and playoff fixture, shown in your local timezone.</p>
          <div className="upcoming-view-controls">
            <span className="upcoming-match-count">{entries.length} matches</span>
            <div className="view-toggle" role="group" aria-label="Upcoming matches view">
              <button type="button" className="view-toggle__button" aria-pressed={view === "cards"} onClick={() => setView("cards")}>
                <span className="view-toggle__icon view-toggle__icon--cards" aria-hidden="true" />Cards
              </button>
              <button type="button" className="view-toggle__button" aria-pressed={view === "list"} onClick={() => setView("list")}>
                <span className="view-toggle__icon view-toggle__icon--list" aria-hidden="true" />List
              </button>
            </div>
          </div>
        </div>
      </div>
      {entries.length > 0 ? (
        <div className={view === "cards" ? "match-grid upcoming-matches-scroll" : "match-list upcoming-matches-scroll"}
          data-testid={view === "cards" ? "upcoming-card-view" : "upcoming-list-view"}
          role="region" aria-label={`All upcoming matches in ${view === "cards" ? "card" : "list"} view`} tabIndex={0}>
          {entries.map((entry) => entry.kind === "group"
            ? view === "cards"
              ? <MatchCard key={`group-${entry.match.id}`} match={entry.match} teams={teams} onOpen={onOpen} showPolymarket />
              : <MatchListRow key={`group-${entry.match.id}`} match={entry.match} teams={teams} onOpen={onOpen} showPolymarket />
            : view === "cards"
              ? <UpcomingPlayoffCard key={`playoff-${entry.match.id}`} match={entry.match} teams={teams} onOpen={onOpen} />
              : <UpcomingPlayoffRow key={`playoff-${entry.match.id}`} match={entry.match} teams={teams} onOpen={onOpen} />)}
        </div>
      ) : (
        <div className="empty-state"><strong>No future fixtures remain.</strong><span>Browse completed matches and the playoff archive below.</span></div>
      )}
    </section>
  );
}
