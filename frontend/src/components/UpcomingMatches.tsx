import { useState } from "react";
import type { KnockoutMatch, MatchPrediction, Team } from "../types";
import {
  describeBracketSlot,
  formatKickoffParts,
  getOngoingMatches,
  getUpcomingMatches,
  ROUND_LABELS,
} from "../utils";
import { MatchCard } from "./MatchCard";
import { MatchListRow } from "./MatchListRow";
import { TeamFlag } from "./TeamFlag";

interface UpcomingMatchesProps {
  matches: MatchPrediction[];
  knockoutMatches: KnockoutMatch[];
  teams: Record<string, Team>;
  onOpen: (match: MatchPrediction, trigger: HTMLElement) => void;
}

type UpcomingEntry =
  | { kind: "group"; match: MatchPrediction }
  | { kind: "playoff"; match: KnockoutMatch };

function PlayoffParticipant({ name, slot, teams }: {
  name?: string;
  slot: string;
  teams: Record<string, Team>;
}) {
  return name && teams[name] ? (
    <TeamFlag team={teams[name]} compact />
  ) : (
    <span className="upcoming-playoff-participant">
      <strong>{name ?? slot}</strong>
      <small>{describeBracketSlot(slot)}</small>
    </span>
  );
}

function UpcomingPlayoffCard({ match, teams }: { match: KnockoutMatch; teams: Record<string, Team> }) {
  const kickoff = formatKickoffParts(match.kickoffUtc);
  return (
    <article className="match-card upcoming-playoff-card">
      <div className="match-card__meta">
        <span className="eyebrow">{ROUND_LABELS[match.round]} · M{match.matchNumber}</span>
        <span>{kickoff.date}</span>
        <strong>{kickoff.time}</strong>
      </div>
      <div className="match-card__teams">
        <div className="match-card__team match-card__team--first">
          <PlayoffParticipant name={match.team1} slot={match.team1Slot} teams={teams} />
        </div>
        <strong className="upcoming-playoff-card__versus">vs</strong>
        <div className="match-card__team match-card__team--second">
          <PlayoffParticipant name={match.team2} slot={match.team2Slot} teams={teams} />
        </div>
      </div>
      <p className="match-card__venue">{match.venue}</p>
      <div className="match-card__footer"><span>Playoff fixture</span></div>
    </article>
  );
}

function UpcomingPlayoffRow({ match, teams }: { match: KnockoutMatch; teams: Record<string, Team> }) {
  const kickoff = formatKickoffParts(match.kickoffUtc);
  return (
    <article className="match-list-row upcoming-playoff-row">
      <div className="match-list-row__kickoff">
        <span className="eyebrow">{ROUND_LABELS[match.round]} · M{match.matchNumber}</span>
        <strong>{kickoff.date}</strong>
        <span>{kickoff.time}</span>
      </div>
      <div className="upcoming-playoff-row__fixture">
        <PlayoffParticipant name={match.team1} slot={match.team1Slot} teams={teams} />
        <strong>vs</strong>
        <PlayoffParticipant name={match.team2} slot={match.team2Slot} teams={teams} />
      </div>
      <span className="match-list-row__venue">{match.venue}</span>
      <strong className="upcoming-playoff-row__stage">Playoff</strong>
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
              ? <UpcomingPlayoffCard key={`playoff-${entry.match.id}`} match={entry.match} teams={teams} />
              : <UpcomingPlayoffRow key={`playoff-${entry.match.id}`} match={entry.match} teams={teams} />)}
        </div>
      ) : (
        <div className="empty-state"><strong>No future fixtures remain.</strong><span>Browse completed matches and the playoff archive below.</span></div>
      )}
    </section>
  );
}
