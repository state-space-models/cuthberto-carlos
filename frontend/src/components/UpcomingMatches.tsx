import { useState } from "react";
import type { MatchPrediction, Team } from "../types";
import { getUpcomingMatches, getOngoingMatches } from "../utils";
import { MatchCard } from "./MatchCard";
import { MatchListRow } from "./MatchListRow";

interface UpcomingMatchesProps {
  matches: MatchPrediction[];
  teams: Record<string, Team>;
  onOpen: (match: MatchPrediction, trigger: HTMLElement) => void;
}

export function UpcomingMatches({ matches, teams, onOpen }: UpcomingMatchesProps) {
  const upcoming = getUpcomingMatches(matches);
  const ongoing = getOngoingMatches(matches);
  const [view, setView] = useState<"cards" | "list">("cards");

  const displayMatches = [...ongoing, ...upcoming];

  return (
    <section className="section section--upcoming" id="upcoming" aria-labelledby="upcoming-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Next up</span>
          <h2 id="upcoming-title">Upcoming matches</h2>
        </div>
        <div className="section-heading__tools">
          <p>Kickoff times automatically use your local timezone.</p>
          <div className="view-toggle" role="group" aria-label="Upcoming matches view">
            <button
              type="button"
              className="view-toggle__button"
              aria-pressed={view === "cards"}
              onClick={() => setView("cards")}
            >
              <span className="view-toggle__icon view-toggle__icon--cards" aria-hidden="true" />
              Cards
            </button>
            <button
              type="button"
              className="view-toggle__button"
              aria-pressed={view === "list"}
              onClick={() => setView("list")}
            >
              <span className="view-toggle__icon view-toggle__icon--list" aria-hidden="true" />
              List
            </button>
          </div>
        </div>
      </div>
      {displayMatches.length > 0 ? (
        view === "cards" ? (
          <div className="match-grid" data-testid="upcoming-card-view">
            {displayMatches.map((match) => (
              <MatchCard key={match.id} match={match} teams={teams} onOpen={onOpen} />
            ))}
          </div>
        ) : (
          <div className="match-list" data-testid="upcoming-list-view">
            {displayMatches.map((match) => (
              <MatchListRow key={match.id} match={match} teams={teams} onOpen={onOpen} />
            ))}
          </div>
        )
      ) : (
        <div className="empty-state">
          <strong>No upcoming group fixtures remain.</strong>
          <span>Browse the group-stage archive and its original predictions below.</span>
        </div>
      )}
    </section>
  );
}
