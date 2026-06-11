import { useState } from "react";
import type { MatchPrediction, Team } from "../types";
import { getOngoingMatches } from "../utils";
import { MatchCard } from "./MatchCard";
import { MatchListRow } from "./MatchListRow";

interface OngoingMatchesProps {
  matches: MatchPrediction[];
  teams: Record<string, Team>;
  onOpen: (match: MatchPrediction, trigger: HTMLElement) => void;
}

export function OngoingMatches({ matches, teams, onOpen }: OngoingMatchesProps) {
  const ongoing = getOngoingMatches(matches);
  const [view, setView] = useState<"cards" | "list">("cards");

  if (ongoing.length === 0) {
    return null;
  }

  return (
    <section className="section section--ongoing" id="ongoing" aria-labelledby="ongoing-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Live now</span>
          <h2 id="ongoing-title">Ongoing matches</h2>
        </div>
        <div className="section-heading__tools">
          <p>Matches currently in progress.</p>
          <div className="view-toggle" role="group" aria-label="Ongoing matches view">
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
      {view === "cards" ? (
        <div className="match-grid" data-testid="ongoing-card-view">
          {ongoing.map((match) => (
            <MatchCard key={match.id} match={match} teams={teams} onOpen={onOpen} />
          ))}
        </div>
      ) : (
        <div className="match-list" data-testid="ongoing-list-view">
          {ongoing.map((match) => (
            <MatchListRow key={match.id} match={match} teams={teams} onOpen={onOpen} />
          ))}
        </div>
      )}
    </section>
  );
}
