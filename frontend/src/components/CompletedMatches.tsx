import { useState } from "react";
import type { KnockoutMatch, MatchPrediction, Team } from "../types";
import { getCompletedMatches, getCompletedKnockoutMatches } from "../utils";
import { MatchCard } from "./MatchCard";
import { MatchListRow } from "./MatchListRow";
import { KnockoutMatchCard } from "./KnockoutMatchCard";
import { KnockoutMatchListRow } from "./KnockoutMatchListRow";

interface CompletedMatchesProps {
  matches: MatchPrediction[];
  knockoutMatches?: KnockoutMatch[];
  teams: Record<string, Team>;
  onOpen: (match: MatchPrediction | KnockoutMatch, trigger: HTMLElement) => void;
}

export function CompletedMatches({ matches, knockoutMatches = [], teams, onOpen }: CompletedMatchesProps) {
  const completedGroup = getCompletedMatches(matches);
  const completedKnockout = getCompletedKnockoutMatches(knockoutMatches);
  
  // Combine and sort by kickoff time (newest first)
  const completed = [
    ...completedGroup.map(m => ({ ...m, _type: 'group' as const })),
    ...completedKnockout.map(m => ({ ...m, _type: 'knockout' as const })),
  ].sort((a, b) => 
    new Date(b.kickoffUtc).getTime() - new Date(a.kickoffUtc).getTime()
  );
  
  const [view, setView] = useState<"cards" | "list">("list");

  return (
    <section className="section section--completed" id="completed" aria-labelledby="completed-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Forecast archive</span>
          <h2 id="completed-title">Completed matches</h2>
        </div>
        <div className="section-heading__tools">
          <p>
            Original pre-match forecasts, newest first. Fixtures move here two hours after scheduled kickoff.
          </p>
          <div className="view-toggle" role="group" aria-label="Completed matches view">
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
      {completed.length > 0 ? (
        view === "cards" ? (
          <div className="match-grid" data-testid="completed-card-view">
            {completed.map((match) =>
              match._type === 'group' ? (
                <MatchCard
                  key={match.id}
                  match={match}
                  teams={teams}
                  onOpen={onOpen}
                  showScoreComparison
                  hidePredictionDetails
                />
              ) : (
                <KnockoutMatchCard
                  key={match.id}
                  match={match}
                  teams={teams}
                  onOpen={onOpen}
                  showScoreComparison
                  hidePredictionDetails
                />
              )
            )}
          </div>
        ) : (
          <div className="match-list" data-testid="completed-list-view">
            {completed.map((match) =>
              match._type === 'group' ? (
                <MatchListRow
                  key={match.id}
                  match={match}
                  teams={teams}
                  onOpen={onOpen}
                  showScoreComparison
                  hidePredictionDetails
                />
              ) : (
                <KnockoutMatchListRow
                  key={match.id}
                  match={match}
                  teams={teams}
                  onOpen={onOpen}
                  showScoreComparison
                  hidePredictionDetails
                />
              )
            )}
          </div>
        )
      ) : (
        <div className="empty-state">
          <strong>No completed fixtures yet.</strong>
          <span>The original predictions will appear here after matches finish.</span>
        </div>
      )}
    </section>
  );
}
