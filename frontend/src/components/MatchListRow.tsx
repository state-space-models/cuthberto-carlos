import type { MouseEvent } from "react";
import type { MatchPrediction, Team } from "../types";
import { formatKickoffParts, formatPercent, isMatchOngoing, mostLikelyOutcome } from "../utils";
import { TeamFlag } from "./TeamFlag";
import { MatchScoreComparison } from "./MatchScoreComparison";

interface MatchListRowProps {
  match: MatchPrediction;
  teams: Record<string, Team>;
  onOpen: (match: MatchPrediction, trigger: HTMLElement) => void;
  showScoreComparison?: boolean;
  hidePredictionDetails?: boolean;
}

export function MatchListRow({
  match,
  teams,
  onOpen,
  showScoreComparison = false,
  hidePredictionDetails = false,
}: MatchListRowProps) {
  const kickoff = formatKickoffParts(match.kickoffUtc);
  const probabilities = match.prediction.probabilities;
  const [predictedHomeScore, predictedAwayScore] = match.prediction.mostLikelyScore;

  // Check match status
  const ongoing = isMatchOngoing(match);
  const hasActualResult = !!match.actualResult;

  // Use actual result if available, otherwise show prediction
  const displayHomeScore = hasActualResult
    ? match.actualResult!.homeScore
    : predictedHomeScore;
  const displayAwayScore = hasActualResult
    ? match.actualResult!.awayScore
    : predictedAwayScore;

  function handleOpen(event: MouseEvent<HTMLButtonElement>) {
    onOpen(match, event.currentTarget);
  }

  return (
    <article className={`match-list-row ${showScoreComparison ? "match-list-row--comparison" : ""} ${hidePredictionDetails ? "match-list-row--without-prediction" : ""} ${ongoing ? "match-list-row--ongoing" : ""}`}>
      <div className="match-list-row__kickoff">
        <span className="eyebrow">Group {match.group}</span>
        <strong>{kickoff.date}</strong>
        <div className="match-list-row__time-wrapper">
          <span>{kickoff.time}</span>
          {ongoing && (
            <span className="match-list-row__live-badge">LIVE</span>
          )}
        </div>
      </div>
      <div className="match-list-row__fixture">
        <div className="match-list-row__team match-list-row__team--home">
          <TeamFlag team={teams[match.homeTeam]} compact />
        </div>
        {showScoreComparison ? (
          <MatchScoreComparison match={match} teams={teams} variant="list" showScorers />
        ) : (
          <strong
            className={`match-list-row__score ${hasActualResult ? "match-list-row__score--actual" : ""}`}
            aria-label={hasActualResult ? "Final score" : "Most likely score"}
          >
            {displayHomeScore}–{displayAwayScore}
          </strong>
        )}
        <div className="match-list-row__team match-list-row__team--away">
          <TeamFlag team={teams[match.awayTeam]} compact />
        </div>
      </div>
      {!showScoreComparison && hasActualResult && !ongoing && (
        <div className="match-list-row__result-badge">Final</div>
      )}
      {!hidePredictionDetails && (
        <div className="match-list-row__prediction">
          <span>{mostLikelyOutcome(probabilities, match.homeTeam, match.awayTeam)}</span>
          <div aria-label="Result probabilities">
            <span>H {formatPercent(probabilities.homeWin)}</span>
            <span>D {formatPercent(probabilities.draw)}</span>
            <span>A {formatPercent(probabilities.awayWin)}</span>
          </div>
        </div>
      )}
      <span className="match-list-row__venue">{match.venue}</span>
      <span className="match-list-row__actions">
        <button
          className="text-button match-list-row__action"
          type="button"
          onClick={handleOpen}
          aria-label={`Explore prediction for ${match.homeTeam} versus ${match.awayTeam}`}
        >
          Explore
        </button>
        <a
          className="text-link"
          href={match.sourceUrl}
          target="_blank"
          rel="noreferrer"
          aria-label={`View source data for ${match.homeTeam} versus ${match.awayTeam}`}
        >
          Source <span aria-hidden="true">↗</span>
        </a>
      </span>
    </article>
  );
}
