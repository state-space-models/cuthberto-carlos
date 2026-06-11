import type { MouseEvent } from "react";
import type { MatchPrediction, Team } from "../types";
import {
  formatKickoffParts,
  formatPercent,
  isMatchOngoing,
  mostLikelyOutcome,
} from "../utils";
import { TeamFlag } from "./TeamFlag";

interface MatchCardProps {
  match: MatchPrediction;
  teams: Record<string, Team>;
  onOpen: (match: MatchPrediction, trigger: HTMLElement) => void;
}

export function MatchCard({ match, teams, onOpen }: MatchCardProps) {
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
    <article className={`match-card ${ongoing ? "match-card--ongoing" : ""}`}>
      <div className="match-card__meta">
        <span className="eyebrow">Group {match.group}</span>
        <span>{kickoff.date}</span>
        <div className="match-card__time-wrapper">
          <strong>{kickoff.time}</strong>
          {ongoing && (
            <span className="match-card__live-badge">LIVE</span>
          )}
        </div>
      </div>
      <div className="match-card__teams">
        <div className="match-card__team match-card__team--home">
          <TeamFlag team={teams[match.homeTeam]} />
        </div>
        <span
          className={`match-card__score ${hasActualResult ? "match-card__score--actual" : ""}`}
          aria-label={hasActualResult ? "Final score" : "Most likely score"}
        >
          {displayHomeScore}–{displayAwayScore}
        </span>
        <div className="match-card__team match-card__team--away">
          <TeamFlag team={teams[match.awayTeam]} />
        </div>
      </div>
      {hasActualResult && !ongoing && (
        <div className="match-card__result-badge">Final</div>
      )}
      <p className="match-card__venue">{match.venue}</p>
      <div className="probability-strip" aria-label="Result probabilities">
        <span
          className="probability-strip__home"
          style={{ width: `${probabilities.homeWin * 100}%` }}
          title={`${match.homeTeam} ${formatPercent(probabilities.homeWin, 1)}`}
        />
        <span
          className="probability-strip__draw"
          style={{ width: `${probabilities.draw * 100}%` }}
          title={`Draw ${formatPercent(probabilities.draw, 1)}`}
        />
        <span
          className="probability-strip__away"
          style={{ width: `${probabilities.awayWin * 100}%` }}
          title={`${match.awayTeam} ${formatPercent(probabilities.awayWin, 1)}`}
        />
      </div>
      <div className="match-card__probabilities" aria-hidden="true">
        <span>H {formatPercent(probabilities.homeWin)}</span>
        <span>D {formatPercent(probabilities.draw)}</span>
        <span>A {formatPercent(probabilities.awayWin)}</span>
      </div>
      <div className="match-card__footer">
        <span>{mostLikelyOutcome(probabilities, match.homeTeam, match.awayTeam)}</span>
        <span className="match-card__actions">
          <button className="text-button" type="button" onClick={handleOpen}>
            Explore prediction
          </button>
          <a className="text-link" href={match.sourceUrl} target="_blank" rel="noreferrer">
            Source <span aria-hidden="true">↗</span>
          </a>
        </span>
      </div>
    </article>
  );
}
