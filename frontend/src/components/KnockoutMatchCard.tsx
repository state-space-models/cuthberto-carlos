import type { MouseEvent } from "react";
import type { KnockoutMatch, Team } from "../types";
import {
  formatKickoffParts,
  formatPercent,
  isMatchOngoing,
  mostLikelyOutcome,
  ROUND_LABELS,
} from "../utils";
import { TeamFlag } from "./TeamFlag";
import { KnockoutScoreComparison } from "./KnockoutScoreComparison";
import { PolymarketCardComparison } from "./PolymarketComparison";

interface KnockoutMatchCardProps {
  match: KnockoutMatch;
  teams: Record<string, Team>;
  onOpen: (match: KnockoutMatch, trigger: HTMLElement) => void;
  showScoreComparison?: boolean;
  hidePredictionDetails?: boolean;
  showPolymarket?: boolean;
}

export function KnockoutMatchCard({
  match,
  teams,
  onOpen,
  showScoreComparison = false,
  hidePredictionDetails = false,
  showPolymarket = false,
}: KnockoutMatchCardProps) {
  const kickoff = formatKickoffParts(match.kickoffUtc);
  const probabilities = match.prediction?.probabilities;
  const predictedScore = match.prediction?.mostLikelyScore;

  // Check match status
  const ongoing = isMatchOngoing(match);
  const hasScore = !!match.score;

  // Get actual final score (penalties > extra time > full time) for simple display
  const finalScore = match.score?.penalties ?? match.score?.extraTime ?? match.score?.fullTime;
  const displayHomeScore = finalScore ? finalScore[0] : predictedScore?.[0] ?? 0;
  const displayAwayScore = finalScore ? finalScore[1] : predictedScore?.[1] ?? 0;

  // Get team names (resolved or slot)
  const homeTeam = match.team1 ?? match.team1Slot;
  const awayTeam = match.team2 ?? match.team2Slot;

  function handleOpen(event: MouseEvent<HTMLElement>) {
    onOpen(match, event.currentTarget);
  }

  return (
    <article 
      className={`match-card ${ongoing ? "match-card--ongoing" : ""}`}
      onClick={handleOpen}
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleOpen(e as unknown as MouseEvent<HTMLElement>); } }}
      style={{ cursor: 'pointer' }}
    >
      <div className="match-card__meta">
        <span className="eyebrow">{ROUND_LABELS[match.round]}</span>
        <span>{kickoff.date}</span>
        <div className="match-card__time-wrapper">
          <strong>{kickoff.time}</strong>
          {ongoing && (
            <span className="match-card__live-badge">LIVE</span>
          )}
        </div>
      </div>
      <div className="match-card__teams">
        <div className="match-card__team match-card__team--first">
          <TeamFlag team={teams[homeTeam]} />
        </div>
        {showScoreComparison ? (
          <KnockoutScoreComparison match={match} teams={teams} variant="card" showScorers />
        ) : (
          <div className="match-card__score-block">
            {!hidePredictionDetails && probabilities && (
              <span className="match-card__outcome">
                {mostLikelyOutcome(probabilities, homeTeam, awayTeam)}
              </span>
            )}
            <span
              className={`match-card__score ${hasScore ? "match-card__score--actual" : ""}`}
              aria-label={hasScore ? "Final score" : "Most likely score"}
            >
              {displayHomeScore}–{displayAwayScore}
            </span>
          </div>
        )}
        <div className="match-card__team match-card__team--second">
          <TeamFlag team={teams[awayTeam]} />
        </div>
      </div>
      {!showScoreComparison && hasScore && !ongoing && (
        <div className="match-card__result-badge">Final</div>
      )}
      <p className="match-card__venue">{match.venue}</p>
      {!hidePredictionDetails && showPolymarket && match.polymarket ? (
        <PolymarketCardComparison match={match as unknown as import("../types").MatchPrediction} />
      ) : !hidePredictionDetails && probabilities ? (
        <>
          <div className="probability-strip" aria-label="Result probabilities">
            <span
              className="probability-strip__first-team"
              style={{ width: `${probabilities.homeWin * 100}%` }}
              title={`${homeTeam} ${formatPercent(probabilities.homeWin, 1)}`}
            />
            <span
              className="probability-strip__draw"
              style={{ width: `${probabilities.draw * 100}%` }}
              title={`Draw ${formatPercent(probabilities.draw, 1)}`}
            />
            <span
              className="probability-strip__second-team"
              style={{ width: `${probabilities.awayWin * 100}%` }}
              title={`${awayTeam} ${formatPercent(probabilities.awayWin, 1)}`}
            />
          </div>
          <div className="match-card__probabilities">
            <span>{homeTeam} {formatPercent(probabilities.homeWin)}</span>
            <span>Draw {formatPercent(probabilities.draw)}</span>
            <span>{awayTeam} {formatPercent(probabilities.awayWin)}</span>
          </div>
        </>
      ) : null}
    </article>
  );
}
