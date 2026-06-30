import type { MouseEvent } from "react";
import type { KnockoutMatch, Team } from "../types";
import { formatKickoffParts, formatPercent, getKnockoutWinner, isMatchOngoing, mostLikelyOutcome, ROUND_LABELS } from "../utils";
import { TeamFlag } from "./TeamFlag";
import { KnockoutScoreComparison } from "./KnockoutScoreComparison";
import { PolymarketCompact } from "./PolymarketComparison";

interface KnockoutMatchListRowProps {
  match: KnockoutMatch;
  teams: Record<string, Team>;
  onOpen: (match: KnockoutMatch, trigger: HTMLElement) => void;
  showScoreComparison?: boolean;
  hidePredictionDetails?: boolean;
  showPolymarket?: boolean;
}

export function KnockoutMatchListRow({
  match,
  teams,
  onOpen,
  showScoreComparison = false,
  hidePredictionDetails = false,
  showPolymarket = false,
}: KnockoutMatchListRowProps) {
  const kickoff = formatKickoffParts(match.kickoffUtc);
  const probabilities = match.prediction?.probabilities;
  const predictedScore = match.prediction?.mostLikelyScore;

  // Check match status
  const ongoing = isMatchOngoing(match);
  const hasScore = !!match.score;

  // Get actual final score (for display in simple mode)
  const finalScore = match.score?.penalties ?? match.score?.extraTime ?? match.score?.fullTime;
  const displayHomeScore = finalScore ? finalScore[0] : predictedScore?.[0] ?? 0;
  const displayAwayScore = finalScore ? finalScore[1] : predictedScore?.[1] ?? 0;

  // Get team names
  const homeTeam = match.team1 ?? match.team1Slot;
  const awayTeam = match.team2 ?? match.team2Slot;

  // Determine winner for display
  const winner = hasScore ? getKnockoutWinner(match) : null;

  function handleOpen(event: MouseEvent<HTMLButtonElement>) {
    onOpen(match, event.currentTarget);
  }

  return (
    <article 
      className={`match-list-row ${showScoreComparison ? "match-list-row--comparison" : ""} ${hidePredictionDetails ? "match-list-row--without-prediction" : ""} ${ongoing ? "match-list-row--ongoing" : ""}`}
      onClick={handleOpen}
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleOpen(e as unknown as MouseEvent<HTMLButtonElement>); } }}
      style={{ cursor: 'pointer' }}
    >
      <div className="match-list-row__kickoff">
        <span className="eyebrow">{ROUND_LABELS[match.round]}</span>
        <strong>{kickoff.date}</strong>
        <div className="match-list-row__time-wrapper">
          <span>{kickoff.time}</span>
          {ongoing && (
            <span className="match-list-row__live-badge">LIVE</span>
          )}
        </div>
      </div>
      <div className="match-list-row__fixture">
        <div className="match-list-row__team match-list-row__team--first">
          <TeamFlag team={teams[homeTeam]} compact />
        </div>
        {showScoreComparison ? (
          <KnockoutScoreComparison match={match} teams={teams} variant="list" showScorers />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.2rem' }}>
            <strong
              className={`match-list-row__score ${hasScore ? "match-list-row__score--actual" : ""}`}
              aria-label={hasScore ? "Final score" : "Most likely score"}
            >
              {displayHomeScore}–{displayAwayScore}
            </strong>
            {winner && (
              <span style={{ fontSize: '0.7rem', color: 'var(--muted)', fontWeight: 600 }}>
                {winner} wins
              </span>
            )}
          </div>
        )}
        <div className="match-list-row__team match-list-row__team--second">
          <TeamFlag team={teams[awayTeam]} compact />
        </div>
      </div>
      {!showScoreComparison && hasScore && !ongoing && (
        <div className="match-list-row__result-badge">Final</div>
      )}
      {!hidePredictionDetails && probabilities && (
        <div className="match-list-row__prediction">
          <span>{mostLikelyOutcome(probabilities, homeTeam, awayTeam)}</span>
          <div aria-label="Result probabilities">
            <span>{homeTeam} {formatPercent(probabilities.homeWin)}</span>
            <span>Draw {formatPercent(probabilities.draw)}</span>
            <span>{awayTeam} {formatPercent(probabilities.awayWin)}</span>
          </div>
          {showPolymarket && <PolymarketCompact match={match as unknown as import("../types").MatchPrediction} />}
        </div>
      )}
      <span className="match-list-row__venue">{match.venue}</span>
    </article>
  );
}
