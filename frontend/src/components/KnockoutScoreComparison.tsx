import type { KnockoutMatch, Team, ActualGoal } from "../types";
import { PlayerName } from "./PlayerName";

interface KnockoutScoreComparisonProps {
  match: KnockoutMatch;
  teams: Record<string, Team>;
  variant?: "card" | "list" | "drawer" | "group";
  showScorers?: boolean;
}

function GoalLine({ goal, team }: { goal: ActualGoal; team?: Team }) {
  const player = team?.players.find((candidate) => candidate.name === goal.name);

  return (
    <li aria-label={`${goal.name}, ${goal.minute} minute${goal.penalty ? ", penalty" : ""}`}>
      <span aria-hidden="true">⚽</span>
      <PlayerName name={goal.name} player={player} />
      <span>{goal.minute}'{goal.penalty ? " (pen.)" : ""}</span>
    </li>
  );
}

function GoalList({ goals, team }: { goals: ActualGoal[]; team?: Team }) {
  if (!goals.length) return <span className="score-comparison__goals-placeholder" aria-hidden="true" />;

  return (
    <ul className="score-comparison__goals" aria-label={`${team?.name ?? "Team"} goal scorers`}>
      {goals.map((goal, index) => (
        <GoalLine
          goal={goal}
          team={team}
          key={`${goal.name}-${goal.minute}-${index}`}
        />
      ))}
    </ul>
  );
}

export function KnockoutScoreComparison({
  match,
  teams,
  variant = "card",
  showScorers = false,
}: KnockoutScoreComparisonProps) {
  const homeTeamName = match.team1 ?? match.team1Slot;
  const awayTeamName = match.team2 ?? match.team2Slot;
  const homeTeam = teams?.[homeTeamName];
  const awayTeam = teams?.[awayTeamName];

  const predictedScore = match.prediction?.mostLikelyScore;
  const predictedHomeScore = predictedScore?.[0] ?? 0;
  const predictedAwayScore = predictedScore?.[1] ?? 0;

  // Get actual score - use penalties if available, then extra time, then full time
  const actualFullTime = match.score?.fullTime;
  const actualExtraTime = match.score?.extraTime;
  const actualPenalties = match.score?.penalties;
  
  // For display: if penalties exist, show the final score after penalties
  // If extra time exists (but no penalties), show extra time score
  // Otherwise show full time score
  const displayScore = actualPenalties ?? actualExtraTime ?? actualFullTime;
  const actualHomeScore = displayScore?.[0];
  const actualAwayScore = displayScore?.[1];
  
  const hasActualScore = actualFullTime !== undefined;
  const hasPenalties = actualPenalties !== undefined;
  const hasExtraTime = actualExtraTime !== undefined;

  // Format actual score display with penalty/extra time indicators
  let actualScoreDisplay: string;
  if (!hasActualScore) {
    actualScoreDisplay = "Currently Not Available";
  } else if (hasPenalties) {
    actualScoreDisplay = `${actualHomeScore}–${actualAwayScore}`;
  } else if (hasExtraTime) {
    actualScoreDisplay = `${actualHomeScore}–${actualAwayScore}`;
  } else {
    actualScoreDisplay = `${actualHomeScore}–${actualAwayScore}`;
  }

  if (showScorers) {
    const homeGoals = match.score?.homeGoals ?? [];
    const awayGoals = match.score?.awayGoals ?? [];

    return (
      <div className={`score-comparison score-comparison--${variant} score-comparison--with-scorers`}>
        <div className="score-comparison__item score-comparison__prediction">
          <span>Predicted score</span>
          <strong>{predictedHomeScore}–{predictedAwayScore}</strong>
        </div>
        <div className="score-comparison__actual-row">
          <GoalList goals={homeGoals} team={homeTeam} />
          <div className="score-comparison__item score-comparison__actual">
            <span>Actual score</span>
            <strong className={!hasActualScore ? "score-comparison__unavailable" : undefined}>
              {actualScoreDisplay}
              {hasPenalties && (
                <small className="score-comparison__penalty-score">
                  ({actualPenalties[0]}–{actualPenalties[1]} pens)
                </small>
              )}
              {hasExtraTime && !hasPenalties && (
                <small className="score-comparison__aet-indicator">AET</small>
              )}
            </strong>
          </div>
          <GoalList goals={awayGoals} team={awayTeam} />
        </div>
      </div>
    );
  }

  return (
    <div className={`score-comparison score-comparison--${variant}`}>
      <div className="score-comparison__item">
        <span>Actual score</span>
        <strong className={!hasActualScore ? "score-comparison__unavailable" : undefined}>
          {actualScoreDisplay}
          {hasPenalties && (
            <small className="score-comparison__penalty-score">
              ({actualPenalties[0]}–{actualPenalties[1]} pens)
            </small>
          )}
          {hasExtraTime && !hasPenalties && (
            <small className="score-comparison__aet-indicator">AET</small>
          )}
        </strong>
      </div>
      <div className="score-comparison__item">
        <span>Predicted score</span>
        <strong>{predictedHomeScore}–{predictedAwayScore}</strong>
      </div>
    </div>
  );
}
