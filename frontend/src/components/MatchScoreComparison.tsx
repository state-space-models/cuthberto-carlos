import type { ActualGoal, MatchPrediction, Team } from "../types";
import { PlayerName } from "./PlayerName";

interface MatchScoreComparisonProps {
  match: MatchPrediction;
  variant: "card" | "list" | "drawer" | "group";
  teams?: Record<string, Team>;
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

export function MatchScoreComparison({
  match,
  variant,
  teams,
  showScorers = false,
}: MatchScoreComparisonProps) {
  const actualScore = match.actualResult
    ? `${match.actualResult.homeScore}–${match.actualResult.awayScore}`
    : "Currently Not Available";
  const [predictedHomeScore, predictedAwayScore] = match.prediction.mostLikelyScore;

  if (showScorers) {
    const homeTeam = teams?.[match.homeTeam];
    const awayTeam = teams?.[match.awayTeam];

    return (
      <div className={`score-comparison score-comparison--${variant} score-comparison--with-scorers`}>
        <div className="score-comparison__item score-comparison__prediction">
          <span>Predicted score</span>
          <strong>{predictedHomeScore}–{predictedAwayScore}</strong>
        </div>
        <div className="score-comparison__actual-row">
          <GoalList goals={match.actualResult?.homeGoals ?? []} team={homeTeam} />
          <div className="score-comparison__item score-comparison__actual">
            <span>Actual score</span>
            <strong className={!match.actualResult ? "score-comparison__unavailable" : undefined}>
              {actualScore}
            </strong>
          </div>
          <GoalList goals={match.actualResult?.awayGoals ?? []} team={awayTeam} />
        </div>
      </div>
    );
  }

  return (
    <div className={`score-comparison score-comparison--${variant}`}>
      <div className="score-comparison__item">
        <span>Actual score</span>
        <strong className={!match.actualResult ? "score-comparison__unavailable" : undefined}>
          {actualScore}
        </strong>
      </div>
      <div className="score-comparison__item">
        <span>Predicted score</span>
        <strong>{predictedHomeScore}–{predictedAwayScore}</strong>
      </div>
    </div>
  );
}
