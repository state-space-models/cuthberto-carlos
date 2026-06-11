import type { MatchPrediction } from "../types";

interface MatchScoreComparisonProps {
  match: MatchPrediction;
  variant: "card" | "list" | "drawer";
}

export function MatchScoreComparison({ match, variant }: MatchScoreComparisonProps) {
  const actualScore = match.actualResult
    ? `${match.actualResult.homeScore}–${match.actualResult.awayScore}`
    : "Currently Not Available";
  const [predictedHomeScore, predictedAwayScore] = match.prediction.mostLikelyScore;

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
