import { useEffect } from "react";
import type { MatchPrediction, SkillEstimate, Team } from "../types";
import {
  aggregateScoreGrid,
  formatKickoff,
  formatPercent,
  isMatchOngoing,
} from "../utils";
import { TeamFlag } from "./TeamFlag";

interface MatchDetailDrawerProps {
  match: MatchPrediction | null;
  teams: Record<string, Team>;
  snapshotDate: string;
  modelName: string;
  onClose: () => void;
}

function SkillLine({ estimate }: { estimate: SkillEstimate }) {
  const lower = Math.max(0, Math.min(100, ((estimate.mean - estimate.sd + 1.5) / 3) * 100));
  const upper = Math.max(0, Math.min(100, ((estimate.mean + estimate.sd + 1.5) / 3) * 100));
  const point = Math.max(0, Math.min(100, ((estimate.mean + 1.5) / 3) * 100));
  return (
    <span className="skill-line" aria-hidden="true">
      <span className="skill-line__range" style={{ left: `${lower}%`, width: `${upper - lower}%` }} />
      <span className="skill-line__point" style={{ left: `${point}%` }} />
    </span>
  );
}

function ProbabilityRow({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="probability-row">
      <div className="probability-row__label">
        <span>{label}</span>
        <strong>{formatPercent(value, 1)}</strong>
      </div>
      <span className="probability-row__track">
        <span style={{ width: `${value * 100}%`, background: color }} />
      </span>
    </div>
  );
}

function ScoreHeatmap({ match }: { match: MatchPrediction }) {
  const grid = aggregateScoreGrid(match.prediction.scoreGrid);
  const maximum = Math.max(...grid.flat());
  const labels = ["0", "1", "2", "3", "4", "5+"];

  return (
    <div className="heatmap-wrap">
      <table className="heatmap">
        <caption>
          Score probability: columns are {match.homeTeam} goals, rows are {match.awayTeam} goals
        </caption>
        <thead>
          <tr>
            <th scope="col">Away ↓ / Home →</th>
            {labels.map((label) => (
              <th scope="col" key={label}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {labels.map((awayLabel, awayGoals) => (
            <tr key={awayLabel}>
              <th scope="row">{awayLabel}</th>
              {labels.map((homeLabel, homeGoals) => {
                const value = grid[homeGoals][awayGoals];
                const intensity = maximum === 0 ? 0 : value / maximum;
                return (
                  <td
                    key={homeLabel}
                    style={{
                      backgroundColor: `rgba(35, 117, 68, ${0.08 + intensity * 0.78})`,
                      color: intensity > 0.58 ? "white" : "#172019",
                    }}
                  >
                    {value < 0.005 ? "<1%" : formatPercent(value)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function MatchDetailDrawer({
  match,
  teams,
  snapshotDate,
  modelName,
  onClose,
}: MatchDetailDrawerProps) {
  useEffect(() => {
    if (!match) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handleKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeydown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeydown);
    };
  }, [match, onClose]);

  if (!match) return null;

  const home = teams[match.homeTeam];
  const away = teams[match.awayTeam];
  const probabilities = match.prediction.probabilities;
  const skills = match.prediction.skills;
  const homeColor = home.colors[0];
  const awayColor = away.colors[0] === homeColor ? away.colors[1] : away.colors[0];
  const ongoing = isMatchOngoing(match);
  const hasActualResult = !!match.actualResult;

  // Use actual result if available, otherwise show prediction
  const displayHomeScore = hasActualResult
    ? match.actualResult!.homeScore
    : match.prediction.mostLikelyScore[0];
  const displayAwayScore = hasActualResult
    ? match.actualResult!.awayScore
    : match.prediction.mostLikelyScore[1];

  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className={`prediction-drawer ${ongoing ? "prediction-drawer--ongoing" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="prediction-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <button className="drawer-close" type="button" onClick={onClose} autoFocus>
          <span aria-hidden="true">×</span>
          <span className="sr-only">Close prediction details</span>
        </button>

        <header className="drawer-header">
          <span className="eyebrow">
            Group {match.group} · {formatKickoff(match.kickoffUtc)}
            {ongoing && <span className="drawer-live-badge">LIVE</span>}
          </span>
          <h2 id="prediction-title">{match.homeTeam} vs {match.awayTeam}</h2>
          <p>{match.venue}</p>
          <div className="drawer-matchup">
            <TeamFlag team={home} />
            <span className="drawer-score">
              <small>{ongoing ? "Current score" : hasActualResult ? "Final score" : "Most likely"}</small>
              {displayHomeScore}–{displayAwayScore}
              {!ongoing && !hasActualResult && <em>{formatPercent(match.prediction.mostLikelyScoreProbability, 1)}</em>}
            </span>
            <TeamFlag team={away} />
          </div>
        </header>

        <div className="drawer-grid">
          <section className="drawer-panel">
            <h3>Result probabilities</h3>
            <ProbabilityRow label={`${match.homeTeam} win`} value={probabilities.homeWin} color={homeColor} />
            <ProbabilityRow label="Draw" value={probabilities.draw} color="#777b76" />
            <ProbabilityRow label={`${match.awayTeam} win`} value={probabilities.awayWin} color={awayColor} />
          </section>

          <section className="drawer-panel drawer-panel--skills">
            <h3>Team strength estimates</h3>
            <div className="skill-table">
              {([
                [match.homeTeam, "Attack", skills.home.attack],
                [match.homeTeam, "Defence", skills.home.defence],
                [match.awayTeam, "Attack", skills.away.attack],
                [match.awayTeam, "Defence", skills.away.defence],
              ] as const).map(([team, metric, estimate]) => (
                <div className="skill-table__row" key={`${team}-${metric}`}>
                  <span>{team} {metric}</span>
                  <SkillLine estimate={estimate} />
                  <strong>{estimate.mean.toFixed(2)} ± {estimate.sd.toFixed(2)}</strong>
                </div>
              ))}
            </div>
            <p className="panel-note">Higher attack and defence values indicate stronger model estimates.</p>
          </section>

          <section className="drawer-panel drawer-panel--wide">
            <h3>Scoreline distribution</h3>
            <p className="panel-note">
              Expected goals: {match.homeTeam} {match.prediction.expectedGoals.home.toFixed(2)}, {match.awayTeam} {match.prediction.expectedGoals.away.toFixed(2)}.
            </p>
            <ScoreHeatmap match={match} />
          </section>
        </div>

        <footer className="drawer-footer">
          <a href={match.sourceUrl} target="_blank" rel="noreferrer">
            Snapshot {snapshotDate} source data <span aria-hidden="true">↗</span>
          </a>
          <span>{modelName}</span>
        </footer>
      </section>
    </div>
  );
}
