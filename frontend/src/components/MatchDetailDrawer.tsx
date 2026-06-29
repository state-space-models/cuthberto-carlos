import { useEffect, useState } from "react";
import type {
  ActualGoal,
  KnockoutMatch,
  MatchPrediction,
  PredictionDetails,
  PredictionHistoryEntry,
  Team,
} from "../types";
import {
  aggregateScoreGrid,
  formatKickoff,
  formatPercent,
  isMatchCompleted,
  isMatchOngoing,
} from "../utils";
import { MatchScoreComparison } from "./MatchScoreComparison";
import { TeamFlag } from "./TeamFlag";
import { PolymarketDetail } from "./PolymarketComparison";

interface DrawerMatch {
  id: string;
  homeTeam: string;
  awayTeam: string;
  kickoffUtc: string;
  venue: string;
  eyebrow: string;
  prediction: PredictionDetails;
  predictionDate: string;
  sourceUrl: string;
  predictionHistory: PredictionHistoryEntry[];
  polymarket?: MatchPrediction["polymarket"];
  actualResult?: MatchPrediction["actualResult"];
  knockoutScore?: KnockoutMatch["score"];
}

interface MatchDetailDrawerProps {
  match: MatchPrediction | KnockoutMatch | null;
  matches: MatchPrediction[];
  teams: Record<string, Team>;
  modelName: string;
  onClose: () => void;
}

function normalizeMatch(match: MatchPrediction | KnockoutMatch): DrawerMatch {
  if ("homeTeam" in match) {
    return {
      id: match.id,
      homeTeam: match.homeTeam,
      awayTeam: match.awayTeam,
      kickoffUtc: match.kickoffUtc,
      venue: match.venue,
      eyebrow: `Group ${match.group}`,
      prediction: match.prediction,
      predictionDate: match.predictionDate,
      sourceUrl: match.sourceUrl,
      predictionHistory: match.predictionHistory,
      polymarket: match.polymarket,
      actualResult: match.actualResult,
    };
  }
  return {
    id: match.id,
    homeTeam: match.team1 ?? match.team1Slot,
    awayTeam: match.team2 ?? match.team2Slot,
    kickoffUtc: match.kickoffUtc,
    venue: match.venue,
    eyebrow: match.round,
    prediction: match.prediction!,
    predictionDate: match.predictionDate!,
    sourceUrl: match.sourceUrl!,
    predictionHistory: match.predictionHistory ?? [],
    polymarket: match.polymarket,
    knockoutScore: match.score,
  };
}

type PredictionVersion = PredictionHistoryEntry;

const FIRST_TEAM_COLOR = "#15803d";
const SECOND_TEAM_COLOR = "#dc2626";
const DRAW_COLOR = "#777b76";

function formatMatchDate(date: string): string {
  return new Intl.DateTimeFormat(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${date}T12:00:00Z`));
}

interface KnockoutScoreComparisonProps {
  match: DrawerMatch;
  prediction: PredictionDetails;
  variant: "drawer" | "card" | "list";
}

function GoalList({ goals }: { goals: ActualGoal[] }) {
  if (!goals.length) return <span className="score-comparison__goals-placeholder" aria-hidden="true" />;

  return (
    <ul className="score-comparison__goals" aria-label="Goal scorers">
      {goals.map((goal, index) => (
        <li key={`${goal.name}-${goal.minute}-${index}`}>
          <span className="player-name">{goal.name}</span>
          <span>{goal.minute}'</span>
        </li>
      ))}
    </ul>
  );
}

function KnockoutScoreComparison({
  match,
  prediction,
  variant,
}: KnockoutScoreComparisonProps) {
  const [predHome, predAway] = prediction.mostLikelyScore;
  const actualScore = match.knockoutScore;

  if (!actualScore) return null;

  const actualHome = actualScore.extraTime?.[0] ?? actualScore.fullTime[0];
  const actualAway = actualScore.extraTime?.[1] ?? actualScore.fullTime[1];
  const hasPenalties = !!actualScore.penalties;
  const hasExtraTime = !!actualScore.extraTime && !hasPenalties;

  const homeGoals = actualScore.homeGoals ?? [];
  const awayGoals = actualScore.awayGoals ?? [];
  const hasScorers = homeGoals.length > 0 || awayGoals.length > 0;

  return (
    <div className={`score-comparison score-comparison--${variant} ${hasScorers ? 'score-comparison--with-scorers' : ''}`}>
      <div className="score-comparison__item score-comparison__prediction">
        <span>Predicted score</span>
        <strong>{predHome}–{predAway}</strong>
      </div>
      <div className="score-comparison__actual-row">
        <GoalList goals={homeGoals} />
        <div className="score-comparison__item score-comparison__actual">
          <span>Actual score</span>
          <strong>
            {actualHome}–{actualAway}
            {hasPenalties && <small> pens {actualScore.penalties![0]}–{actualScore.penalties![1]}</small>}
            {hasExtraTime && <small> AET</small>}
          </strong>
        </div>
        <GoalList goals={awayGoals} />
      </div>
    </div>
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

function ScoreHeatmap({
  prediction,
  homeTeam,
  awayTeam,
}: {
  prediction: PredictionDetails;
  homeTeam: string;
  awayTeam: string;
}) {
  const grid = aggregateScoreGrid(prediction.scoreGrid);
  const maximum = Math.max(...grid.flat());
  const labels = ["0", "1", "2", "3", "4", "5+"];

  return (
    <div className="heatmap-wrap">
      <table className="heatmap">
        <caption>
          Score probability: columns are {homeTeam} goals, rows are {awayTeam} goals
        </caption>
        <thead>
          <tr>
            <th scope="col">{awayTeam} ↓ / {homeTeam} →</th>
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

function StrengthHistory({
  match,
  versions,
  selectedDate,
}: {
  match: DrawerMatch;
  versions: PredictionVersion[];
  selectedDate: string;
}) {
  const width = 560;
  const height = 220;
  const horizontalPadding = 24;
  const verticalPadding = 20;
  const x = (index: number) =>
    versions.length === 1
      ? width / 2
      : horizontalPadding +
        (index * (width - horizontalPadding * 2)) / (versions.length - 1);
  const y = (value: number) =>
    height - verticalPadding - ((Math.max(-1.5, Math.min(1.5, value)) + 1.5) / 3) *
      (height - verticalPadding * 2);
  const selected = versions.find((version) => version.predictionDate === selectedDate)!;
  const series = [
    {
      key: "home-attack",
      label: `${match.homeTeam} Attack`,
      team: match.homeTeam,
      metric: "Attack",
      color: "first",
      getEstimate: (prediction: PredictionDetails) => prediction.skills.home.attack,
    },
    {
      key: "home-defence",
      label: `${match.homeTeam} Defence`,
      team: match.homeTeam,
      metric: "Defence",
      color: "first",
      getEstimate: (prediction: PredictionDetails) => prediction.skills.home.defence,
    },
    {
      key: "away-attack",
      label: `${match.awayTeam} Attack`,
      team: match.awayTeam,
      metric: "Attack",
      color: "second",
      getEstimate: (prediction: PredictionDetails) => prediction.skills.away.attack,
    },
    {
      key: "away-defence",
      label: `${match.awayTeam} Defence`,
      team: match.awayTeam,
      metric: "Defence",
      color: "second",
      getEstimate: (prediction: PredictionDetails) => prediction.skills.away.defence,
    },
  ] as const;

  return (
    <div className="strength-history">
      <div className="strength-history__legend" aria-label="Team strength chart legend">
        <span className="strength-history__team strength-history__team--first">{match.homeTeam}</span>
        <span className="strength-history__team strength-history__team--second">{match.awayTeam}</span>
        <span className="strength-history__metric"><i />Attack</span>
        <span className="strength-history__metric strength-history__metric--defence"><i />Defence</span>
      </div>
      <svg
        className="strength-history__chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Team strength history for ${match.homeTeam} and ${match.awayTeam}`}
      >
        {[-1, 0, 1].map((tick) => (
          <g key={tick}>
            <line className="strength-history__grid" x1="0" y1={y(tick)} x2={width} y2={y(tick)} />
            <text className="strength-history__axis-label" x="2" y={y(tick) - 4}>{tick.toFixed(0)}</text>
          </g>
        ))}
        {series.map((item) => {
          const points = versions
            .map((version, index) => `${x(index)},${y(item.getEstimate(version.prediction).mean)}`)
            .join(" ");
          return (
            <g
              className={`strength-series strength-series--${item.color} ${item.metric === "Defence" ? "strength-series--defence" : ""}`}
              key={item.key}
            >
              {versions.length > 1 && <polyline className="strength-series__line" points={points} />}
              {versions.map((version, index) => {
                const estimate = item.getEstimate(version.prediction);
                const selectedPoint = version.predictionDate === selectedDate;
                return (
                  <g key={version.predictionDate}>
                    <line
                      className="strength-series__uncertainty"
                      x1={x(index)}
                      y1={y(estimate.mean - estimate.sd)}
                      x2={x(index)}
                      y2={y(estimate.mean + estimate.sd)}
                    />
                    <circle
                      className={selectedPoint ? "strength-series__point strength-series__point--selected" : "strength-series__point"}
                      cx={x(index)}
                      cy={y(estimate.mean)}
                      r={selectedPoint ? 5 : 3.5}
                    />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      <div className="strength-history__dates" aria-label="Prediction dates">
        {versions.map((version) => (
          <span className={version.predictionDate === selectedDate ? "strength-history__date--selected" : ""} key={version.predictionDate}>
            {version.predictionDate}
          </span>
        ))}
      </div>
      <div className="strength-history__values">
        {series.map((item) => {
          const estimate = item.getEstimate(selected.prediction);
          return (
            <span className={`strength-history__value strength-history__value--${item.color}`} key={item.key}>
              <strong>{item.team}</strong> {item.metric} {estimate.mean.toFixed(2)} ± {estimate.sd.toFixed(2)}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function RecentTeamMatches({
  match,
  matches,
  predictionDate,
  teams,
}: {
  match: DrawerMatch;
  matches: MatchPrediction[];
  predictionDate: string;
  teams: Record<string, Team>;
}) {
  const teamNames = new Set([match.homeTeam, match.awayTeam]);
  const recentMatches = matches
    .filter(
      (candidate) =>
        candidate.id !== match.id &&
        candidate.actualResult &&
        candidate.date < predictionDate &&
        (teamNames.has(candidate.homeTeam) || teamNames.has(candidate.awayTeam)),
    )
    .sort((left, right) => right.kickoffUtc.localeCompare(left.kickoffUtc));

  return (
    <section className="drawer-panel drawer-panel--wide recent-matches">
      <h3>Recent matches</h3>
      <p className="panel-note">
        Completed matches involving {match.homeTeam} or {match.awayTeam} before {predictionDate}.
      </p>
      {recentMatches.length > 0 ? (
        <ul
          className="recent-matches__list"
          aria-label={`Matches involving ${match.homeTeam} or ${match.awayTeam} before ${predictionDate}`}
        >
          {recentMatches.map((recentMatch) => {
            const result = recentMatch.actualResult!;
            return (
              <li className="recent-matches__item" key={recentMatch.id}>
                <time dateTime={recentMatch.date}>{formatMatchDate(recentMatch.date)}</time>
                <span className="recent-matches__fixture">
                  <TeamFlag team={teams[recentMatch.homeTeam]} compact />
                  <strong aria-label="Final score">
                    {result.homeScore}–{result.awayScore}
                  </strong>
                  <TeamFlag team={teams[recentMatch.awayTeam]} compact />
                </span>
                <span className="recent-matches__venue">{recentMatch.venue}</span>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="recent-matches__empty">
          No completed matches involving either team before this prediction.
        </p>
      )}
    </section>
  );
}

export function MatchDetailDrawer({
  match,
  matches,
  teams,
  modelName,
  onClose,
}: MatchDetailDrawerProps) {
  const [selectedPredictionDate, setSelectedPredictionDate] = useState<string | null>(null);

  useEffect(() => {
    setSelectedPredictionDate(match?.predictionDate ?? null);
  }, [match]);

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

  const normalized = normalizeMatch(match);
  const home = teams[normalized.homeTeam];
  const away = teams[normalized.awayTeam];
  const versions: PredictionVersion[] = [
    ...normalized.predictionHistory,
    {
      predictionDate: normalized.predictionDate,
      sourceUrl: normalized.sourceUrl,
      prediction: normalized.prediction,
    },
  ].sort((left, right) => left.predictionDate.localeCompare(right.predictionDate));
  const selectedVersion =
    versions.find((version) => version.predictionDate === selectedPredictionDate) ??
    versions[versions.length - 1];
  const latestPredictionDate = versions[versions.length - 1].predictionDate;
  const selectedPrediction = selectedVersion.prediction;
  const probabilities = selectedPrediction.probabilities;
  const ongoing = isMatchOngoing(normalized);
  const completed = isMatchCompleted(normalized);
  const hasActualResult = !!normalized.actualResult;
  const hasKnockoutScore = !!normalized.knockoutScore;
  const isKnockout = hasKnockoutScore;

  // Use actual result if available, otherwise show prediction
  const displayHomeScore = hasActualResult
    ? normalized.actualResult!.homeScore
    : hasKnockoutScore
      ? (normalized.knockoutScore!.extraTime ?? normalized.knockoutScore!.fullTime)[0]
      : selectedPrediction.mostLikelyScore[0];
  const displayAwayScore = hasActualResult
    ? normalized.actualResult!.awayScore
    : hasKnockoutScore
      ? (normalized.knockoutScore!.extraTime ?? normalized.knockoutScore!.fullTime)[1]
      : selectedPrediction.mostLikelyScore[1];

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
            {normalized.eyebrow} · {formatKickoff(normalized.kickoffUtc)}
            {ongoing && <span className="drawer-live-badge">LIVE</span>}
          </span>
          <h2 id="prediction-title">{normalized.homeTeam} vs {normalized.awayTeam}</h2>
          <p>{normalized.venue}</p>
          <div className="drawer-matchup">
            <TeamFlag team={home} />
            {(completed || hasActualResult) && !isKnockout ? (
              <MatchScoreComparison match={match as MatchPrediction} variant="drawer" />
            ) : (completed || hasKnockoutScore) && isKnockout ? (
              <KnockoutScoreComparison
                match={normalized}
                prediction={selectedPrediction}
                variant="drawer"
              />
            ) : (
              <span className="drawer-score">
                <small>{ongoing ? "Current score" : hasActualResult || hasKnockoutScore ? "Final score" : "Most likely"}</small>
                {displayHomeScore}–{displayAwayScore}
                {!ongoing && !hasActualResult && !hasKnockoutScore && <em>{formatPercent(selectedPrediction.mostLikelyScoreProbability, 1)}</em>}
                {hasKnockoutScore && normalized.knockoutScore?.penalties && (
                  <em>pens {normalized.knockoutScore.penalties[0]}–{normalized.knockoutScore.penalties[1]}</em>
                )}
                {hasKnockoutScore && !normalized.knockoutScore?.penalties && normalized.knockoutScore?.extraTime && (
                  <em>AET</em>
                )}
              </span>
            )}
            <TeamFlag team={away} />
          </div>
        </header>

        {versions.length > 1 && (
          <div className="prediction-date-selector" role="group" aria-label="Prediction date">
            <label htmlFor="prediction-date-select">View prediction</label>
            <span className="prediction-date-selector__control">
              <select
                id="prediction-date-select"
                value={selectedVersion.predictionDate}
                onChange={(event) => setSelectedPredictionDate(event.target.value)}
              >
                {versions.slice().reverse().map((version) => (
                  <option key={version.predictionDate} value={version.predictionDate}>
                    {version.predictionDate}{version.predictionDate === latestPredictionDate ? " (latest)" : ""}
                  </option>
                ))}
              </select>
            </span>
          </div>
        )}

        <div className="drawer-grid">
          <section className="drawer-panel">
            <h3>Result probabilities</h3>
            <ProbabilityRow label={`${normalized.homeTeam} win`} value={probabilities.homeWin} color={FIRST_TEAM_COLOR} />
            <ProbabilityRow label="Draw" value={probabilities.draw} color={DRAW_COLOR} />
            <ProbabilityRow label={`${normalized.awayTeam} win`} value={probabilities.awayWin} color={SECOND_TEAM_COLOR} />
          </section>

          <PolymarketDetail match={normalized} />

          <section className="drawer-panel drawer-panel--skills">
            <h3>Team strength estimates</h3>
            <StrengthHistory
              match={normalized}
              versions={versions}
              selectedDate={selectedVersion.predictionDate}
            />
            <p className="panel-note">Higher attack and defence values indicate stronger model estimates.</p>
          </section>

          <section className="drawer-panel drawer-panel--wide">
            <h3>Scoreline distribution</h3>
            <p className="panel-note">
              Expected goals: {normalized.homeTeam} {selectedPrediction.expectedGoals.home.toFixed(2)}, {normalized.awayTeam} {selectedPrediction.expectedGoals.away.toFixed(2)}.
            </p>
            <p className="panel-note predicted-score">
              <strong>Predicted Score</strong>
              <span>{selectedPrediction.mostLikelyScore[0]}–{selectedPrediction.mostLikelyScore[1]}</span>
              <em>{formatPercent(selectedPrediction.mostLikelyScoreProbability, 1)}</em>
            </p>
            <ScoreHeatmap
              prediction={selectedPrediction}
              homeTeam={normalized.homeTeam}
              awayTeam={normalized.awayTeam}
            />
          </section>

          <RecentTeamMatches
            match={normalized}
            matches={matches}
            predictionDate={selectedVersion.predictionDate}
            teams={teams}
          />
        </div>

        <footer className="drawer-footer">
          <div className="drawer-sources">
            <a href={selectedVersion.sourceUrl} target="_blank" rel="noreferrer">
              Prediction generated {selectedVersion.predictionDate} <span aria-hidden="true">↗</span>
            </a>
            {normalized.predictionHistory.length > 0 && (
              <div className="prediction-history">
                <strong>Previous predictions</strong>
                <ul>
                  {normalized.predictionHistory.map((historical) => (
                    <li key={historical.predictionDate}>
                      <a href={historical.sourceUrl} target="_blank" rel="noreferrer">
                        {historical.predictionDate} <span aria-hidden="true">↗</span>
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
          <span>{modelName}</span>
        </footer>
      </section>
    </div>
  );
}
