import type { MatchPrediction } from "../types";
import { formatPercent, isMatchCompleted } from "../utils";

function formatDelta(market: number, model: number): string {
  const points = (market - model) * 100;
  return `${points >= 0 ? "+" : ""}${points.toFixed(1)} pp`;
}

function comparisonRows(match: MatchPrediction) {
  if (!match.polymarket) return [];
  const model = match.prediction.probabilities;
  const market = match.polymarket;
  return [
    { label: `${match.homeTeam} win`, model: model.homeWin, market: market.homeWin },
    { label: "Draw", model: model.draw, market: market.draw },
    { label: `${match.awayTeam} win`, model: model.awayWin, market: market.awayWin },
  ];
}

export function PolymarketCompact({ match }: { match: MatchPrediction }) {
  if (!match.polymarket || isMatchCompleted(match)) return null;
  const market = match.polymarket;
  return (
    <div className="polymarket-compact" aria-label="Polymarket probabilities">
      <a className="polymarket-compact__heading" href={market.eventUrl} target="_blank" rel="noreferrer">Polymarket ↗</a>
      <div className="probability-strip" aria-label="Polymarket result probabilities">
        <span className="probability-strip__first-team" style={{ width: `${market.homeWin * 100}%` }} />
        <span className="probability-strip__draw" style={{ width: `${market.draw * 100}%` }} />
        <span className="probability-strip__second-team" style={{ width: `${market.awayWin * 100}%` }} />
      </div>
      <div className="polymarket-compact__values">
        <span>{match.homeTeam} {formatPercent(market.homeWin)}</span>
        <span>Draw {formatPercent(market.draw)}</span>
        <span>{match.awayTeam} {formatPercent(market.awayWin)}</span>
      </div>
    </div>
  );
}

export function PolymarketDetail({ match }: { match: MatchPrediction }) {
  if (!match.polymarket || isMatchCompleted(match)) return null;
  const market = match.polymarket;
  const rows = comparisonRows(match);
  return (
    <section className="drawer-panel drawer-panel--polymarket">
      <div className="polymarket-heading">
        <h3>Model vs Polymarket</h3>
        <a href={market.eventUrl} target="_blank" rel="noreferrer">View market ↗</a>
      </div>
      <table className="polymarket-table">
        <thead><tr><th>Outcome</th><th>Model</th><th>Market</th><th>Difference</th></tr></thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <th scope="row">{row.label}</th>
              <td>{formatPercent(row.model, 1)}</td>
              <td>{formatPercent(row.market, 1)}</td>
              <td>{formatDelta(row.market, row.model)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="panel-note">Market-implied probabilities are live prices and may not total exactly 100%.</p>
    </section>
  );
}

export function PolymarketCardComparison({ match }: { match: MatchPrediction }) {
  if (!match.polymarket || isMatchCompleted(match)) return null;
  const market = match.polymarket;
  const rows = comparisonRows(match);
  return (
    <section className="polymarket-card-comparison" aria-label="Model versus Polymarket probabilities">
      <div className="polymarket-card-comparison__heading">
        <h3>Model vs Polymarket</h3>
        <a href={market.eventUrl} target="_blank" rel="noreferrer">View market ↗</a>
      </div>
      <table className="polymarket-table polymarket-table--card">
        <thead><tr><th>Outcome</th><th>Model</th><th>Market</th><th>Difference</th></tr></thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <th scope="row">{row.label}</th>
              <td>{formatPercent(row.model, 1)}</td>
              <td>{formatPercent(row.market, 1)}</td>
              <td>{formatDelta(row.market, row.model)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>Market-implied probabilities are live prices and may not total exactly 100%.</p>
    </section>
  );
}
