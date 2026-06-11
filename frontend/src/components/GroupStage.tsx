import { useMemo, useState, type MouseEvent } from "react";
import type { GroupProjection, MatchPrediction, Team } from "../types";
import { formatKickoffParts, formatPercent, mostLikelyOutcome } from "../utils";
import { TeamFlag } from "./TeamFlag";

interface GroupStageProps {
  groups: GroupProjection[];
  matches: MatchPrediction[];
  teams: Record<string, Team>;
  onOpen: (match: MatchPrediction, trigger: HTMLElement) => void;
}

function GroupCard({
  group,
  matchMap,
  teams,
  onOpen,
}: {
  group: GroupProjection;
  matchMap: Map<string, MatchPrediction>;
  teams: Record<string, Team>;
  onOpen: (match: MatchPrediction, trigger: HTMLElement) => void;
}) {
  const groupMatches = group.matchIds.map((id) => matchMap.get(id)).filter(Boolean) as MatchPrediction[];

  function handleOpen(match: MatchPrediction, event: MouseEvent<HTMLButtonElement>) {
    onOpen(match, event.currentTarget);
  }

  return (
    <article className="group-card" id={`group-${group.id}`}>
      <header className="group-card__header">
        <div>
          <span className="eyebrow">Projected table</span>
          <h3>{group.name}</h3>
        </div>
        <span className="group-card__badge">6 matches</span>
      </header>

      <div className="table-scroll">
        <table className="standings-table">
          <caption>Model-projected {group.name} standings</caption>
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">Team</th>
              <th scope="col" title="Expected points">xPts</th>
              <th scope="col" title="Expected goal difference">xGD</th>
              <th scope="col" title="Expected goals for">xGF</th>
            </tr>
          </thead>
          <tbody>
            {group.projection.map((row) => (
              <tr key={row.team}>
                <td>
                  <span className={`rank-marker rank-marker--${row.rank}`}>{row.rank}</span>
                </td>
                <th scope="row"><TeamFlag team={teams[row.team]} compact /></th>
                <td>{row.expectedPoints.toFixed(1)}</td>
                <td>{row.expectedGoalDifference > 0 ? "+" : ""}{row.expectedGoalDifference.toFixed(1)}</td>
                <td>{row.expectedGoalsFor.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="group-fixtures">
        {groupMatches.map((match) => {
          const kickoff = formatKickoffParts(match.kickoffUtc);
          const probabilities = match.prediction.probabilities;
          return (
            <button
              className="fixture-row"
              type="button"
              key={match.id}
              onClick={(event) => handleOpen(match, event)}
              aria-label={`Open prediction for ${match.homeTeam} versus ${match.awayTeam}`}
            >
              <span className="fixture-row__date">{kickoff.date}<small>{kickoff.time}</small></span>
              <span className="fixture-row__teams">
                <TeamFlag team={teams[match.homeTeam]} compact />
                <strong>{match.prediction.mostLikelyScore[0]}–{match.prediction.mostLikelyScore[1]}</strong>
                <TeamFlag team={teams[match.awayTeam]} compact />
              </span>
              <span className="fixture-row__pick">
                {mostLikelyOutcome(probabilities, match.homeTeam, match.awayTeam)}
                <small>
                  {formatPercent(Math.max(probabilities.homeWin, probabilities.draw, probabilities.awayWin), 1)}
                </small>
              </span>
            </button>
          );
        })}
      </div>
    </article>
  );
}

export function GroupStage({ groups, matches, teams, onOpen }: GroupStageProps) {
  const [selectedGroup, setSelectedGroup] = useState("All");
  const matchMap = useMemo(() => new Map(matches.map((match) => [match.id, match])), [matches]);
  const visibleGroups = selectedGroup === "All"
    ? groups
    : groups.filter((group) => group.id === selectedGroup);

  return (
    <section className="section section--groups" id="groups" aria-labelledby="groups-title">
      <div className="section-heading section-heading--stacked">
        <div>
          <span className="eyebrow">72 match predictions</span>
          <h2 id="groups-title">Group stage</h2>
        </div>
        <p>
          Tables use expected points and goals from every scoreline distribution. They are model projections, not official standings. The top two in each group and eight best third-place teams advance.
        </p>
      </div>

      <nav className="group-filter" aria-label="Filter group stage">
        {["All", ...groups.map((group) => group.id)].map((groupId) => (
          <button
            type="button"
            key={groupId}
            className={selectedGroup === groupId ? "is-active" : ""}
            aria-pressed={selectedGroup === groupId}
            onClick={() => setSelectedGroup(groupId)}
          >
            {groupId === "All" ? "All groups" : groupId}
          </button>
        ))}
      </nav>

      <div className="group-grid">
        {visibleGroups.map((group) => (
          <GroupCard key={group.id} group={group} matchMap={matchMap} teams={teams} onOpen={onOpen} />
        ))}
      </div>
    </section>
  );
}
