import { useMemo, useState, type MouseEvent } from "react";
import type { GroupProjection, MatchPrediction, Team } from "../types";
import { formatKickoffParts, getActualGroupStats, isMatchCompleted, isMatchOngoing } from "../utils";
import { MatchScoreComparison } from "./MatchScoreComparison";
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
  const actualStats = getActualGroupStats(groupMatches);

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
          <colgroup>
            <col className="standings-table__rank-column" />
            <col className="standings-table__team-column" />
            {Array.from({ length: 10 }, (_, index) => (
              <col className="standings-table__stat-column" key={index} />
            ))}
          </colgroup>
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">Team</th>
              <th scope="col" title="Games played">G</th>
              <th scope="col" title="Wins">W</th>
              <th scope="col" title="Draws">D</th>
              <th scope="col" title="Losses">L</th>
              <th scope="col" title="Points">PTS</th>
              <th scope="col" title="Goal difference">GD</th>
              <th scope="col" title="Goals scored">GS</th>
              <th className="standings-table__projection-start" scope="col" title="Expected points">xPts</th>
              <th scope="col" title="Expected goal difference">xGD</th>
              <th scope="col" title="Expected goals for">xGF</th>
            </tr>
          </thead>
          <tbody>
            {group.projection.map((row) => {
              const stats = actualStats[row.team] ?? {
                games: 0,
                wins: 0,
                draws: 0,
                losses: 0,
                points: 0,
                goalDifference: 0,
                goalsScored: 0,
              };
              return (
                <tr key={row.team}>
                  <td>
                    <span className={`rank-marker rank-marker--${row.rank}`}>{row.rank}</span>
                  </td>
                  <th scope="row"><TeamFlag team={teams[row.team]} compact /></th>
                  <td>{stats.games}</td>
                  <td>{stats.wins}</td>
                  <td>{stats.draws}</td>
                  <td>{stats.losses}</td>
                  <td>{stats.points}</td>
                  <td>{stats.goalDifference > 0 ? "+" : ""}{stats.goalDifference}</td>
                  <td>{stats.goalsScored}</td>
                  <td className="standings-table__projection-start">{row.expectedPoints.toFixed(1)}</td>
                  <td>{row.expectedGoalDifference > 0 ? "+" : ""}{row.expectedGoalDifference.toFixed(1)}</td>
                  <td>{row.expectedGoalsFor.toFixed(1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="group-fixtures">
        {groupMatches.map((match) => {
          const kickoff = formatKickoffParts(match.kickoffUtc);
          const completed = isMatchCompleted(match);
          const ongoing = isMatchOngoing(match);
          return (
            <div
              className="fixture-row"
              key={match.id}
            >
              <span className="fixture-row__teams">
                <TeamFlag team={teams[match.homeTeam]} compact />
                <span className="fixture-row__score-area">
                  <span className="fixture-row__date">
                    {kickoff.date}
                    <small>{kickoff.time}</small>
                    {completed && <strong className="fixture-row__status">Played</strong>}
                    {ongoing && <strong className="fixture-row__status fixture-row__status--live">Live</strong>}
                  </span>
                  <MatchScoreComparison match={match} teams={teams} variant="group" showScorers />
                </span>
                <TeamFlag team={teams[match.awayTeam]} compact />
              </span>
              <button
                className="fixture-row__prediction"
                type="button"
                onClick={(event) => handleOpen(match, event)}
              >
                <span className="sr-only">Open prediction for {match.homeTeam} versus {match.awayTeam}</span>
                <span aria-hidden="true">→</span>
              </button>
            </div>
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
          G, W, D, L, PTS, GD, and GS reflect results available from the OpenFootball schedule. PTS awards three points for a win and one for a draw, GD is goal difference, and GS is goals scored. xPTS means expected points, xGD expected goal difference, and xGF expected goals scored. Rankings remain model projections, not official standings. The top two in each group and eight best third-place teams advance.
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
