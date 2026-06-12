import { useMemo, useRef, useState } from "react";
import type { Team } from "../types";
import { formatPlayerBirthDate, PlayerName } from "./PlayerName";
import { TeamFlag } from "./TeamFlag";

interface CountriesProps {
  teams: Record<string, Team>;
}

export function Countries({ teams }: CountriesProps) {
  const countries = useMemo(
    () => Object.values(teams).sort((left, right) => left.name.localeCompare(right.name)),
    [teams],
  );
  const [selectedGroup, setSelectedGroup] = useState("All");
  const [selectedTeamName, setSelectedTeamName] = useState(countries[0]?.name ?? "");
  const rosterRef = useRef<HTMLElement>(null);
  const visibleCountries = selectedGroup === "All"
    ? countries
    : countries.filter((team) => team.group === selectedGroup);
  const selectedTeam = teams[selectedTeamName] ?? visibleCountries[0];

  function selectGroup(group: string) {
    const firstTeam = group === "All"
      ? countries[0]
      : countries.find((team) => team.group === group);
    setSelectedGroup(group);
    if (firstTeam) setSelectedTeamName(firstTeam.name);
  }

  function selectTeam(teamName: string) {
    setSelectedTeamName(teamName);
    if (window.matchMedia?.("(max-width: 1050px)").matches) {
      requestAnimationFrame(() => rosterRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    }
  }

  return (
    <section className="section section--countries" id="countries" aria-labelledby="countries-title">
      <div className="section-heading section-heading--stacked">
        <div>
          <span className="eyebrow">48 national squads</span>
          <h2 id="countries-title">Countries</h2>
        </div>
        <p>
          Select a country to explore its World Cup squad. Hover, focus, or tap a player name to see their profile.
        </p>
      </div>

      <nav className="group-filter" aria-label="Filter countries by group">
        {["All", ..."ABCDEFGHIJKL"].map((group) => (
          <button
            type="button"
            key={group}
            className={selectedGroup === group ? "is-active" : ""}
            aria-pressed={selectedGroup === group}
            onClick={() => selectGroup(group)}
          >
            {group === "All" ? "All groups" : group}
          </button>
        ))}
      </nav>

      <div className="countries-layout">
        <div className="country-grid" aria-label="Countries">
          {visibleCountries.map((team) => (
            <button
              type="button"
              className={`country-card${selectedTeam?.name === team.name ? " is-active" : ""}`}
              key={team.name}
              aria-pressed={selectedTeam?.name === team.name}
              aria-controls="country-roster"
              onClick={() => selectTeam(team.name)}
            >
              <TeamFlag team={team} compact />
              <span className="country-card__meta">
                <strong>{team.fifaCode}</strong>
                <span>Group {team.group} · {team.players.length} players</span>
              </span>
            </button>
          ))}
        </div>

        {selectedTeam && (
          <article
            className="country-roster"
            id="country-roster"
            aria-labelledby="country-roster-title"
            ref={rosterRef}
          >
            <header className="country-roster__header">
              <TeamFlag team={selectedTeam} />
              <div>
                <span className="eyebrow">Group {selectedTeam.group} · {selectedTeam.fifaCode}</span>
                <h3 id="country-roster-title">{selectedTeam.name} squad</h3>
              </div>
            </header>
            <div className="roster-table-wrap">
              <table className="roster-table">
                <caption>{selectedTeam.name} player roster</caption>
                <thead>
                  <tr>
                    <th scope="col">No.</th>
                    <th scope="col">Pos.</th>
                    <th scope="col">Player</th>
                    <th scope="col">Born</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedTeam.players.map((player) => (
                    <tr key={player.number}>
                      <td>{player.number}</td>
                      <td><span className="position-badge">{player.position}</span></td>
                      <th scope="row"><PlayerName name={player.name} player={player} /></th>
                      <td>{formatPlayerBirthDate(player.dateOfBirth)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        )}
      </div>
    </section>
  );
}
