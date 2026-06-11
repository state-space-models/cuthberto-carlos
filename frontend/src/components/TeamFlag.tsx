import type { Team } from "../types";
import { flagClassNames } from "../flags";

interface TeamFlagProps {
  team: Team;
  compact?: boolean;
}

export function TeamFlag({ team, compact = false }: TeamFlagProps) {
  const flagClass = flagClassNames[team.flagCode] || "fi fi-unknown";
  
  return (
    <span className={`team-identity${compact ? " team-identity--compact" : ""}`}>
      <span className={`team-flag ${flagClass}`} aria-hidden="true" />
      <span>{team.name}</span>
    </span>
  );
}
