import { useId, useRef, useState, type KeyboardEvent } from "react";
import type { Player } from "../types";

interface PlayerNameProps {
  name: string;
  player?: Player;
  className?: string;
}

export function formatPlayerBirthDate(dateOfBirth: string) {
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${dateOfBirth}T00:00:00Z`));
}

export function PlayerName({ name, player, className = "" }: PlayerNameProps) {
  const [open, setOpen] = useState(false);
  const profileId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);

  if (!player) {
    return <span className={className}>{name}</span>;
  }

  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      setOpen(false);
    }
  }

  return (
    <span
      className={`player-name ${className}`.trim()}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => {
        if (document.activeElement !== triggerRef.current) setOpen(false);
      }}
    >
      <button
        type="button"
        ref={triggerRef}
        className="player-name__trigger"
        aria-expanded={open}
        aria-describedby={open ? profileId : undefined}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((current) => !current)}
        onFocus={() => setOpen(true)}
        onKeyDown={handleKeyDown}
        onMouseDown={(event) => event.preventDefault()}
      >
        {name}
      </button>
      {open && (
        <span className="player-profile" id={profileId} role="tooltip">
          <strong>{player.name}</strong>
          <span>No. {player.number} · {player.position}</span>
          <span>Born {formatPlayerBirthDate(player.dateOfBirth)}</span>
        </span>
      )}
    </span>
  );
}
