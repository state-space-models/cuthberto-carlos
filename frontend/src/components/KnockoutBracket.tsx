import { useMemo, useState } from "react";
import type { KnockoutMatch, KnockoutRound } from "../types";
import {
  describeBracketSlot,
  formatKickoff,
  KNOCKOUT_ROUNDS,
  ROUND_LABELS,
} from "../utils";

interface KnockoutBracketProps {
  matches: KnockoutMatch[];
}

function BracketSlot({ slot }: { slot: string }) {
  return (
    <span className="bracket-slot">
      <strong>{slot}</strong>
      <small>{describeBracketSlot(slot)}</small>
    </span>
  );
}

function BracketMatchCard({
  match,
  selected,
  onSelect,
}: {
  match: KnockoutMatch;
  selected: boolean;
  onSelect: (match: KnockoutMatch) => void;
}) {
  return (
    <button
      type="button"
      className={`bracket-match${selected ? " is-selected" : ""}`}
      onClick={() => onSelect(match)}
      aria-pressed={selected}
    >
      <span className="bracket-match__number">M{match.matchNumber}</span>
      <BracketSlot slot={match.team1Slot} />
      <span className="bracket-match__divider" />
      <BracketSlot slot={match.team2Slot} />
      <span className="bracket-match__date">{formatKickoff(match.kickoffUtc)}</span>
    </button>
  );
}

export function KnockoutBracket({ matches }: KnockoutBracketProps) {
  const [selectedRound, setSelectedRound] = useState<KnockoutRound>("Round of 32");
  const [selectedMatch, setSelectedMatch] = useState<KnockoutMatch>(matches[0]);
  const matchesByRound = useMemo(
    () => new Map(KNOCKOUT_ROUNDS.map((round) => [round, matches.filter((match) => match.round === round)])),
    [matches],
  );

  function selectRound(round: KnockoutRound) {
    setSelectedRound(round);
    const firstMatch = matchesByRound.get(round)?.[0];
    if (firstMatch) setSelectedMatch(firstMatch);
  }

  return (
    <section className="section section--finals" id="finals" aria-labelledby="finals-title">
      <div className="section-heading section-heading--stacked section-heading--light">
        <div>
          <span className="eyebrow">Official tournament path</span>
          <h2 id="finals-title">Finals</h2>
        </div>
        <p>
          Knockout predictions are intentionally withheld until teams qualify. Select any match to inspect its official feeder slots, kickoff, and venue.
        </p>
      </div>

      <nav className="round-filter" aria-label="Select knockout round">
        {KNOCKOUT_ROUNDS.map((round) => (
          <button
            type="button"
            key={round}
            className={selectedRound === round ? "is-active" : ""}
            aria-pressed={selectedRound === round}
            onClick={() => selectRound(round)}
          >
            {ROUND_LABELS[round]}
          </button>
        ))}
      </nav>

      <div className="bracket-desktop" aria-label="Knockout bracket">
        {KNOCKOUT_ROUNDS.map((round) => (
          <section className={`bracket-round bracket-round--${round.replaceAll(" ", "-").toLowerCase()}`} key={round}>
            <h3>{ROUND_LABELS[round]}</h3>
            <div className="bracket-round__matches">
              {matchesByRound.get(round)?.map((match) => (
                <BracketMatchCard
                  key={match.id}
                  match={match}
                  selected={selectedMatch.id === match.id}
                  onSelect={setSelectedMatch}
                />
              ))}
            </div>
          </section>
        ))}
      </div>

      <div className="bracket-mobile" aria-live="polite">
        {matchesByRound.get(selectedRound)?.map((match) => (
          <BracketMatchCard
            key={match.id}
            match={match}
            selected={selectedMatch.id === match.id}
            onSelect={setSelectedMatch}
          />
        ))}
      </div>

      <aside className="knockout-detail" aria-label={`Details for Match ${selectedMatch.matchNumber}`}>
        <div>
          <span className="eyebrow">Match {selectedMatch.matchNumber} · {ROUND_LABELS[selectedMatch.round]}</span>
          <h3>{selectedMatch.team1Slot} vs {selectedMatch.team2Slot}</h3>
        </div>
        <dl>
          <div><dt>Kickoff</dt><dd>{formatKickoff(selectedMatch.kickoffUtc)}</dd></div>
          <div><dt>Venue</dt><dd>{selectedMatch.venue}</dd></div>
          <div><dt>{selectedMatch.team1Slot}</dt><dd>{describeBracketSlot(selectedMatch.team1Slot)}</dd></div>
          <div><dt>{selectedMatch.team2Slot}</dt><dd>{describeBracketSlot(selectedMatch.team2Slot)}</dd></div>
        </dl>
      </aside>
    </section>
  );
}
