import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { KnockoutMatch, KnockoutRound, Team } from "../types";
import { describeBracketSlot, formatKickoff, KNOCKOUT_ROUNDS, ROUND_LABELS } from "../utils";
import { TeamFlag } from "./TeamFlag";

interface KnockoutBracketProps {
  matches: KnockoutMatch[];
  teams?: Record<string, Team>;
}

type View = "bracket" | "list";
type RoundFilter = "All" | KnockoutRound;

const BRACKET_ROUNDS: KnockoutRound[] = [
  "Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final",
];

const BRACKET_MATCH_ORDER: Partial<Record<KnockoutRound, number[]>> = {
  "Round of 32": [74, 77, 73, 75, 83, 84, 81, 82, 76, 78, 79, 80, 86, 88, 85, 87],
  "Round of 16": [89, 90, 93, 94, 91, 92, 95, 96],
  "Quarter-final": [97, 98, 99, 100],
  "Semi-final": [101, 102],
  Final: [104],
};

function participant(match: KnockoutMatch, side: 1 | 2): { name?: string; slot: string } {
  return side === 1
    ? { name: match.team1, slot: match.team1Slot }
    : { name: match.team2, slot: match.team2Slot };
}

function finalScore(match: KnockoutMatch): [number, number] | null {
  return match.score?.extraTime ?? match.score?.fullTime ?? null;
}

function winnerSide(match: KnockoutMatch): 1 | 2 | null {
  const score = match.score?.penalties ?? finalScore(match);
  if (!score || score[0] === score[1]) return null;
  return score[0] > score[1] ? 1 : 2;
}

function Participant({ match, side, teams }: { match: KnockoutMatch; side: 1 | 2; teams?: Record<string, Team> }) {
  const value = participant(match, side);
  const winner = winnerSide(match) === side;
  return (
    <span className={`bracket-slot${winner ? " bracket-slot--winner" : ""}`}>
      {value.name && teams?.[value.name] && <TeamFlag team={teams[value.name]} compact />}
      <span>
        <strong>{value.name ?? value.slot}{winner ? " · Winner" : ""}</strong>
        <small>{describeBracketSlot(value.slot)}</small>
      </span>
    </span>
  );
}

function Score({ match }: { match: KnockoutMatch }) {
  const score = finalScore(match);
  if (!score) return null;
  return (
    <span className="bracket-match__score">
      {score[0]}–{score[1]}
      {match.score?.penalties && <small>pens {match.score.penalties[0]}–{match.score.penalties[1]}</small>}
      {!match.score?.penalties && match.score?.extraTime && <small>AET</small>}
    </span>
  );
}

function BracketMatchCard({ match, teams }: {
  match: KnockoutMatch;
  teams?: Record<string, Team>;
}) {
  return (
    <article className="bracket-match" data-match-number={match.matchNumber}>
      <span className="bracket-match__number">M{match.matchNumber}</span>
      <Participant match={match} side={1} teams={teams} />
      <span className="bracket-match__divider" />
      <Participant match={match} side={2} teams={teams} />
      <span className="bracket-match__footer"><span>{formatKickoff(match.kickoffUtc)}</span><Score match={match} /></span>
    </article>
  );
}

export function KnockoutBracket({ matches, teams }: KnockoutBracketProps) {
  const [view, setView] = useState<View>("bracket");
  const [selectedRound, setSelectedRound] = useState<KnockoutRound>("Round of 32");
  const [listRound, setListRound] = useState<RoundFilter>("All");
  const bracketRef = useRef<HTMLDivElement>(null);
  const [connectorPaths, setConnectorPaths] = useState<string[]>([]);
  const [connectorSize, setConnectorSize] = useState({ width: 0, height: 0 });
  const matchesByRound = useMemo(() => new Map(
    KNOCKOUT_ROUNDS.map((round) => {
      const order = BRACKET_MATCH_ORDER[round];
      const roundMatches = matches.filter((match) => match.round === round);
      return [round, order
        ? order.map((number) => roundMatches.find((match) => match.matchNumber === number)!).filter(Boolean)
        : roundMatches];
    }),
  ), [matches]);

  useLayoutEffect(() => {
    const bracket = bracketRef.current;
    if (!bracket) return;

    function drawConnectors() {
      const root = bracketRef.current;
      if (!root) return;
      const rootRect = root.getBoundingClientRect();
      const cards = new Map<number, DOMRect>();
      root.querySelectorAll<HTMLElement>(".bracket-match[data-match-number]").forEach((card) => {
        cards.set(Number(card.dataset.matchNumber), card.getBoundingClientRect());
      });
      const paths: string[] = [];
      for (const match of matches) {
        if (!BRACKET_ROUNDS.includes(match.round) || match.round === "Round of 32") continue;
        const target = cards.get(match.matchNumber);
        if (!target) continue;
        for (const slot of [match.team1Slot, match.team2Slot]) {
          const feeder = slot.match(/^[WL](\d+)$/);
          if (!feeder) continue;
          const source = cards.get(Number(feeder[1]));
          if (!source) continue;
          const sourceX = source.right - rootRect.left;
          const sourceY = source.top + source.height / 2 - rootRect.top;
          const targetX = target.left - rootRect.left;
          const targetY = target.top + target.height / 2 - rootRect.top;
          const middleX = sourceX + (targetX - sourceX) / 2;
          paths.push(`M ${sourceX} ${sourceY} H ${middleX} V ${targetY} H ${targetX}`);
        }
      }
      setConnectorPaths(paths);
      setConnectorSize({ width: root.scrollWidth, height: root.scrollHeight });
    }

    const frame = window.requestAnimationFrame(drawConnectors);
    window.addEventListener("resize", drawConnectors);
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(drawConnectors);
    observer?.observe(bracket);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", drawConnectors);
      observer?.disconnect();
    };
  }, [matches]);

  function selectRound(round: KnockoutRound) {
    setSelectedRound(round);
  }

  if (matches.length === 0) return null;
  const visibleRounds = listRound === "All" ? KNOCKOUT_ROUNDS : [listRound];

  return (
    <section className="section section--finals" id="playoffs" aria-labelledby="playoffs-title">
      <div className="section-heading section-heading--stacked">
        <div><span className="eyebrow">Official tournament path</span><h2 id="playoffs-title">Playoffs</h2></div>
        <div className="section-heading__tools">
          <p>Kickoff times use your local timezone. Teams and results refresh from OpenFootball.</p>
          <div className="view-toggle" role="group" aria-label="Playoffs view">
            {(["bracket", "list"] as View[]).map((candidate) => (
              <button key={candidate} type="button" className="view-toggle__button"
                aria-pressed={view === candidate} onClick={() => setView(candidate)}>
                {candidate === "bracket" ? "Bracket" : "List"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {view === "bracket" ? <>
        <nav className="round-filter" aria-label="Select knockout round">
          {KNOCKOUT_ROUNDS.map((round) => <button type="button" key={round}
            className={selectedRound === round ? "is-active" : ""} aria-pressed={selectedRound === round}
            onClick={() => selectRound(round)}>{ROUND_LABELS[round]}</button>)}
        </nav>
        <div className="bracket-desktop" role="region" aria-label="Playoff bracket" tabIndex={0} ref={bracketRef}>
          <svg className="bracket-connectors" width={connectorSize.width} height={connectorSize.height} aria-hidden="true">
            {connectorPaths.map((path, index) => <path key={`${index}-${path}`} d={path} />)}
          </svg>
          {BRACKET_ROUNDS.map((round) => <section className={`bracket-round bracket-round--${round.replaceAll(" ", "-").toLowerCase()}`} key={round}>
            <h3>{ROUND_LABELS[round]}</h3>
            <div className="bracket-round__matches">{matchesByRound.get(round)?.map((match) =>
              <BracketMatchCard key={match.id} match={match} teams={teams} />)}</div>
          </section>)}
        </div>
        <div className="playoff-third-place">
          <h3>{ROUND_LABELS["Match for third place"]}</h3>
          {matchesByRound.get("Match for third place")?.map((match) =>
            <BracketMatchCard key={match.id} match={match} teams={teams} />)}
        </div>
        <div className="bracket-mobile" aria-live="polite">{matchesByRound.get(selectedRound)?.map((match) =>
          <BracketMatchCard key={match.id} match={match} teams={teams} />)}</div>
      </> : <>
        <nav className="playoff-list-filter" aria-label="Filter playoff list">
          {(["All", ...KNOCKOUT_ROUNDS] as RoundFilter[]).map((round) => <button type="button" key={round}
            className={listRound === round ? "is-active" : ""} aria-pressed={listRound === round}
            onClick={() => setListRound(round)}>{round === "All" ? "All rounds" : ROUND_LABELS[round]}</button>)}
        </nav>
        <div className="playoff-list" role="region" aria-label="Playoff fixtures in list view">
          {visibleRounds.map((round) => <section key={round} className="playoff-list__round">
            <h3>{ROUND_LABELS[round]}</h3>
            {matchesByRound.get(round)?.map((match) => <article key={match.id} className="playoff-list-row">
              <span className="playoff-list-row__meta"><strong>M{match.matchNumber}</strong><small>{formatKickoff(match.kickoffUtc)}</small></span>
              <span className="playoff-list-row__teams"><Participant match={match} side={1} teams={teams} /><Participant match={match} side={2} teams={teams} /></span>
              <span className="playoff-list-row__score"><Score match={match} /></span>
              <span className="playoff-list-row__venue">{match.venue}</span>
            </article>)}
          </section>)}
        </div>
      </>}
    </section>
  );
}
