import type {
  GroupActualStats,
  KnockoutMatch,
  KnockoutRound,
  MatchPrediction,
  ResultProbabilities,
} from "./types";

export const KNOCKOUT_ROUNDS: KnockoutRound[] = [
  "Round of 32",
  "Round of 16",
  "Quarter-final",
  "Semi-final",
  "Match for third place",
  "Final",
];

export const ROUND_LABELS: Record<KnockoutRound, string> = {
  "Round of 32": "Round of 32",
  "Round of 16": "Round of 16",
  "Quarter-final": "Quarter-finals",
  "Semi-final": "Semi-finals",
  "Match for third place": "Third place",
  Final: "Final",
};

export function formatPercent(value: number, digits = 0): string {
  return new Intl.NumberFormat(undefined, {
    style: "percent",
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

export function formatKickoff(kickoffUtc: string): string {
  return new Intl.DateTimeFormat(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  }).format(new Date(kickoffUtc));
}

export function formatKickoffParts(kickoffUtc: string): {
  date: string;
  time: string;
} {
  const date = new Date(kickoffUtc);
  return {
    date: new Intl.DateTimeFormat(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
    }).format(date),
    time: new Intl.DateTimeFormat(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      timeZoneName: "short",
    }).format(date),
  };
}

export function getUpcomingMatches(
  matches: MatchPrediction[],
  now = new Date(),
  limit?: number,
): MatchPrediction[] {
  const nowTime = now.getTime();
  const upcoming = [...matches]
    .filter((match) => new Date(match.kickoffUtc).getTime() > nowTime)
    .sort(
      (left, right) =>
        new Date(left.kickoffUtc).getTime() -
        new Date(right.kickoffUtc).getTime(),
    );
  return limit === undefined ? upcoming : upcoming.slice(0, limit);
}

const ASSUMED_MATCH_DURATION_MS = 2 * 60 * 60 * 1000;

interface KickoffLike {
  kickoffUtc: string;
}

export function isMatchCompleted(match: KickoffLike, now = new Date()): boolean {
  return new Date(match.kickoffUtc).getTime() + ASSUMED_MATCH_DURATION_MS <= now.getTime();
}

export function isMatchOngoing(match: KickoffLike, now = new Date()): boolean {
  const nowTime = now.getTime();
  const kickoffTime = new Date(match.kickoffUtc).getTime();
  const endTime = kickoffTime + ASSUMED_MATCH_DURATION_MS;
  return kickoffTime <= nowTime && nowTime < endTime;
}

export function getOngoingMatches(
  matches: MatchPrediction[],
  now = new Date(),
): MatchPrediction[] {
  const nowTime = now.getTime();
  return [...matches]
    .filter((match) => {
      const kickoffTime = new Date(match.kickoffUtc).getTime();
      const endTime = kickoffTime + ASSUMED_MATCH_DURATION_MS;
      return kickoffTime <= nowTime && nowTime < endTime;
    })
    .sort(
      (left, right) =>
        new Date(left.kickoffUtc).getTime() -
        new Date(right.kickoffUtc).getTime(),
    );
}

export function getCompletedMatches(
  matches: MatchPrediction[],
  now = new Date(),
): MatchPrediction[] {
  return [...matches]
    .filter((match) => isMatchCompleted(match, now))
    .sort(
      (left, right) =>
        new Date(right.kickoffUtc).getTime() -
        new Date(left.kickoffUtc).getTime(),
    );
}

export function getCompletedKnockoutMatches(
  matches: KnockoutMatch[],
  now = new Date(),
): KnockoutMatch[] {
  return [...matches]
    .filter((match) => isMatchCompleted(match, now) && match.score)
    .sort(
      (left, right) =>
        new Date(right.kickoffUtc).getTime() -
        new Date(left.kickoffUtc).getTime(),
    );
}

export function getActualGroupStats(
  matches: MatchPrediction[],
): Record<string, GroupActualStats> {
  const stats: Record<string, GroupActualStats> = {};
  const emptyStats = (): GroupActualStats => ({
    games: 0,
    wins: 0,
    draws: 0,
    losses: 0,
    points: 0,
    goalDifference: 0,
    goalsScored: 0,
  });

  for (const match of matches) {
    stats[match.homeTeam] ??= emptyStats();
    stats[match.awayTeam] ??= emptyStats();

    if (!match.actualResult) continue;

    const home = stats[match.homeTeam];
    const away = stats[match.awayTeam];
    const { homeScore, awayScore } = match.actualResult;
    home.games += 1;
    away.games += 1;
    home.goalsScored += homeScore;
    away.goalsScored += awayScore;
    home.goalDifference += homeScore - awayScore;
    away.goalDifference += awayScore - homeScore;

    if (homeScore > awayScore) {
      home.wins += 1;
      home.points += 3;
      away.losses += 1;
    } else if (homeScore < awayScore) {
      away.wins += 1;
      away.points += 3;
      home.losses += 1;
    } else {
      home.draws += 1;
      away.draws += 1;
      home.points += 1;
      away.points += 1;
    }
  }

  return stats;
}

export function getClinchedTopTwoTeams(
  matches: MatchPrediction[],
  now = new Date(),
): Set<string> {
  const teams = Array.from(new Set(matches.flatMap((match) => [match.homeTeam, match.awayTeam])));
  const appearances = Object.fromEntries(teams.map((team) => [team, 0]));
  const pairs = new Set<string>();
  for (const match of matches) {
    appearances[match.homeTeam] += 1;
    appearances[match.awayTeam] += 1;
    pairs.add([match.homeTeam, match.awayTeam].sort().join("|"));
  }
  if (
    teams.length !== 4 ||
    matches.length !== 6 ||
    pairs.size !== 6 ||
    teams.some((team) => appearances[team] !== 3)
  ) {
    return new Set();
  }

  const resolved = matches.filter((match) => match.actualResult && isMatchCompleted(match, now));
  const unresolved = matches.filter((match) => !match.actualResult || !isMatchCompleted(match, now));
  const actualStats = getActualGroupStats(resolved);
  const points = Object.fromEntries(teams.map((team) => [team, actualStats[team]?.points ?? 0]));
  const clinched = new Set(teams);

  function evaluateRemaining(index: number): void {
    if (clinched.size === 0) return;
    if (index === unresolved.length) {
      for (const team of clinched) {
        const opponentsAtOrAbove = teams.filter(
          (opponent) => opponent !== team && points[opponent] >= points[team],
        ).length;
        if (opponentsAtOrAbove >= 2) clinched.delete(team);
      }
      return;
    }

    const match = unresolved[index];
    points[match.homeTeam] += 3;
    evaluateRemaining(index + 1);
    points[match.homeTeam] -= 3;

    points[match.homeTeam] += 1;
    points[match.awayTeam] += 1;
    evaluateRemaining(index + 1);
    points[match.homeTeam] -= 1;
    points[match.awayTeam] -= 1;

    points[match.awayTeam] += 3;
    evaluateRemaining(index + 1);
    points[match.awayTeam] -= 3;
  }

  evaluateRemaining(0);
  return clinched;
}

export function aggregateScoreGrid(grid: number[][]): number[][] {
  const aggregate = Array.from({ length: 6 }, () => Array(6).fill(0));
  grid.forEach((row, homeGoals) => {
    row.forEach((probability, awayGoals) => {
      aggregate[Math.min(homeGoals, 5)][Math.min(awayGoals, 5)] += probability;
    });
  });
  return aggregate;
}

export function mostLikelyOutcome(
  probabilities: ResultProbabilities,
  homeTeam: string,
  awayTeam: string,
): string {
  const outcomes = [
    { label: `${homeTeam} win`, value: probabilities.homeWin },
    { label: "Draw", value: probabilities.draw },
    { label: `${awayTeam} win`, value: probabilities.awayWin },
  ];
  return outcomes.sort((left, right) => right.value - left.value)[0].label;
}

export function describeBracketSlot(slot: string): string {
  const groupPosition = slot.match(/^([12])([A-L])$/);
  if (groupPosition) {
    return `${groupPosition[1] === "1" ? "Winner" : "Runner-up"} Group ${groupPosition[2]}`;
  }
  const thirdPlace = slot.match(/^3([A-L/]+)$/);
  if (thirdPlace) {
    return `Best third-place team from ${thirdPlace[1].split("/").join(", ")}`;
  }
  const feeder = slot.match(/^([WL])(\d+)$/);
  if (feeder) {
    return `${feeder[1] === "W" ? "Winner" : "Loser"} Match ${feeder[2]}`;
  }
  return slot;
}

export function getKnockoutWinner(match: KnockoutMatch): string | null {
  if (!match.score) return null;
  
  // Use penalties if available, then extra time, then full time
  const finalScore = match.score.penalties ?? match.score.extraTime ?? match.score.fullTime;
  if (!finalScore) return null;
  
  const [homeScore, awayScore] = finalScore;
  const homeTeam = match.team1;
  const awayTeam = match.team2;
  
  if (!homeTeam || !awayTeam) return null;
  
  if (homeScore > awayScore) return homeTeam;
  if (awayScore > homeScore) return awayTeam;
  return null; // Draw (shouldn't happen in knockout)
}
