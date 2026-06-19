import { useEffect, useState } from "react";
import type { ActualGoal, ActualResult, KnockoutMatch, KnockoutScore, MatchPrediction, Team } from "./types";

export type LiveResultsStatus = "loading" | "current" | "fallback";

interface LiveResultsState {
  matches: MatchPrediction[];
  knockoutMatches: KnockoutMatch[];
  status: LiveResultsStatus;
  lastCheckedAt: string | null;
}

interface OpenFootballGoal {
  name: string;
  minute: string | number;
  penalty?: boolean;
}

interface OpenFootballFixture {
  num?: number;
  round?: string;
  date: string;
  time?: string;
  ground?: string;
  team1: string;
  team2: string;
  score?: { ft?: [number, number]; et?: [number, number]; p?: [number, number] };
  goals1?: OpenFootballGoal[];
  goals2?: OpenFootballGoal[];
}

interface OpenFootballSchedule {
  matches: OpenFootballFixture[];
}

const TEAM_ALIASES: Record<string, string> = {
  "Bosnia & Herzegovina": "Bosnia and Herzegovina",
  USA: "United States",
};

const FIVE_MINUTES_MS = 5 * 60 * 1000;
let scheduleRequest: Promise<OpenFootballSchedule> | null = null;

function canonicalTeam(name: string): string {
  return TEAM_ALIASES[name] ?? name;
}

function fixtureKey(date: string, team1: string, team2: string): string {
  return `${date}|${[canonicalTeam(team1), canonicalTeam(team2)].sort().join("|")}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isScore(value: unknown): value is [number, number] {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    value.every((score) => Number.isInteger(score) && score >= 0)
  );
}

function parseGoal(value: unknown): OpenFootballGoal | null {
  if (!isRecord(value) || typeof value.name !== "string") return null;
  if (typeof value.minute !== "string" && typeof value.minute !== "number") return null;
  if (value.penalty !== undefined && typeof value.penalty !== "boolean") return null;
  return {
    name: value.name,
    minute: value.minute,
    ...(value.penalty === undefined ? {} : { penalty: value.penalty }),
  };
}

function parseFixture(value: unknown): OpenFootballFixture | null {
  if (
    !isRecord(value) ||
    typeof value.date !== "string" ||
    typeof value.team1 !== "string" ||
    typeof value.team2 !== "string"
  ) {
    return null;
  }

  const fixture: OpenFootballFixture = {
    date: value.date,
    team1: value.team1,
    team2: value.team2,
  };
  if (value.num !== undefined) {
    if (!Number.isInteger(value.num)) return null;
    fixture.num = value.num as number;
  }
  for (const key of ["round", "time", "ground"] as const) {
    if (value[key] !== undefined) {
      if (typeof value[key] !== "string") return null;
      fixture[key] = value[key] as string;
    }
  }
  if (isRecord(value.score) && isScore(value.score.ft)) {
    fixture.score = { ft: value.score.ft };
    if (value.score.et !== undefined) {
      if (!isScore(value.score.et)) return null;
      fixture.score.et = value.score.et;
    }
    if (value.score.p !== undefined) {
      if (!isScore(value.score.p)) return null;
      fixture.score.p = value.score.p;
    }
  }
  if (Array.isArray(value.goals1)) {
    const goals = value.goals1.map(parseGoal);
    if (goals.some((goal) => goal === null)) return null;
    fixture.goals1 = goals as OpenFootballGoal[];
  }
  if (Array.isArray(value.goals2)) {
    const goals = value.goals2.map(parseGoal);
    if (goals.some((goal) => goal === null)) return null;
    fixture.goals2 = goals as OpenFootballGoal[];
  }
  return fixture;
}

export function parseOpenFootballSchedule(value: unknown): OpenFootballSchedule {
  if (!isRecord(value) || !Array.isArray(value.matches)) {
    throw new Error("OpenFootball schedule is missing a matches array");
  }
  const matches = value.matches.map(parseFixture);
  if (matches.some((fixture) => fixture === null)) {
    throw new Error("OpenFootball schedule contains an invalid fixture");
  }
  return { matches: matches as OpenFootballFixture[] };
}

function normalizeName(name: string): string {
  return name.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase().trim();
}

function convertGoals(goals: OpenFootballGoal[], team?: Team): ActualGoal[] {
  return goals.map((goal) => {
    const normalized = normalizeName(goal.name);
    const squadMatches = team?.players.filter((player) => normalizeName(player.name) === normalized) ?? [];
    return {
      name: squadMatches.length === 1 ? squadMatches[0].name : goal.name,
      minute: String(goal.minute),
      penalty: goal.penalty ?? null,
    };
  });
}

function actualResultFor(
  fixture: OpenFootballFixture,
  match: MatchPrediction,
  teams: Record<string, Team>,
): ActualResult | null {
  const score = fixture.score?.ft;
  if (!score) return null;

  const team1 = canonicalTeam(fixture.team1);
  const team2 = canonicalTeam(fixture.team2);
  const goals1 = fixture.goals1 ?? [];
  const goals2 = fixture.goals2 ?? [];

  if (team1 === match.homeTeam && team2 === match.awayTeam) {
    return {
      homeScore: score[0],
      awayScore: score[1],
      homeGoals: convertGoals(goals1, teams[match.homeTeam]),
      awayGoals: convertGoals(goals2, teams[match.awayTeam]),
    };
  }
  if (team1 === match.awayTeam && team2 === match.homeTeam) {
    return {
      homeScore: score[1],
      awayScore: score[0],
      homeGoals: convertGoals(goals2, teams[match.homeTeam]),
      awayGoals: convertGoals(goals1, teams[match.awayTeam]),
    };
  }
  return null;
}

export function mergeLiveResults(
  matches: MatchPrediction[],
  schedule: OpenFootballSchedule,
  teams: Record<string, Team>,
): MatchPrediction[] {
  const fixtures = new Map(
    schedule.matches
      .filter((fixture) => fixture.score?.ft)
      .map((fixture) => [fixtureKey(fixture.date, fixture.team1, fixture.team2), fixture]),
  );

  return matches.map((match) => {
    const fixture = fixtures.get(fixtureKey(match.date, match.homeTeam, match.awayTeam));
    if (!fixture) return match;
    const actualResult = actualResultFor(fixture, match, teams);
    return actualResult ? { ...match, actualResult } : match;
  });
}

const BRACKET_SLOT = /^(?:[12][A-L]|3[A-L/]+|[WL]\d+)$/;

function knockoutScore(fixture: OpenFootballFixture): KnockoutScore | undefined {
  if (!fixture.score?.ft) return undefined;
  return {
    fullTime: fixture.score.ft,
    ...(fixture.score.et ? { extraTime: fixture.score.et } : {}),
    ...(fixture.score.p ? { penalties: fixture.score.p } : {}),
  };
}

function parseSourceKickoff(date: string, time: string | undefined): string | null {
  if (!time) return null;
  const match = time.match(/^(\d{2}):(\d{2}) UTC([+-]\d{1,2})$/);
  if (!match) return null;
  const [, hour, minute, offsetHour] = match;
  const sign = Number(offsetHour) >= 0 ? "+" : "-";
  const offset = `${sign}${String(Math.abs(Number(offsetHour))).padStart(2, "0")}:00`;
  const parsed = new Date(`${date}T${hour}:${minute}:00${offset}`);
  return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
}

export function mergeKnockoutSchedule(
  staticMatches: KnockoutMatch[],
  schedule: OpenFootballSchedule,
): KnockoutMatch[] {
  const fixtures = new Map(
    schedule.matches
      .filter((fixture) => fixture.num !== undefined)
      .map((fixture) => [fixture.num!, fixture]),
  );
  return staticMatches.map((match) => {
    const fixture = fixtures.get(match.matchNumber);
    if (!fixture) return match;
    const kickoffUtc = parseSourceKickoff(fixture.date, fixture.time) ?? match.kickoffUtc;
    const team1 = BRACKET_SLOT.test(fixture.team1) ? undefined : canonicalTeam(fixture.team1);
    const team2 = BRACKET_SLOT.test(fixture.team2) ? undefined : canonicalTeam(fixture.team2);
    const score = knockoutScore(fixture);
    return {
      ...match,
      date: fixture.date,
      kickoffUtc,
      venue: fixture.ground ?? match.venue,
      ...(team1 ? { team1 } : {}),
      ...(team2 ? { team2 } : {}),
      ...(score ? { score } : {}),
    };
  });
}

function scheduleUrl(dataUrl: string, now = Date.now()): string {
  const url = new URL(dataUrl);
  url.searchParams.set("refresh", String(Math.floor(now / FIVE_MINUTES_MS)));
  return url.toString();
}

async function fetchSchedule(dataUrl: string): Promise<OpenFootballSchedule> {
  const response = await fetch(scheduleUrl(dataUrl), { cache: "no-store" });
  if (!response.ok) throw new Error(`OpenFootball request failed with ${response.status}`);
  return parseOpenFootballSchedule(await response.json());
}

function getSchedule(dataUrl: string): Promise<OpenFootballSchedule> {
  scheduleRequest ??= fetchSchedule(dataUrl);
  return scheduleRequest;
}

export function resetLiveResultsCacheForTests(): void {
  scheduleRequest = null;
}

export function useLiveResults(
  staticMatches: MatchPrediction[],
  dataUrl: string | undefined,
  teams: Record<string, Team>,
  staticKnockoutMatches: KnockoutMatch[] = [],
): LiveResultsState {
  const [state, setState] = useState<LiveResultsState>({
    matches: staticMatches,
    knockoutMatches: staticKnockoutMatches,
    status: dataUrl ? "loading" : "fallback",
    lastCheckedAt: null,
  });

  useEffect(() => {
    let active = true;
    if (!dataUrl) {
      setState({ matches: staticMatches, knockoutMatches: staticKnockoutMatches, status: "fallback", lastCheckedAt: null });
      return () => { active = false; };
    }

    getSchedule(dataUrl)
      .then((schedule) => {
        if (!active) return;
        setState({
          matches: mergeLiveResults(staticMatches, schedule, teams),
          knockoutMatches: mergeKnockoutSchedule(staticKnockoutMatches, schedule),
          status: "current",
          lastCheckedAt: new Date().toISOString(),
        });
      })
      .catch(() => {
        if (!active) return;
        setState({
          matches: staticMatches,
          knockoutMatches: staticKnockoutMatches,
          status: "fallback",
          lastCheckedAt: new Date().toISOString(),
        });
      });

    return () => { active = false; };
  }, [dataUrl, staticMatches, staticKnockoutMatches, teams]);

  return state;
}
