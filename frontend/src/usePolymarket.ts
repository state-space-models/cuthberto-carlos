import { useEffect, useState } from "react";
import type { MatchPrediction, PolymarketPrediction } from "./types";
import { isMatchCompleted } from "./utils";

export type PolymarketStatus = "loading" | "current" | "fallback";

interface PolymarketState {
  predictions: Record<string, PolymarketPrediction>;
  status: PolymarketStatus;
  lastCheckedAt: string | null;
}

interface PolymarketSource {
  dataUrl: string;
  tagId: string;
  seriesId: string;
  marketType: string;
}

const TEAM_ALIASES: Record<string, string> = {
  "Bosnia-Herzegovina": "Bosnia and Herzegovina",
  "Bosnia & Herzegovina": "Bosnia and Herzegovina",
  "Cabo Verde": "Cape Verde",
  "Côte d'Ivoire": "Ivory Coast",
  Czechia: "Czech Republic",
  "IR Iran": "Iran",
  "Korea Republic": "South Korea",
  Türkiye: "Turkey",
  USA: "United States",
};
const FIVE_MINUTES_MS = 5 * 60 * 1000;
let marketRequest: Promise<unknown[]> | null = null;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function canonicalTeam(name: string): string {
  return TEAM_ALIASES[name] ?? name;
}

function teamPairKey(team1: string, team2: string): string {
  return [canonicalTeam(team1), canonicalTeam(team2)].sort().join("|");
}

function stringList(value: unknown): string[] | null {
  if (typeof value !== "string") return null;
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) && parsed.every((item) => typeof item === "string")
      ? parsed
      : null;
  } catch {
    return null;
  }
}

function yesPrice(market: Record<string, unknown>): number | null {
  const outcomes = stringList(market.outcomes);
  const prices = stringList(market.outcomePrices);
  if (!outcomes || !prices || outcomes.length !== prices.length || outcomes.filter((item) => item === "Yes").length !== 1) {
    return null;
  }
  const price = Number(prices[outcomes.indexOf("Yes")]);
  return Number.isFinite(price) && price >= 0 && price <= 1 ? price : null;
}

interface MarketGroup {
  eventUrl: string;
  updatedAt: string;
  outcomes: Record<string, number>;
}

export function parsePolymarketMarkets(markets: unknown[]): Record<string, MarketGroup> {
  const groups: Record<string, MarketGroup> = {};
  const invalid = new Set<string>();

  for (const value of markets) {
    if (!isRecord(value) || value.sportsMarketType !== "moneyline" || value.active !== true || value.closed === true) continue;
    if (!Array.isArray(value.events) || value.events.length !== 1 || !isRecord(value.events[0])) continue;
    const event = value.events[0];
    const date = event.eventDate;
    const title = event.title;
    const slug = event.slug;
    const selection = isRecord(value.marketMetadata)
      ? value.marketMetadata.opticOddsSelection
      : value.groupItemTitle;
    const updatedAt = value.updatedAt;
    if (![date, title, slug, selection, updatedAt].every((item) => typeof item === "string" && item.length > 0)) continue;
    const teams = (title as string).split(/\s+vs\.?\s+/i);
    if (teams.length !== 2) continue;
    const key = teamPairKey(teams[0], teams[1]);
    const price = yesPrice(value);
    if (price === null) {
      invalid.add(key);
      continue;
    }
    const canonicalSelection = canonicalTeam(selection as string);
    const outcome = (selection as string).toLocaleLowerCase() === "draw" ? "draw" : canonicalSelection;
    const validOutcomes = new Set([canonicalTeam(teams[0]), canonicalTeam(teams[1]), "draw"]);
    if (!validOutcomes.has(outcome)) {
      invalid.add(key);
      continue;
    }
    const eventUrl = `https://polymarket.com/event/${slug}`;
    groups[key] ??= { eventUrl, updatedAt: updatedAt as string, outcomes: {} };
    if (groups[key].eventUrl !== eventUrl || outcome in groups[key].outcomes) {
      invalid.add(key);
      continue;
    }
    groups[key].outcomes[outcome] = price;
    if ((updatedAt as string) > groups[key].updatedAt) groups[key].updatedAt = updatedAt as string;
  }

  return Object.fromEntries(
    Object.entries(groups).filter(([key, group]) => {
      if (invalid.has(key)) return false;
      const [team1, team2] = key.split("|");
      return Object.keys(group.outcomes).length === 3 && team1 in group.outcomes && team2 in group.outcomes && "draw" in group.outcomes;
    }),
  );
}

export function predictionsForMatches(
  matches: MatchPrediction[],
  markets: unknown[],
  now = new Date(),
): Record<string, PolymarketPrediction> {
  const groups = parsePolymarketMarkets(markets);
  return Object.fromEntries(matches.flatMap((match) => {
    if (isMatchCompleted(match, now)) return [];
    const group = groups[teamPairKey(match.homeTeam, match.awayTeam)];
    if (!group) return [];
    return [[match.id, {
      homeWin: group.outcomes[canonicalTeam(match.homeTeam)],
      draw: group.outcomes.draw,
      awayWin: group.outcomes[canonicalTeam(match.awayTeam)],
      eventUrl: group.eventUrl,
      updatedAt: group.updatedAt,
    } satisfies PolymarketPrediction]];
  }));
}

function fallbackPredictions(matches: MatchPrediction[], now = new Date()): Record<string, PolymarketPrediction> {
  return Object.fromEntries(matches.flatMap((match) =>
    match.polymarket && !isMatchCompleted(match, now)
      ? [[match.id, match.polymarket]]
      : [],
  ));
}

async function fetchAllMarkets(source: PolymarketSource): Promise<unknown[]> {
  const markets: unknown[] = [];
  let cursor: string | undefined;
  do {
    const url = new URL(source.dataUrl);
    url.searchParams.set("limit", "100");
    url.searchParams.set("tag_id", source.tagId);
    url.searchParams.set("series_id", source.seriesId);
    url.searchParams.set("closed", "false");
    url.searchParams.set("decimalized", "true");
    url.searchParams.set("refresh", String(Math.floor(Date.now() / FIVE_MINUTES_MS)));
    if (cursor) url.searchParams.set("after_cursor", cursor);
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`Polymarket request failed with ${response.status}`);
    const page: unknown = await response.json();
    if (!isRecord(page) || !Array.isArray(page.events)) throw new Error("Polymarket response is invalid");
    for (const event of page.events) {
      if (!isRecord(event) || !Array.isArray(event.markets)) continue;
      for (const market of event.markets) {
        if (isRecord(market)) markets.push({ ...market, events: [event] });
      }
    }
    if (page.next_cursor !== undefined && typeof page.next_cursor !== "string") throw new Error("Polymarket cursor is invalid");
    cursor = page.next_cursor as string | undefined;
  } while (cursor);
  return markets;
}

function getMarkets(source: PolymarketSource): Promise<unknown[]> {
  marketRequest ??= fetchAllMarkets(source);
  return marketRequest;
}

export function resetPolymarketCacheForTests(): void {
  marketRequest = null;
}

export function usePolymarket(
  matches: MatchPrediction[],
  source: PolymarketSource | undefined,
): PolymarketState {
  const fallback = fallbackPredictions(matches);
  const [state, setState] = useState<PolymarketState>({
    predictions: fallback,
    status: source ? "loading" : "fallback",
    lastCheckedAt: null,
  });

  useEffect(() => {
    let active = true;
    if (!source) {
      setState({ predictions: fallback, status: "fallback", lastCheckedAt: null });
      return () => { active = false; };
    }
    getMarkets(source)
      .then((markets) => {
        if (!active) return;
        setState({
          predictions: { ...fallback, ...predictionsForMatches(matches, markets) },
          status: "current",
          lastCheckedAt: new Date().toISOString(),
        });
      })
      .catch(() => {
        if (!active) return;
        setState({ predictions: fallback, status: "fallback", lastCheckedAt: new Date().toISOString() });
      });
    return () => { active = false; };
  }, [matches, source?.dataUrl, source?.marketType, source?.seriesId, source?.tagId]);

  return state;
}
