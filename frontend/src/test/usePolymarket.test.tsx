import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { MatchPrediction } from "../types";
import {
  parsePolymarketMarkets,
  predictionsForMatches,
  usePolymarket,
} from "../usePolymarket";

function market(selection: string, price: number, title = "USA vs. Paraguay") {
  return {
    active: true,
    closed: false,
    sportsMarketType: "moneyline",
    outcomes: '["Yes", "No"]',
    outcomePrices: JSON.stringify([String(price), String(1 - price)]),
    updatedAt: "2026-06-18T12:00:00Z",
    marketMetadata: { opticOddsSelection: selection },
    events: [{ eventDate: "2026-06-19", title, slug: "fifwc-usa-par-2026-06-19" }],
  };
}

const markets = [market("Paraguay", 0.2), market("Draw", 0.3), market("USA", 0.5)];
const match = {
  id: "usa-paraguay",
  date: "2026-06-19",
  kickoffUtc: "2026-06-19T20:00:00Z",
  homeTeam: "United States",
  awayTeam: "Paraguay",
} as MatchPrediction;

describe("Polymarket data", () => {
  it("matches aliases and reversed market order", () => {
    const prediction = predictionsForMatches(
      [match],
      markets,
      new Date("2026-06-19T10:00:00Z"),
    )[match.id];

    expect(prediction).toMatchObject({ homeWin: 0.5, draw: 0.3, awayWin: 0.2 });
  });

  it("matches Polymarket team names and tolerates its event-date discrepancies", () => {
    const providerMarkets = [
      market("Türkiye", 0.45, "Türkiye vs. Paraguay"),
      market("Draw", 0.3, "Türkiye vs. Paraguay"),
      market("Paraguay", 0.25, "Türkiye vs. Paraguay"),
    ].map((item) => ({
      ...item,
      events: [{ ...item.events[0], eventDate: "2026-06-20" }],
    }));
    const projectMatch = {
      ...match,
      id: "turkey-paraguay",
      homeTeam: "Turkey",
      date: "2026-06-19",
    };
    expect(
      predictionsForMatches(
        [projectMatch],
        providerMarkets,
        new Date("2026-06-19T10:00:00Z"),
      )[projectMatch.id],
    ).toMatchObject({ homeWin: 0.45, draw: 0.3, awayWin: 0.25 });
  });

  it("drops malformed, duplicate, and incomplete fixture groups", () => {
    expect(Object.keys(parsePolymarketMarkets(markets.slice(0, 2)))).toHaveLength(0);
    expect(Object.keys(parsePolymarketMarkets([...markets, market("USA", 0.4)]))).toHaveLength(0);
    expect(Object.keys(parsePolymarketMarkets([
      ...markets.slice(0, 2),
      { ...market("USA", 0.5), outcomePrices: "invalid" },
    ]))).toHaveLength(0);
  });

  it("falls back to the group item title when market metadata is missing", () => {
    const providerMarkets = [
      market("Qatar", 0.15, "Bosnia-Herzegovina vs. Qatar"),
      market("Draw", 0.2, "Bosnia-Herzegovina vs. Qatar"),
      {
        ...market("Bosnia-Herzegovina", 0.65, "Bosnia-Herzegovina vs. Qatar"),
        marketMetadata: null,
        groupItemTitle: "Bosnia-Herzegovina",
      },
    ];
    const projectMatch = {
      ...match,
      id: "bosnia-qatar",
      homeTeam: "Bosnia and Herzegovina",
      awayTeam: "Qatar",
    };

    expect(
      predictionsForMatches(
        [projectMatch],
        providerMarkets,
        new Date("2026-06-19T10:00:00Z"),
      )[projectMatch.id],
    ).toMatchObject({ homeWin: 0.65, draw: 0.2, awayWin: 0.15 });
  });

  it("keeps prices during the live window and removes them after completion", () => {
    expect(predictionsForMatches([match], markets, new Date(match.kickoffUtc))[match.id]).toMatchObject({
      homeWin: 0.5,
      draw: 0.3,
      awayWin: 0.2,
    });
    expect(predictionsForMatches([match], markets, new Date("2026-06-19T22:00:00Z"))).toEqual({});
  });

  it("shows no stale fallback while loading, then displays all fetched pages", async () => {
    const futureMatch = {
      ...match,
      kickoffUtc: "2099-06-21T20:00:00Z",
      polymarket: {
        homeWin: 0.4,
        draw: 0.4,
        awayWin: 0.2,
        eventUrl: "https://polymarket.com/event/fallback",
        updatedAt: "2026-06-18T00:00:00Z",
      },
    };
    const futureMatches = [futureMatch];
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          events: [{ ...markets[0].events[0], markets: [markets[0]] }],
          next_cursor: "next",
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          events: [{ ...markets[1].events[0], markets: markets.slice(1) }],
        }),
      });
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => usePolymarket(futureMatches, {
      dataUrl: "https://gamma-api.polymarket.com/events/keyset",
      tagId: "102232",
      seriesId: "11433",
      marketType: "moneyline",
    }));

    expect(result.current.status).toBe("loading");
    expect(result.current.predictions[match.id]).toBeUndefined();
    await waitFor(() => expect(result.current.status).toBe("current"));
    expect(result.current.predictions[match.id].homeWin).toBe(0.5);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(String(fetchMock.mock.calls[1][0])).toContain("after_cursor=next");
  });

  it("starts a fresh no-store request for each frontend mount", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        events: [{ ...markets[0].events[0], markets }],
      }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const futureMatch = { ...match, kickoffUtc: "2099-06-21T20:00:00Z" };
    const futureMatches = [futureMatch];
    const source = {
      dataUrl: "https://gamma-api.polymarket.com/events/keyset",
      tagId: "102232",
      seriesId: "11433",
      marketType: "moneyline",
    };

    const first = renderHook(() => usePolymarket(futureMatches, source));
    await waitFor(() => expect(first.result.current.status).toBe("current"));
    first.unmount();
    const second = renderHook(() => usePolymarket(futureMatches, source));
    await waitFor(() => expect(second.result.current.status).toBe("current"));
    second.unmount();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls.every(([, options]) => options?.cache === "no-store")).toBe(true);
    const requestUrls = fetchMock.mock.calls.map(([url]) => String(url));
    expect(requestUrls[0]).not.toBe(requestUrls[1]);
  });

  it("keeps deployment fallback when the live request fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    const fallback = {
      homeWin: 0.4,
      draw: 0.4,
      awayWin: 0.2,
      eventUrl: "https://polymarket.com/event/fallback",
      updatedAt: "2026-06-18T00:00:00Z",
    };
    const fallbackMatches = [{
      ...match,
      kickoffUtc: "2099-06-21T20:00:00Z",
      polymarket: fallback,
    }];
    const { result } = renderHook(() => usePolymarket(fallbackMatches, {
      dataUrl: "https://gamma-api.polymarket.com/events/keyset",
      tagId: "102232",
      seriesId: "11433",
      marketType: "moneyline",
    }));

    await waitFor(() => expect(result.current.status).toBe("fallback"));
    expect(result.current.predictions[match.id]).toEqual(fallback);
  });
});
