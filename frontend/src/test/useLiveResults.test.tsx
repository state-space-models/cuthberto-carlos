import { StrictMode } from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import tournamentData from "../data/tournament.json";
import type { TournamentDataset } from "../types";
import {
  mergeKnockoutSchedule,
  mergeLiveResults,
  parseOpenFootballSchedule,
  useLiveResults,
} from "../useLiveResults";

const data = tournamentData as unknown as TournamentDataset;

describe("OpenFootball result merging", () => {
  it("resolves knockout participants and preserves extra-time and penalty scores", () => {
    const schedule = parseOpenFootballSchedule({ matches: [{
      num: 104,
      round: "Final",
      date: "2026-07-19",
      time: "15:00 UTC-4",
      ground: "New final venue",
      team1: "USA",
      team2: "Argentina",
      score: { ft: [1, 1], et: [2, 2], p: [4, 3] },
    }] });
    const final = mergeKnockoutSchedule(data.knockoutMatches, schedule).at(-1)!;
    expect(final).toMatchObject({
      matchNumber: 104,
      team1Slot: "W101",
      team2Slot: "W102",
      team1: "United States",
      team2: "Argentina",
      venue: "New final venue",
      kickoffUtc: "2026-07-19T19:00:00.000Z",
      score: { fullTime: [1, 1], extraTime: [2, 2], penalties: [4, 3] },
    });
  });

  it("merges aliases, reversed fixtures, scores, and scorer details", () => {
    const match = structuredClone(
      data.groupMatches.find(
        (candidate) => candidate.homeTeam === "United States" && candidate.awayTeam === "Paraguay",
      )!,
    );
    delete match.actualResult;
    const schedule = parseOpenFootballSchedule({
      matches: [{
        date: match.date,
        team1: "Paraguay",
        team2: "USA",
        score: { ft: [1, 3] },
        goals1: [{ name: "Mauricio", minute: 73 }],
        goals2: [{ name: "Folarin Balogun", minute: "45+5", penalty: true }],
      }],
    });

    const [merged] = mergeLiveResults([match], schedule, data.teams);
    expect(merged.actualResult).toEqual({
      homeScore: 3,
      awayScore: 1,
      homeGoals: [{ name: "Folarin Balogun", minute: "45+5", penalty: true }],
      awayGoals: [{ name: "Maurício", minute: "73", penalty: null }],
    });
  });

  it("ignores incomplete and unknown fixtures", () => {
    const match = structuredClone(data.groupMatches[0]);
    const schedule = parseOpenFootballSchedule({
      matches: [
        { date: match.date, team1: match.homeTeam, team2: match.awayTeam },
        { date: "2026-06-01", team1: "Unknown", team2: "Missing", score: { ft: [1, 0] } },
      ],
    });
    expect(mergeLiveResults([match], schedule, data.teams)).toEqual([match]);
  });

  it("rejects malformed responses", () => {
    expect(() => parseOpenFootballSchedule({})).toThrow("matches array");
    expect(() => parseOpenFootballSchedule({ matches: [{ date: "2026-06-11" }] })).toThrow(
      "invalid fixture",
    );
  });
});

describe("useLiveResults", () => {
  it("fetches once under StrictMode and uses a five-minute cache bucket", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ matches: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(
      () => useLiveResults(data.groupMatches, data.sources.schedule.dataUrl, data.teams),
      { wrapper: StrictMode },
    );

    await waitFor(() => expect(result.current.status).toBe("current"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\?refresh=\d+$/),
      { cache: "no-store" },
    );
  });

  it("keeps deployment-time results when loading fails", async () => {
    const staticMatches = [structuredClone(data.groupMatches[0])];
    const { result } = renderHook(() =>
      useLiveResults(staticMatches, data.sources.schedule.dataUrl, data.teams),
    );

    await waitFor(() => expect(result.current.status).toBe("fallback"));
    expect(result.current.matches).toEqual(staticMatches);
    expect(result.current.lastCheckedAt).not.toBeNull();
  });
});
