import { describe, expect, it } from "vitest";
import type { MatchPrediction } from "../types";
import {
  aggregateScoreGrid,
  describeBracketSlot,
  getCompletedMatches,
  getUpcomingMatches,
} from "../utils";

function match(id: string, kickoffUtc: string): MatchPrediction {
  return {
    id,
    date: kickoffUtc.slice(0, 10),
    kickoffUtc,
    group: "A",
    venue: "Test venue",
    homeTeam: "Mexico",
    awayTeam: "South Africa",
    sourceUrl: "https://example.test/source",
    prediction: {
      probabilities: { homeWin: 0.5, draw: 0.3, awayWin: 0.2 },
      scoreGrid: [[1]],
      mostLikelyScore: [0, 0],
      mostLikelyScoreProbability: 1,
      expectedGoals: { home: 0, away: 0 },
      skills: {
        home: { attack: { mean: 0, sd: 0.1 }, defence: { mean: 0, sd: 0.1 } },
        away: { attack: { mean: 0, sd: 0.1 }, defence: { mean: 0, sd: 0.1 } },
      },
    },
  };
}

describe("getUpcomingMatches", () => {
  it("filters past matches, sorts kickoff times, and applies a limit", () => {
    const matches = [
      match("later", "2026-06-12T19:00:00Z"),
      match("past", "2026-06-10T19:00:00Z"),
      match("next", "2026-06-11T20:00:00Z"),
    ];

    expect(getUpcomingMatches(matches, new Date("2026-06-11T18:00:00Z"), 1).map((item) => item.id)).toEqual(["next"]);
  });
});

describe("getCompletedMatches", () => {
  it("waits two hours after kickoff and sorts newest fixtures first", () => {
    const matches = [
      match("older", "2026-06-11T12:00:00Z"),
      match("newer", "2026-06-11T14:00:00Z"),
      match("in-progress", "2026-06-11T16:30:00Z"),
    ];

    expect(
      getCompletedMatches(matches, new Date("2026-06-11T18:00:00Z")).map(
        (item) => item.id,
      ),
    ).toEqual(["newer", "older"]);
  });
});

describe("aggregateScoreGrid", () => {
  it("keeps 0-4 scores and combines all larger scores into 5+", () => {
    const grid = Array.from({ length: 9 }, () => Array(9).fill(0));
    grid[0][0] = 0.25;
    grid[5][7] = 0.3;
    grid[8][8] = 0.45;

    const aggregate = aggregateScoreGrid(grid);
    expect(aggregate[0][0]).toBe(0.25);
    expect(aggregate[5][5]).toBe(0.75);
  });
});

describe("describeBracketSlot", () => {
  it("describes fixed, third-place, winner, and loser slots", () => {
    expect(describeBracketSlot("1A")).toBe("Winner Group A");
    expect(describeBracketSlot("3A/B/C/D/F")).toContain("A, B, C, D, F");
    expect(describeBracketSlot("W73")).toBe("Winner Match 73");
    expect(describeBracketSlot("L101")).toBe("Loser Match 101");
  });
});
