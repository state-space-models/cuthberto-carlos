import { describe, expect, it } from "vitest";
import type { MatchPrediction } from "../types";
import {
  aggregateScoreGrid,
  describeBracketSlot,
  getActualGroupStats,
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
    predictionDate: "2026-06-11",
    sourceUrl: "https://example.test/source",
    predictionHistory: [],
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
      getCompletedMatches(matches, new Date("2026-06-11T18:00:00Z")).map((item) => item.id),
    ).toEqual(["newer", "older"]);
  });
});

describe("getActualGroupStats", () => {
  it("counts home and away wins, draws, losses, and games", () => {
    const homeWin = match("home-win", "2026-06-11T12:00:00Z");
    homeWin.actualResult = { homeScore: 2, awayScore: 0, homeGoals: [], awayGoals: [] };

    const awayWin = match("away-win", "2026-06-12T12:00:00Z");
    awayWin.homeTeam = "South Africa";
    awayWin.awayTeam = "Mexico";
    awayWin.actualResult = { homeScore: 1, awayScore: 3, homeGoals: [], awayGoals: [] };

    const draw = match("draw", "2026-06-13T12:00:00Z");
    draw.homeTeam = "Mexico";
    draw.awayTeam = "Czech Republic";
    draw.actualResult = { homeScore: 1, awayScore: 1, homeGoals: [], awayGoals: [] };

    const future = match("future", "2026-06-14T12:00:00Z");
    future.homeTeam = "South Korea";
    future.awayTeam = "Czech Republic";

    expect(getActualGroupStats([homeWin, awayWin, draw, future])).toEqual({
      Mexico: {
        games: 3, wins: 2, draws: 1, losses: 0,
        points: 7, goalDifference: 4, goalsScored: 6,
      },
      "South Africa": {
        games: 2, wins: 0, draws: 0, losses: 2,
        points: 0, goalDifference: -4, goalsScored: 1,
      },
      "Czech Republic": {
        games: 1, wins: 0, draws: 1, losses: 0,
        points: 1, goalDifference: 0, goalsScored: 1,
      },
      "South Korea": {
        games: 0, wins: 0, draws: 0, losses: 0,
        points: 0, goalDifference: 0, goalsScored: 0,
      },
    });
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
