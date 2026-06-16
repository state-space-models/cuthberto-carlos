import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import App from "../App";
import { CompletedMatches } from "../components/CompletedMatches";
import { Countries } from "../components/Countries";
import { GroupStage } from "../components/GroupStage";
import { KnockoutBracket } from "../components/KnockoutBracket";
import tournamentData from "../data/tournament.json";
import type { MatchPrediction, PredictionHistoryEntry, TournamentDataset } from "../types";

const data = tournamentData as unknown as TournamentDataset;
const repositoryUrl = `https://github.com/${process.env.GITHUB_REPOSITORY ?? "state-space-models/cuthberto-carlos"}`;
type PredictionVersion = PredictionHistoryEntry;

function predictionVersions(match: MatchPrediction): PredictionVersion[] {
  return [
    ...match.predictionHistory,
    {
      predictionDate: match.predictionDate,
      sourceUrl: match.sourceUrl,
      prediction: match.prediction,
    },
  ].sort((left, right) => left.predictionDate.localeCompare(right.predictionDate));
}

function latestPredictionVersion(match: MatchPrediction): PredictionVersion {
  return predictionVersions(match).at(-1)!;
}

function previousPredictionVersions(match: MatchPrediction): PredictionVersion[] {
  const latest = latestPredictionVersion(match);
  return predictionVersions(match)
    .filter((version) => version.predictionDate < latest.predictionDate)
    .sort((left, right) => right.predictionDate.localeCompare(left.predictionDate));
}

function formatPercentForTest(value: number): string {
  return new Intl.NumberFormat(undefined, {
    style: "percent",
    maximumFractionDigits: 1,
    minimumFractionDigits: 1,
  }).format(value);
}

describe("generated tournament data", () => {
  it("contains the complete initial tournament shape", () => {
    expect(tournamentData.schemaVersion).toBe(5);
    expect(tournamentData.repositoryUrl).toBe(repositoryUrl);
    expect(tournamentData.snapshotDate).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(data.groupMatches).toHaveLength(72);
    expect(data.groups).toHaveLength(12);
    expect(data.groups.every((group) => group.matchIds.length === 6)).toBe(true);
    expect(data.knockoutMatches).toHaveLength(32);
    expect(tournamentData.snapshotUrl).toContain(`/outputs/predictions/${tournamentData.snapshotDate}`);
    expect(tournamentData.sources.schedule.url).toBe(
      "https://github.com/openfootball/worldcup.json/blob/master/2026/worldcup.json",
    );
    expect(tournamentData.sources.squads.url).toBe(
      "https://github.com/openfootball/worldcup.json/blob/master/2026/worldcup.squads.json",
    );
    expect(Object.keys(tournamentData.teams)).toHaveLength(48);
    expect(tournamentData.teams.Mexico.players).toHaveLength(26);
    const mexicoSouthKorea = data.groupMatches.find(
      (match) => match.homeTeam === "Mexico" && match.awayTeam === "South Korea",
    );
    expect(mexicoSouthKorea).toBeDefined();
    expect(mexicoSouthKorea!.predictionDate).toBe(latestPredictionVersion(mexicoSouthKorea!).predictionDate);
    expect(mexicoSouthKorea!.predictionHistory.length).toBeGreaterThan(0);
    expect(
      mexicoSouthKorea!.predictionHistory.every(
        (prediction) => prediction.predictionDate < mexicoSouthKorea!.predictionDate,
      ),
    ).toBe(true);
    expect(mexicoSouthKorea!.predictionHistory).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          prediction: expect.objectContaining({
            probabilities: expect.objectContaining({ homeWin: expect.any(Number) }),
            skills: expect.any(Object),
          }),
        }),
      ]),
    );

    const mexicoSouthAfrica = data.groupMatches.find(
      (match) => match.homeTeam === "Mexico" && match.awayTeam === "South Africa",
    );
    expect(mexicoSouthAfrica?.predictionDate).toBe("2026-06-11");
    expect(mexicoSouthAfrica?.predictionHistory).toEqual([]);
  });
});

describe("App interactions", () => {
  it("shows every upcoming match in scrollable card and list views", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-11T12:00:00Z"));

    try {
      render(<App />);

      const expectedMatches = data.groupMatches.filter(
        (match) => new Date(match.kickoffUtc).getTime() > Date.now(),
      );
      const viewToggle = screen.getByRole("group", { name: "Upcoming matches view" });
      const cardView = screen.getByRole("region", { name: "All upcoming matches in card view" });
      expect(cardView).toHaveClass("upcoming-matches-scroll");
      expect(cardView).toHaveAttribute("tabindex", "0");
      expect(within(cardView).getAllByRole("article")).toHaveLength(expectedMatches.length);
      const firstCard = within(cardView).getAllByRole("article")[0];
      expect(within(firstCard).getByText(new RegExp(`^${expectedMatches[0].homeTeam} \\d+%$`))).toBeInTheDocument();
      expect(within(firstCard).getByText(/^Draw \d+%$/)).toBeInTheDocument();
      expect(within(firstCard).getByText(new RegExp(`^${expectedMatches[0].awayTeam} \\d+%$`))).toBeInTheDocument();
      expect(within(firstCard).queryByText(/^H \d+%$/)).not.toBeInTheDocument();
      expect(within(firstCard).queryByText(/^A \d+%$/)).not.toBeInTheDocument();
      expect(within(viewToggle).getByRole("button", { name: "Cards" })).toHaveAttribute("aria-pressed", "true");

      fireEvent.click(within(viewToggle).getByRole("button", { name: "List" }));
      const listView = screen.getByRole("region", { name: "All upcoming matches in list view" });
      expect(listView).toHaveClass("upcoming-matches-scroll");
      expect(within(listView).getAllByRole("article")).toHaveLength(expectedMatches.length);
      const firstListRow = within(listView).getAllByRole("article")[0];
      expect(within(firstListRow).getByText(new RegExp(`^${expectedMatches[0].homeTeam} \\d+%$`))).toBeInTheDocument();
      expect(within(firstListRow).getByText(/^Draw \d+%$/)).toBeInTheDocument();
      expect(within(firstListRow).getByText(new RegExp(`^${expectedMatches[0].awayTeam} \\d+%$`))).toBeInTheDocument();
      expect(within(viewToggle).getByRole("button", { name: "List" })).toHaveAttribute("aria-pressed", "true");

      fireEvent.click(within(listView).getByRole("button", { name: /Explore prediction for Mexico versus South Korea/i }));
      expect(screen.getByRole("dialog", { name: /Mexico vs South Korea/i })).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("archives matches two hours after kickoff", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-12T05:00:00Z"));
    const completedMatch = structuredClone(data.groupMatches[0]);
    completedMatch.actualResult = {
      homeScore: 2,
      awayScore: 1,
      homeGoals: [
        { name: "Julián Quiñones", minute: "45+2", penalty: true },
        { name: "Unknown Scorer", minute: "90+4" },
      ],
      awayGoals: [{ name: "Lyle Foster", minute: "52" }],
    };
    const awaitingResult = structuredClone(data.groupMatches[1]);
    delete awaitingResult.actualResult;

    try {
      render(
        <CompletedMatches
          matches={[completedMatch, awaitingResult]}
          teams={data.teams}
          onOpen={vi.fn()}
        />,
      );

      expect(screen.getByText(/two hours after scheduled kickoff/i)).toBeInTheDocument();
      const completed = screen.getByTestId("completed-list-view");
      expect(within(completed).getAllByText("Actual score")).toHaveLength(2);
      expect(within(completed).getAllByText("Predicted score")).toHaveLength(2);
      expect(within(completed).getByText("2–1")).toBeInTheDocument();
      expect(within(completed).getByText("45+2' (pen.)")).toBeInTheDocument();
      expect(within(completed).getByRole("button", { name: "Julián Quiñones" })).toBeInTheDocument();
      expect(within(completed).getByRole("button", { name: "Lyle Foster" })).toBeInTheDocument();
      const mexicoScorers = within(completed).getByRole("list", { name: "Mexico goal scorers" });
      const southAfricaScorers = within(completed).getByRole("list", { name: "South Africa goal scorers" });
      expect(within(mexicoScorers).getByText("Julián Quiñones")).toBeInTheDocument();
      expect(within(mexicoScorers).queryByText("Lyle Foster")).not.toBeInTheDocument();
      expect(within(southAfricaScorers).getByText("Lyle Foster")).toBeInTheDocument();
      expect(within(southAfricaScorers).queryByText("Julián Quiñones")).not.toBeInTheDocument();
      const mexicoMatch = within(completed).getByRole("button", { name: "Julián Quiñones" }).closest("article");
      const comparisonRows = mexicoMatch?.querySelector(".score-comparison--with-scorers")?.children;
      expect(comparisonRows?.[0]).toHaveClass("score-comparison__prediction");
      expect(comparisonRows?.[1]).toHaveClass("score-comparison__actual-row");
      expect(comparisonRows?.[1].children[0]).toBe(mexicoScorers);
      expect(comparisonRows?.[1].children[2]).toBe(southAfricaScorers);
      expect(within(completed).getByText("Unknown Scorer").tagName).toBe("SPAN");
      expect(completed.textContent).toContain("⚽");
      expect(within(completed).getByText("Currently Not Available")).toBeInTheDocument();
      expect(within(completed).getByText("South Korea")).toBeInTheDocument();
      expect(within(completed).queryByLabelText("Result probabilities")).not.toBeInTheDocument();
      expect(within(completed).queryByText("Mexico win")).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("updates an open match drawer when live results arrive", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-12T05:00:00Z"));
    let resolveFetch!: (response: unknown) => void;
    vi.stubGlobal("fetch", vi.fn().mockReturnValue(new Promise((resolve) => {
      resolveFetch = resolve;
    })));

    try {
      render(<App />);
      const completed = screen.getByTestId("completed-list-view");
      fireEvent.click(
        within(completed).getByRole("button", {
          name: /Explore prediction for Mexico versus South Africa/i,
        }),
      );
      const drawer = screen.getByRole("dialog", { name: /Mexico vs South Africa/i });
      expect(within(drawer).getByText("2–0")).toBeInTheDocument();

      await act(async () => {
        resolveFetch({
          ok: true,
          json: async () => ({
            matches: [{
              date: "2026-06-11",
              team1: "Mexico",
              team2: "South Africa",
              score: { ft: [9, 8] },
              goals1: [],
              goals2: [],
            }],
          }),
        });
        await Promise.resolve();
      });

      expect(within(drawer).getByText("9–8")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows actual group stats and compares completed fixture scores", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-12T05:00:00Z"));
    const group = data.groups[0];
    const groupMatches = data.groupMatches
      .filter((match) => match.group === group.id)
      .map((match) => {
        const copy = structuredClone(match);
        delete copy.actualResult;
        return copy;
      });
    groupMatches[0].actualResult = {
      homeScore: 2,
      awayScore: 0,
      homeGoals: [{ name: "Julián Quiñones", minute: "9" }],
      awayGoals: [],
    };

    try {
      render(
        <GroupStage
          groups={[group]}
          matches={groupMatches}
          teams={data.teams}
          onOpen={vi.fn()}
        />,
      );

    expect(screen.getByTitle("Games played")).toHaveTextContent("G");
    expect(screen.getByTitle("Wins")).toHaveTextContent("W");
    expect(screen.getByTitle("Draws")).toHaveTextContent("D");
    expect(screen.getByTitle("Losses")).toHaveTextContent("L");
    expect(screen.getByTitle("Points")).toHaveTextContent("PTS");
    expect(screen.getByRole("columnheader", { name: "Seed" })).toBeInTheDocument();
    expect(screen.getByTitle("Goal difference")).toHaveTextContent("GD");
    expect(screen.getByTitle("Goals scored")).toHaveTextContent("GS");
    expect(screen.getByText(/xPTS means expected points/i)).toBeInTheDocument();

    const mexicoRow = screen.getByRole("row", { name: /Mexico/ });
    const mexicoCells = within(mexicoRow).getAllByRole("cell");
    expect(mexicoCells.slice(1, 5).map((cell) => cell.textContent)).toEqual(["1", "1", "0", "0"]);
    expect(mexicoCells.slice(5, 8).map((cell) => cell.textContent)).toEqual(["3", "+2", "2"]);
    expect(screen.getAllByRole("row")[1]).toHaveAccessibleName(/Mexico/);

      expect(screen.getAllByText("Played")).toHaveLength(2);
      expect(screen.getAllByText("Actual score")).toHaveLength(6);
      expect(screen.getAllByText("Predicted score")).toHaveLength(6);
      expect(screen.getByText("2–0")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Julián Quiñones" })).toBeInTheDocument();
      expect(screen.getAllByText("Currently Not Available")).toHaveLength(5);
      expect(screen.queryByText(/Mexico win/i)).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("marks ongoing fixtures live under Upcoming matches", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-11T20:00:00Z"));

    try {
      render(<App />);
      const upcoming = screen.getByRole("region", { name: "All upcoming matches in card view" });
      expect(within(upcoming).getByText("LIVE")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("filters groups and opens an accessible prediction drawer", async () => {
    const user = userEvent.setup();
    const canadaSwitzerland = data.groupMatches.find(
      (match) => match.homeTeam === "Canada" && match.awayTeam === "Switzerland",
    );
    expect(canadaSwitzerland).toBeDefined();
    const latestVersion = latestPredictionVersion(canadaSwitzerland!);
    const previousVersions = previousPredictionVersions(canadaSwitzerland!);
    const historicalVersion =
      previousVersions.find((version) => version.predictionDate === "2026-06-11") ??
      previousVersions.at(-1);
    expect(historicalVersion).toBeDefined();

    render(<App />);

    const groupFilter = screen.getByRole("navigation", { name: "Filter group stage" });
    await user.click(within(groupFilter).getByRole("button", { name: /^B$/ }));
    expect(screen.getByRole("heading", { name: "Group B" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Group A" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Open prediction for Canada versus Switzerland/i }));
    expect(screen.getByRole("dialog", { name: /Canada vs Switzerland/i })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /Score probability/i })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Switzerland ↓ / Canada →" })).toBeInTheDocument();
    const dateSelector = screen.getByRole("group", { name: "Prediction date" });
    const predictionDateSelect = within(dateSelector).getByRole("combobox", { name: "View prediction" });
    expect(predictionDateSelect).toHaveValue(latestVersion.predictionDate);
    expect(
      within(dateSelector).getByRole("option", {
        name: new RegExp(`${latestVersion.predictionDate} \\(latest\\)`, "i"),
      }),
    ).toBeInTheDocument();
    expect(screen.getByText(formatPercentForTest(latestVersion.prediction.probabilities.homeWin))).toBeInTheDocument();
    const probabilityBars = document.querySelectorAll<HTMLElement>(".probability-row__track > span");
    expect(probabilityBars[0]).toHaveStyle({ background: "#15803d" });
    expect(probabilityBars[1]).toHaveStyle({ background: "#777b76" });
    expect(probabilityBars[2]).toHaveStyle({ background: "#dc2626" });
    expect(
      screen.getByRole("img", { name: "Team strength history for Canada and Switzerland" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Team strength chart legend")).toHaveTextContent(
      "CanadaSwitzerlandAttackDefence",
    );
    expect(
      screen.getByRole("link", {
        name: new RegExp(`Prediction generated ${latestVersion.predictionDate}`, "i"),
      }),
    ).toHaveAttribute("href", latestVersion.sourceUrl);
    expect(screen.getAllByText("Previous predictions")).toHaveLength(2);
    const historicalMatchLink = screen
      .getAllByRole("link", { name: new RegExp(`^${historicalVersion!.predictionDate}`) })
      .find((link) => link.getAttribute("href") === historicalVersion!.sourceUrl);
    expect(historicalMatchLink).toHaveAttribute("href", historicalVersion!.sourceUrl);

    await user.selectOptions(predictionDateSelect, historicalVersion!.predictionDate);
    expect(predictionDateSelect).toHaveValue(historicalVersion!.predictionDate);
    expect(
      screen.getByText(formatPercentForTest(historicalVersion!.prediction.probabilities.homeWin)),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", {
        name: new RegExp(`Prediction generated ${historicalVersion!.predictionDate}`, "i"),
      }),
    ).toHaveAttribute("href", historicalVersion!.sourceUrl);

    await user.click(screen.getByRole("button", { name: "Close prediction details" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("shows team results available before the selected prediction date", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Open prediction for Japan versus Sweden/i }));
    const drawer = screen.getByRole("dialog", { name: /Japan vs Sweden/i });
    const dateSelector = within(drawer).getByRole("combobox", { name: "View prediction" });
    const latestPredictionDate = within(dateSelector).getByRole("option", { name: /\(latest\)/i });
    const selectedDate = (dateSelector as HTMLSelectElement).value;

    expect(dateSelector).toHaveValue(latestPredictionDate.getAttribute("value"));
    const recentMatches = within(drawer).getByRole("list", {
      name: new RegExp(`Matches involving Japan or Sweden before ${selectedDate}`),
    });
    expect(within(recentMatches).getByText("Netherlands")).toBeInTheDocument();
    expect(within(recentMatches).getByText("Japan")).toBeInTheDocument();
    expect(within(recentMatches).getByText("Sweden")).toBeInTheDocument();
    expect(within(recentMatches).getByText("Tunisia")).toBeInTheDocument();
    expect(within(recentMatches).getByText("2–2")).toBeInTheDocument();
    expect(within(recentMatches).getByText("5–1")).toBeInTheDocument();

    await user.selectOptions(dateSelector, "2026-06-11");
    expect(
      within(drawer).getByText("No completed matches involving either team before this prediction."),
    ).toBeInTheDocument();
    expect(within(drawer).queryByRole("list", { name: /Matches involving Japan or Sweden/ })).not.toBeInTheDocument();
  });

  it("filters countries, selects a squad, and exposes accessible player profiles", async () => {
    const user = userEvent.setup();
    render(<Countries teams={data.teams} />);

    const countryFilter = screen.getByRole("navigation", { name: "Filter countries by group" });
    await user.click(within(countryFilter).getByRole("button", { name: /^A$/ }));
    expect(screen.getAllByRole("button", { name: /players$/ })).toHaveLength(4);

    await user.click(screen.getByRole("button", { name: /Mexico MEX Group A · 26 players/ }));
    expect(screen.getByRole("heading", { name: "Mexico squad" })).toBeInTheDocument();
    const roster = screen.getByRole("table", { name: "Mexico player roster" });
    const playerRows = within(roster).getAllByRole("row").slice(1);
    expect(within(playerRows[0]).getAllByRole("cell")[0]).toHaveTextContent("1");
    expect(within(playerRows.at(-1)!).getAllByRole("cell")[0]).toHaveTextContent("26");
    expect(within(roster).getByText("25 Feb 2000")).toBeInTheDocument();

    const player = within(roster).getByRole("button", { name: "Raúl Rangel" });
    await user.hover(player);
    expect(screen.getByRole("tooltip")).toHaveTextContent("No. 1 · GK");
    await user.unhover(player);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

    fireEvent.focus(player);
    expect(screen.getByRole("tooltip")).toHaveTextContent("Born 25 Feb 2000");
    fireEvent.keyDown(player, { key: "Escape" });
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

    fireEvent.click(player);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    fireEvent.click(player);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("keeps official knockout feeder labels visible", () => {
    render(<KnockoutBracket matches={data.knockoutMatches} />);
    expect(screen.getAllByText("2A").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Runner-up Group A").length).toBeGreaterThan(0);
  });

  it("updates knockout details when a bracket match is selected", async () => {
    const user = userEvent.setup();
    render(<KnockoutBracket matches={data.knockoutMatches} />);

    await user.click(screen.getByRole("button", { name: "Final" }));
    expect(screen.getByLabelText("Details for Match 104")).toHaveTextContent("W101 vs W102");
  });
});
