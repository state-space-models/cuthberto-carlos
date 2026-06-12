import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import App from "../App";
import { CompletedMatches } from "../components/CompletedMatches";
import { Countries } from "../components/Countries";
import { GroupStage } from "../components/GroupStage";
import { KnockoutBracket } from "../components/KnockoutBracket";
import tournamentData from "../data/tournament.json";
import type { TournamentDataset } from "../types";

const data = tournamentData as unknown as TournamentDataset;
const repositoryUrl = `https://github.com/${process.env.GITHUB_REPOSITORY ?? "state-space-models/cuthberto-carlos"}`;

describe("generated tournament data", () => {
  it("contains the complete initial tournament shape", () => {
    expect(tournamentData.schemaVersion).toBe(3);
    expect(tournamentData.repositoryUrl).toBe(repositoryUrl);
    expect(tournamentData.snapshotDate).toBe("2026-06-11");
    expect(tournamentData.groupMatches).toHaveLength(72);
    expect(tournamentData.groups).toHaveLength(12);
    expect(tournamentData.groups.every((group) => group.matchIds.length === 6)).toBe(true);
    expect(tournamentData.knockoutMatches).toHaveLength(32);
    expect(tournamentData.snapshotUrl).toContain("/outputs/predictions/2026-06-11");
    expect(tournamentData.sources.schedule.url).toBe(
      "https://github.com/openfootball/worldcup.json/blob/master/2026/worldcup.json",
    );
    expect(tournamentData.sources.squads.url).toBe(
      "https://github.com/openfootball/worldcup.json/blob/master/2026/worldcup.squads.json",
    );
    expect(Object.keys(tournamentData.teams)).toHaveLength(48);
    expect(tournamentData.teams.Mexico.players).toHaveLength(26);
    expect(tournamentData.groupMatches.every((match) => match.sourceUrl.includes("/2026-06-11/"))).toBe(true);
  });
});

describe("App interactions", () => {
  it("toggles upcoming matches between card and list views", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-11T12:00:00Z"));

    try {
      render(<App />);

      const viewToggle = screen.getByRole("group", { name: "Upcoming matches view" });
      expect(screen.getByTestId("upcoming-card-view")).toBeInTheDocument();
      expect(within(viewToggle).getByRole("button", { name: "Cards" })).toHaveAttribute("aria-pressed", "true");

      fireEvent.click(within(viewToggle).getByRole("button", { name: "List" }));
      expect(screen.queryByTestId("upcoming-card-view")).not.toBeInTheDocument();
      expect(screen.getByTestId("upcoming-list-view")).toBeInTheDocument();
      expect(within(viewToggle).getByRole("button", { name: "List" })).toHaveAttribute("aria-pressed", "true");

      fireEvent.click(screen.getByRole("button", { name: /Explore prediction for Mexico versus South Africa/i }));
      expect(screen.getByRole("dialog", { name: /Mexico vs South Africa/i })).toBeInTheDocument();
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
    expect(screen.getByTitle("Goal difference")).toHaveTextContent("GD");
    expect(screen.getByTitle("Goals scored")).toHaveTextContent("GS");
    expect(screen.getByText(/xPTS means expected points/i)).toBeInTheDocument();

    const mexicoRow = screen.getByRole("row", { name: /Mexico/ });
    const mexicoCells = within(mexicoRow).getAllByRole("cell");
    expect(mexicoCells.slice(1, 5).map((cell) => cell.textContent)).toEqual(["1", "1", "0", "0"]);
    expect(mexicoCells.slice(5, 8).map((cell) => cell.textContent)).toEqual(["3", "+2", "2"]);

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
      const upcoming = screen.getByTestId("upcoming-card-view");
      expect(within(upcoming).getByText("LIVE")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("filters groups and opens an accessible prediction drawer", async () => {
    const user = userEvent.setup();
    render(<App />);

    const groupFilter = screen.getByRole("navigation", { name: "Filter group stage" });
    await user.click(within(groupFilter).getByRole("button", { name: /^B$/ }));
    expect(screen.getByRole("heading", { name: "Group B" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Group A" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Open prediction for Canada versus Bosnia and Herzegovina/i }));
    expect(screen.getByRole("dialog", { name: /Canada vs Bosnia and Herzegovina/i })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /Score probability/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Snapshot 2026-06-11 source data/i })).toHaveAttribute(
      "href",
      expect.stringContaining("/outputs/predictions/2026-06-11/"),
    );

    await user.click(screen.getByRole("button", { name: "Close prediction details" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
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
