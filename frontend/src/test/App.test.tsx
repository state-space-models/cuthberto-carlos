import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import App from "../App";
import tournamentData from "../data/tournament.json";

describe("generated tournament data", () => {
  it("contains the complete initial tournament shape", () => {
    expect(tournamentData.schemaVersion).toBe(2);
    expect(tournamentData.snapshotDate).toBe("2026-06-11");
    expect(tournamentData.groupMatches).toHaveLength(72);
    expect(tournamentData.groups).toHaveLength(12);
    expect(tournamentData.groups.every((group) => group.matchIds.length === 6)).toBe(true);
    expect(tournamentData.knockoutMatches).toHaveLength(32);
    expect(tournamentData.snapshotUrl).toContain("/outputs/predictions/2026-06-11");
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

  it("archives completed matches with actual and predicted scores", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-12T05:00:00Z"));

    try {
      render(<App />);

      const completed = screen.getByTestId("completed-list-view");
      expect(completed).toBeInTheDocument();
      expect(within(completed).getByText("Mexico win")).toBeInTheDocument();
      expect(within(completed).getAllByText("Actual score")).toHaveLength(2);
      expect(within(completed).getAllByText("Predicted score")).toHaveLength(2);
      expect(within(completed).getAllByText("1–0")).toHaveLength(2);
      expect(within(completed).getByText("Currently not available")).toBeInTheDocument();
      expect(
        within(completed).getByRole("link", { name: /View source data for Mexico versus South Africa/i }),
      ).toHaveAttribute("href", expect.stringContaining("/outputs/predictions/2026-06-11/"));
    } finally {
      vi.useRealTimers();
    }
  });

  it("filters groups and opens an accessible prediction drawer", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /^B$/ }));
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

  it("keeps official knockout feeder labels visible", () => {
    render(<App />);
    expect(screen.getAllByText("2A").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Runner-up Group A").length).toBeGreaterThan(0);
  });

  it("updates knockout details when a bracket match is selected", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /M104 W101 Winner Match 101 W102 Winner Match 102/ }));
    expect(screen.getByLabelText("Details for Match 104")).toHaveTextContent("W101 vs W102");
  });
});
