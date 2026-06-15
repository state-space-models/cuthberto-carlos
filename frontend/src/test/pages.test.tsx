import { render, screen } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { getPagesBase, getRepositoryName } from "../../pages";
import App from "../App";
import tournamentData from "../data/tournament.json";

const repositorySlug = process.env.GITHUB_REPOSITORY ?? "state-space-models/cuthberto-carlos";
const repositoryUrl = `https://github.com/${repositorySlug}`;

describe("GitHub Pages configuration", () => {
  it("deploys on prediction dispatches without an hourly schedule", () => {
    const pagesWorkflow = readFileSync(resolve("../.github/workflows/pages.yml"), "utf8");
    const predictionsWorkflow = readFileSync(
      resolve("../.github/workflows/daily-predictions.yml"),
      "utf8",
    );

    expect(pagesWorkflow).not.toContain("schedule:");
    expect(pagesWorkflow).toContain("types: [predictions-updated]");
    expect(predictionsWorkflow).toContain("event_type=predictions-updated");
    expect(predictionsWorkflow.indexOf("git diff --cached --quiet")).toBeLessThan(
      predictionsWorkflow.indexOf("event_type=predictions-updated"),
    );
  });

  it("derives the project base path from the repository slug", () => {
    expect(getRepositoryName("state-space-models/cuthberto-carlos")).toBe("cuthberto-carlos");
    expect(getPagesBase("state-space-models/cuthberto-carlos")).toBe("/cuthberto-carlos/");
    expect(getPagesBase("example-org/example-site")).toBe("/example-site/");
    expect(() => getPagesBase("invalid-repository")).toThrow("Invalid GitHub repository slug");
  });

  it("uses the active repository for generated and rendered links", () => {
    expect(tournamentData.repositoryUrl).toBe(repositoryUrl);
    expect(tournamentData.snapshotUrl).toMatch(new RegExp(`^${repositoryUrl}/tree/main/`));
    expect(
      tournamentData.groupMatches.every((match) =>
        match.sourceUrl.startsWith(`${repositoryUrl}/tree/main/`),
      ),
    ).toBe(true);

    render(<App />);

    expect(screen.getByRole("link", { name: /View model/i })).toHaveAttribute(
      "href",
      repositoryUrl,
    );
    expect(screen.getByRole("link", { name: /Latest model snapshot/i })).toHaveAttribute(
      "href",
      expect.stringContaining(`${repositoryUrl}/tree/main/`),
    );
    expect(screen.getByLabelText("Browse previous predictions")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /2026-06-11/i })).toHaveAttribute(
      "href",
      `${repositoryUrl}/tree/main/outputs/predictions/2026-06-11`,
    );
    expect(screen.queryByRole("link", { name: "Finals" })).not.toBeInTheDocument();
    expect(screen.queryByText(/complete path to the final/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/finals fixtures/i)).not.toBeInTheDocument();
  });
});
