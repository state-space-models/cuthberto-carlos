import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { getPagesBase, getRepositoryName } from "../../pages";
import App from "../App";
import tournamentData from "../data/tournament.json";

const repositorySlug = process.env.GITHUB_REPOSITORY ?? "state-space-models/cuthberto-carlos";
const repositoryUrl = `https://github.com/${repositorySlug}`;

describe("GitHub Pages configuration", () => {
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
    expect(screen.queryByRole("link", { name: "Finals" })).not.toBeInTheDocument();
    expect(screen.queryByText(/complete path to the final/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/finals fixtures/i)).not.toBeInTheDocument();
  });
});
