import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { getPagesBase, getRepositoryName } from "../../pages";
import App from "../App";
import tournamentData from "../data/tournament.json";

describe("GitHub Pages configuration", () => {
  it("derives the project base path from the repository slug", () => {
    expect(getRepositoryName("state-space-models/cuthberto-carlos")).toBe("cuthberto-carlos");
    expect(getPagesBase("state-space-models/cuthberto-carlos")).toBe("/cuthberto-carlos/");
    expect(getPagesBase("example-org/example-site")).toBe("/example-site/");
    expect(() => getPagesBase("invalid-repository")).toThrow("Invalid GitHub repository slug");
  });

  it("contains no legacy fork URLs and links to the organization repository", () => {
    expect(tournamentData.repositoryUrl).toBe("https://github.com/state-space-models/cuthberto-carlos");
    expect(tournamentData.snapshotUrl).toMatch(
      /^https:\/\/github\.com\/state-space-models\/cuthberto-carlos\/tree\/main\//,
    );
    expect(
      tournamentData.groupMatches.every((match) =>
        match.sourceUrl.startsWith(
          "https://github.com/state-space-models/cuthberto-carlos/tree/main/",
        ),
      ),
    ).toBe(true);

    render(<App />);

    expect(screen.getByRole("link", { name: /View model/i })).toHaveAttribute(
      "href",
      "https://github.com/state-space-models/cuthberto-carlos",
    );
    expect(screen.getByRole("link", { name: /Latest model snapshot/i })).toHaveAttribute(
      "href",
      expect.stringContaining("https://github.com/state-space-models/cuthberto-carlos/tree/main/"),
    );
    expect(screen.queryByRole("link", { name: "Finals" })).not.toBeInTheDocument();
    expect(screen.queryByText(/complete path to the final/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/finals fixtures/i)).not.toBeInTheDocument();
  });
});
