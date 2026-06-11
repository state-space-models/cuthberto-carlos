export interface Team {
  name: string;
  flagCode: string;
  colors: string[];
}

export interface ResultProbabilities {
  homeWin: number;
  draw: number;
  awayWin: number;
}

export interface SkillEstimate {
  mean: number;
  sd: number;
}

export interface TeamSkills {
  attack: SkillEstimate;
  defence: SkillEstimate;
}

export interface ActualResult {
  homeScore: number;
  awayScore: number;
  homeGoals: Array<{ name: string; minute: number; penalty?: boolean }>;
  awayGoals: Array<{ name: string; minute: number; penalty?: boolean }>;
}

export interface MatchPrediction {
  id: string;
  date: string;
  kickoffUtc: string;
  group: string;
  venue: string;
  homeTeam: string;
  awayTeam: string;
  sourceUrl: string;
  prediction: {
    probabilities: ResultProbabilities;
    scoreGrid: number[][];
    mostLikelyScore: [number, number];
    mostLikelyScoreProbability: number;
    expectedGoals: {
      home: number;
      away: number;
    };
    skills: {
      home: TeamSkills;
      away: TeamSkills;
    };
  };
  actualResult?: ActualResult;
}

export interface GroupProjectionRow {
  rank: number;
  team: string;
  expectedPoints: number;
  expectedGoalsFor: number;
  expectedGoalsAgainst: number;
  expectedGoalDifference: number;
}

export interface GroupProjection {
  id: string;
  name: string;
  projection: GroupProjectionRow[];
  matchIds: string[];
}

export type KnockoutRound =
  | "Round of 32"
  | "Round of 16"
  | "Quarter-final"
  | "Semi-final"
  | "Match for third place"
  | "Final";

export interface KnockoutMatch {
  id: string;
  matchNumber: number;
  round: KnockoutRound;
  date: string;
  kickoffUtc: string;
  venue: string;
  team1Slot: string;
  team2Slot: string;
}

export interface TournamentDataset {
  schemaVersion: number;
  snapshotDate: string;
  snapshotPath: string;
  snapshotUrl: string;
  generatedAt: string;
  sourceCommit: string;
  model: {
    name: string;
    resultProbabilityOrder: string[];
    scoreGridMaxGoals: number;
  };
  sources: {
    schedule: DataSource;
    historicalResults: DataSource;
  };
  teams: Record<string, Team>;
  groupMatches: MatchPrediction[];
  groups: GroupProjection[];
  knockoutMatches: KnockoutMatch[];
}

export interface DataSource {
  name: string;
  url: string;
  license: string;
  commit?: string;
}
