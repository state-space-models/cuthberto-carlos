export interface Player {
  number: number;
  position: "GK" | "DF" | "MF" | "FW";
  name: string;
  dateOfBirth: string;
}

export interface Team {
  name: string;
  flagCode: string;
  colors: string[];
  fifaCode: string;
  group: string;
  players: Player[];
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
  homeGoals: ActualGoal[];
  awayGoals: ActualGoal[];
}

export interface ActualGoal {
  name: string;
  minute: string;
  penalty?: boolean | null;
}

export interface MatchPrediction {
  id: string;
  date: string;
  kickoffUtc: string;
  group: string;
  venue: string;
  homeTeam: string;
  awayTeam: string;
  predictionDate: string;
  sourceUrl: string;
  predictionHistory: PredictionHistoryEntry[];
  prediction: PredictionDetails;
  polymarket?: PolymarketPrediction;
  actualResult?: ActualResult;
}

export interface PolymarketPrediction {
  homeWin: number;
  draw: number;
  awayWin: number;
  eventUrl: string;
  updatedAt: string;
}

export interface PredictionHistoryEntry {
  predictionDate: string;
  sourceUrl: string;
  prediction: PredictionDetails;
}

export interface PredictionDetails {
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
}

export interface GroupProjectionRow {
  rank: number;
  team: string;
  expectedPoints: number;
  expectedGoalsFor: number;
  expectedGoalsAgainst: number;
  expectedGoalDifference: number;
}

export interface GroupActualStats {
  games: number;
  wins: number;
  draws: number;
  losses: number;
  points: number;
  goalDifference: number;
  goalsScored: number;
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
  team1?: string;
  team2?: string;
  score?: KnockoutScore;
  prediction?: PredictionDetails;
  predictionDate?: string;
  sourceUrl?: string;
  predictionHistory?: PredictionHistoryEntry[];
  polymarket?: PolymarketPrediction;
}

export interface KnockoutScore {
  fullTime: [number, number];
  extraTime?: [number, number];
  penalties?: [number, number];
}

export interface TournamentDataset {
  schemaVersion: number;
  repositoryUrl: string;
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
    squads: DataSource;
    historicalResults: DataSource;
    polymarket: PolymarketDataSource;
  };
  teams: Record<string, Team>;
  groupMatches: MatchPrediction[];
  groups: GroupProjection[];
  knockoutMatches: KnockoutMatch[];
}

export interface DataSource {
  name: string;
  url: string;
  license?: string;
  commit?: string;
  dataUrl?: string;
  ref?: string;
}

export interface PolymarketDataSource extends DataSource {
  dataUrl: string;
  tagId: string;
  seriesId: string;
  marketType: string;
}
