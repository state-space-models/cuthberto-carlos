const DEFAULT_REPOSITORY = "state-space-models/cuthberto-carlos";

export function getRepositoryName(
  repository = process.env.GITHUB_REPOSITORY ?? DEFAULT_REPOSITORY,
): string {
  if (!/^[^/\s]+\/[^/\s]+$/.test(repository)) {
    throw new Error(`Invalid GitHub repository slug: ${repository}`);
  }

  return repository.split("/")[1];
}

export function getPagesBase(repository?: string): string {
  return `/${getRepositoryName(repository)}/`;
}
