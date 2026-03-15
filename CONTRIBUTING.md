# Contributing

## Workflow

1. Fork the repo (or create a branch for owned repos)
2. First commit = `PROJECT_SPEC.md` + standards files
3. Work in a git worktree: `git worktree add ../<repo>-issue-<N> issue-<N>`
4. One issue = one worktree = one PR
5. Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`)

## Pull Requests

- All PRs require CI to pass (lint + type check + tests)
- No force push to `main`, `dev`, or `trunk`
- If a PR needs major changes, close it and open a new one
- Don't delete failing tests — fix the patch
- Reviews require PE (Principle Engineer) + Scientist sign-off
- No auto-merge

## Commit Messages

Use conventional commits:

```
feat: add GP interpolation to PSF module
fix: handle NaN values in batch normalization
chore: update ruff to 0.3.0
docs: add ADR for config migration
```

Write like a developer. Brief context, clear change. No boilerplate.

## Pre-Push Checklist

```bash
make prepush   # runs: lint, typecheck, test-fast
```

Always run before pushing. CI will catch it anyway, but save the round-trip.

## Issues

- Only post issues on owned repos (`ctr26/*`, `craggles17/*`)
- Sound human — no robotic templates
- Brief context, clear ask
- Example: "PSF interpolation missing GP — need it for the comparison table"

## Agent Contributions

Agents follow the same rules as humans:

- Spawn via worktree, one issue per agent
- Commit with conventional messages
- Create PR, verify CI passes
- Model selection: opus for architecture, sonnet for standard work, cheap for docs

## Code Review

- PE reviews architecture + code quality
- Focus on: correctness, type safety, test coverage, simplicity
- Every 10 issues/PRs = leadership meeting
