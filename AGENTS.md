# Agents

Instructions for AI agents working in this repository.

## What this repo is

Hooke is a **hub / navigation repo**. It contains no source code, no packages,
and no deployable artifacts. Its purpose is to serve as the single entry point
for discovering and understanding the Hooke ecosystem of repositories.

## What belongs here

- The README table of related repositories (keep it accurate and up-to-date).
- Documentation about the ecosystem as a whole (architecture overviews, onboarding guides).
- Agent and editor configuration (this file, `.cursor/rules/`).
- Contributor guidelines (`CONTRIBUTING.md`).

## What does NOT belong here

- Application or library source code.
- Package manifests (`pyproject.toml`, `setup.py`, `requirements.txt`).
- CI/CD workflows that build or deploy software.
- Large binary assets.

## Conventions

- **Markdown only.** All content is Markdown. Follow existing table formatting
  in the README when adding or editing repository entries.
- **Keep links canonical.** Always link to the GitHub URL of a repo, not to
  mirrors or forks, unless the fork is the canonical copy.
- **One concern per PR.** A PR should either update documentation *or* change
  agent/editor config, not both at once (this bootstrap PR is the exception).
- **Branch naming.** Use `feat/`, `fix/`, or `docs/` prefixes.

## Editor configuration

Cursor-specific rules live in `.cursor/rules/`. These are loaded automatically
by Cursor and provide project-aware guidance to AI assistants.
