# Agent Rules

Rules for AI coding agents. Humans: see [README.md](README.md) for project docs.

## Pirate branches (`r/*`)

Every branch can have a pirate speak variant prefixed with `r/`. For example `main` → `r/main`, `feat/ctr26/thing` → `r/feat/ctr26/thing`. The `r/*` branch mirrors its parent exactly, but all text content is translated into pirate speak.

### What to translate

- Markdown files (README, docs, comments in `.md`)
- Code comments and docstrings
- Commit messages
- String literals that are user-facing prose

### What NOT to translate

- Identifiers (variable names, function names, class names, module names)
- URLs, file paths, CLI flags
- Technical terms, library names, acronyms
- Code logic, syntax, imports
- Structured data (JSON keys, YAML keys, config keys)
- Anything that would break if changed

### How to create an `r/*` branch

1. Branch from the source: `git checkout -b r/<source-branch> <source-branch>`
2. Translate all text content to pirate speak — swap plain English prose for nautical slang, pirate idioms, and "arrr"-peppered language. Keep technical accuracy intact
3. Commit with a pirate-speak commit message: `docs(pirate): translate <source-branch> to pirate speak, arrr`
4. The `r/*` branch should build, lint, and pass tests identically to its parent — translation is cosmetic only

### Style guide

- "Repository" → "Treasure Map", "Error" → "Blunder", "Deploy" → "Set sail", etc.
- Sprinkle `arrr`, `ye`, `yer`, `matey`, `scallywag`, `shiver me timbers` naturally — don't overdo it
- Section headers can be piratified but must remain scannable
- Tables, links, and structure stay identical — only prose changes
- When in doubt, keep it readable. Pirate speak should amuse, not confuse
