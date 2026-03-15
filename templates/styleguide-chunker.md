# Style Guide Chunker — Skill Template

> Copy `.claude/commands/chunk-styleguide.md` and `.claude/rules/styleguide.mdc` into any project to enable `/chunk-styleguide`.

## What It Does

Downloads a style guide (URL or local file) and segments it into agent-friendly chunks with a lightweight knowledge graph. Output goes to `.claude/styleguides/<guide-name>/`.

## Output Structure

```
.claude/styleguides/<guide-name>/
├── _index.md          # Tag-based lookup table
├── _graph.jsonl       # Entity-relationship triples (JSONL)
├── _summary.md        # Global summary with communities and key rules
└── chunks/
    ├── 001_naming.md  # ≤100 lines, frontmatter with entities
    ├── 002_imports.md
    └── ...
```

## Chunk Frontmatter

```yaml
---
id: "002_imports"
topic: "Imports"
entities: ["import-order", "absolute-imports", "wildcard-imports"]
community: "code-structure"
globs: ["*.py"]
---
```

## Graph Format (`_graph.jsonl`)

One JSON object per line:
```json
{"source": "snake-case", "target": "function-naming", "relation": "APPLIES_TO", "chunk": "001_naming"}
{"source": "wildcard-imports", "target": "explicit-imports", "relation": "CONFLICTS_WITH", "chunk": "002_imports"}
```

### Relationship Types

| Relation | Meaning |
|----------|---------|
| `APPLIES_TO` | Rule applies to a concept |
| `CONFLICTS_WITH` | Mutually exclusive rules/patterns |
| `EXAMPLE_OF` | Code example demonstrates a rule |
| `OVERRIDES` | Specific rule overrides a general one |
| `RELATED_TO` | Entities co-occur in same chunk |
| `REFERENCED_IN` | Entity appears in multiple chunks |

## Usage

```
/chunk-styleguide https://google.github.io/styleguide/pyguide.html
/chunk-styleguide ./docs/style-guide.md
```

## Required Files

| File | Purpose |
|------|---------|
| `.claude/commands/chunk-styleguide.md` | Slash command definition |
| `.claude/rules/styleguide.mdc` | Agent rule for querying chunks |
| `.claude/hooks/styleguide-chunks.sh` | Optional: nudges agents to use the command |

## Constraints

- No external dependencies — uses only Claude Code built-in tools
- Each chunk ≤100 lines (excluding frontmatter)
- Entity IDs are kebab-case
- Globs inferred from guide language (e.g., `*.py`, `*.ts`)
- Large guides (>50 H2 sections) group related sections to keep file count manageable
