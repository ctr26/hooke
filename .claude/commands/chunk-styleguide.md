Download and chunk a style guide into agent-friendly files with a lightweight knowledge graph.

## Input

$ARGUMENTS should be a URL to a style guide (HTML or markdown), or a local file path.
If no argument is provided, ask the user for the URL.

## Workflow

### Step 1: Download

Use WebFetch to download the document from the URL (or Read for local files).
Extract the main content — strip navigation, footers, sidebars.
Convert HTML to markdown if needed (preserve headings, code blocks, lists).

### Step 2: Determine guide name

Derive a short kebab-case name from the document title.
Example: "Google Python Style Guide" → `google-python`

### Step 3: Split on H2 boundaries

Parse the markdown into sections by H2 headings.
For each H2 section:
- If > 100 lines, split further on H3 boundaries
- If still > 100 lines, split on paragraph boundaries (double newline)
- Assign sequential IDs: `001_naming`, `002_imports`, etc.
- Use the heading text (kebab-cased) as the ID suffix

### Step 4: Extract entities per chunk

For each chunk, identify:
- **Rules**: Prescriptive statements ("always use", "never do", "must", "should")
- **Concepts**: Named ideas (e.g., "list comprehension", "type annotation")
- **Patterns**: Recommended code patterns (do this)
- **AntiPatterns**: Discouraged patterns (don't do this)
- **Examples**: Code examples illustrating rules

Output as a list of entity IDs (kebab-case).

### Step 5: Build relationship graph

For each pair of entities, determine relationships:
- `APPLIES_TO`: rule applies to a concept (e.g., "snake-case APPLIES_TO function-naming")
- `CONFLICTS_WITH`: two rules/patterns that are mutually exclusive
- `EXAMPLE_OF`: code example demonstrates a rule
- `OVERRIDES`: specific rule overrides a general one
- `RELATED_TO`: entities that co-occur in same chunk
- `REFERENCED_IN`: entity appears in multiple chunks

### Step 6: Detect communities

Group chunks by theme using the heading hierarchy:
- H1 = global community (the whole guide)
- H2 = topic community (e.g., "Naming", "Imports", "Functions")
- H3+ = sub-community

### Step 7: Write output files

Create the output directory: `.claude/styleguides/<guide-name>/`

**Write each chunk** to `chunks/NNN_topic.md` with frontmatter:
```yaml
---
id: "NNN_topic"
topic: "Human-Readable Topic Name"
entities: ["entity-1", "entity-2"]
community: "community-name"
globs: ["*.py"]  # file patterns this chunk applies to
---
```
Followed by the chunk content.

**Write `_graph.jsonl`** — one JSON object per line:
```json
{"source": "entity-id", "target": "entity-id", "relation": "APPLIES_TO", "chunk": "001_naming"}
```

**Write `_index.md`** — tag-based lookup table:
```markdown
# <Guide Name> Index

| Topic | Chunk | Entities | Community |
|-------|-------|----------|-----------|
| Naming | chunks/001_naming.md | snake-case, camel-case | style |
| Imports | chunks/002_imports.md | import-order, star-imports | structure |
```

**Write `_summary.md`** — high-level summary:
```markdown
# <Guide Name> Summary

## Communities
- **style**: Naming conventions, formatting, whitespace
- **structure**: Imports, modules, packages
- ...

## Key Rules
- <top 10 most-referenced rules with one-line descriptions>
```

### Step 8: Report

Print:
- Number of chunks created
- Number of entities extracted
- Number of relationships
- Output directory path
- Largest chunk (should be ≤100 lines)

## Constraints

- No external dependencies — use only Claude Code built-in tools
- Each chunk must be ≤100 lines (excluding frontmatter)
- Entity IDs are kebab-case
- Determine appropriate `globs` patterns from the guide's language (e.g., `*.py` for Python guides)
- If the guide is very large (>50 H2 sections), group related H2s into larger chunks to keep total files manageable
