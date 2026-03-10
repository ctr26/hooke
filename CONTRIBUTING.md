# CONTRIBUTING — Contributor Guide

> For humans and AI agents. Full specification: [GUIDELINES.md](GUIDELINES.md).

---

## For AI Agents

**Always use skills before acting.** Run through core skills at `.claude/skills/`:
- `what` — understand requirements
- `why` — understand rationale
- `where` — find the right location
- `how` — find prior art, plan approach

**Always contextualize via root docs.** Before non-trivial work:
1. Read README.md → Contents → relevant docs
2. Cite STANDARDS.md or GUIDELINES.md to justify decisions
3. If no doc supports the decision, flag as new convention

**Standardise iteratively:**
- Small patches, not big rewrites
- One section at a time
- Commit after each change
- Information SHALL NOT be lost
- Minimize token/char count (tables > prose)

---

## Quick Start

```bash
gh repo fork ctr26/hooke --clone
cd hooke
git checkout -b feature/issue-N-description
# ... make changes ...
gh pr create --fill
```

## Commit Format

```
<type>: <description>

Types: feat | fix | docs | test | chore | refactor
```

- Present tense ("add" not "added")
- Lowercase
- No trailing period
- Reference issue when applicable: `docs: update repo index (#42)`

## Non-Negotiable

| Rule | Enforcement |
|------|-------------|
| Conventional Commits | `feat:`, `fix:`, `docs:`, `test:`, `chore:` |
| Issue linkage | PR links issue |
| No secrets | Never commit credentials or tokens |

## Workflow

1. Issue first — `gh issue create --title "..."`
2. Branch: `feature/issue-N-desc` or `docs/issue-N-desc`
3. PR links issue
4. Squash and merge

## Ecosystem Contributions

This is the hub repo for Virtual Cells. For code contributions to specific repos:

| Repo | Guide |
|------|-------|
| hooke-forge | See its own CONTRIBUTING.md |
| HookeTx | See its own CONTRIBUTING.md |
| vcb | See its own CONTRIBUTING.md |
| Other repos | Check for HUMANS.md → follow local docs |

Standards in this repo's [STANDARDS.md](STANDARDS.md) serve as ecosystem defaults.

## Review Comments

Use [Conventional Comments](https://conventionalcomments.org/):

| Prefix | Meaning | Action Required |
|--------|---------|-----------------|
| `issue:` | Defect | Yes |
| `suggestion:` | Improvement | Optional |
| `question:` | Clarification needed | Response |
| `nitpick:` | Minor style | Optional |
| `praise:` | Positive feedback | None |

## PR Labels

| Label | Usage |
|-------|-------|
| `agent` | AI-generated PR (required for agent PRs) |
| `docs` | Documentation changes |
| `feat` | New feature |
| `fix` | Bug fix |
| `chore` | Maintenance |
