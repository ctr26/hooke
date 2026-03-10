# GUIDELINES — Ecosystem Workflows

**Version:** 1.0.0
**Status:** Normative
**Scope:** Virtual Cells ecosystem

---

## 1. Purpose

This document defines workflows for the Virtual Cells ecosystem. For technical standards and tool requirements, see [STANDARDS.md](STANDARDS.md).

---

## 2. Development Workflow

### 2.1 Issue-First Development

All work SHALL begin with an issue:

```bash
gh issue create --title "<description>" --body "<details>"
```

### 2.2 Branch Strategy

```
main ─────────●───────────●─────────────
               ↑           ↑
              PR₁         PR₂
               ↑           ↑
         cherry-pick   cherry-pick
               ↑           ↑
unstable/feat ──●──●──●──●──●──●──●──●──
```

| Branch Type | Purpose | Merges To |
|-------------|---------|-----------|
| `main` | Production-ready | — |
| `feature/*` | Clean PR history | `main` |
| `unstable/*` | Iteration workspace | `feature/*` via cherry-pick |

### 2.3 Atomic Commits & PRs

Commits and PRs SHALL be atomic:
- One logical change per commit
- One feature/fix per PR
- If PR grows too large, split into multiple PRs

**Never force push.** Instead:
```bash
git checkout main && git pull
git checkout -b feature/issue-42-v2
git cherry-pick <good-commits>
```

### 2.4 Feature Development Sequence

```bash
# 1. Create branches
git checkout main && git pull
git checkout -b feature/issue-42-description
git checkout -b unstable/issue-42-description

# 2. Iterate on unstable (commit freely)

# 3. Cherry-pick clean commits to feature
git checkout feature/issue-42-description
git cherry-pick <hash>

# 4. Open PR
git push -u origin feature/issue-42-description
gh pr create --fill

# 5. Continue on unstable while PR reviews
git checkout unstable/issue-42-description
```

---

## 3. Pre-Push Validation

For code-bearing repos, before every push:

```bash
make prepush
```

This target SHALL execute linting, type checking, and tests.

For docs-only repos (like hooke hub), validate:
- Markdown syntax
- Link integrity
- No secrets committed

---

## 4. Code Review

### 4.1 Author Responsibilities

| Requirement |
|-------------|
| Self-review diff before requesting review |
| Ensure CI passes |
| Link issue in PR description |
| Respond to all comments |
| Do not force-push during active review |

### 4.2 Reviewer Responsibilities

| Requirement |
|-------------|
| Use Conventional Comments (see STANDARDS §3.5) |
| Approve when acceptable, not perfect |
| Trust author to address feedback |

---

## 5. Ecosystem Coordination

### 5.1 Hub Repo (hooke)

This repo serves as:
- Repository index for the ecosystem
- Source of default engineering standards
- Home for ecosystem-wide documentation

Changes to STANDARDS.md or GUIDELINES.md here affect ecosystem defaults.

### 5.2 Child Repos

Each code repo in the ecosystem:
- MAY have its own STANDARDS.md (overrides ecosystem defaults)
- SHOULD follow the standardise pattern (HUMANS.md present)
- SHALL follow version control conventions from §2

### 5.3 Cross-Repo Changes

For changes spanning multiple repos:
1. Create an issue in each affected repo
2. Link issues together
3. Coordinate PRs with consistent descriptions

---

## 6. Documentation Management

### 6.1 Token Budget

All documentation files SHALL be ≤8,000 tokens. See STANDARDS §5.1.

### 6.2 Single Source of Truth

Content SHALL appear once, with pointers elsewhere. See STANDARDS §5.3.

### 6.3 What + Why Pattern

GUIDELINES and STANDARDS sections SHOULD include both:

```markdown
## [Topic]

### What
Concrete rules, formats, examples.

### Why
Rationale, benefits, trade-offs.
```

### 6.4 Worktree Standards

The working tree MAY include unstaged files for local configuration:

| File | Purpose | Staged |
|------|---------|--------|
| `CLAUDE.md` | Personal AI agent preferences | No |
| `PROJECT.md` | Project-specific context | Optional |
| `.env` | Environment variables | No |

---

## 7. Standardising a New Repo

To bring a new repo into the ecosystem:

1. Copy templates from `.claude/skills/standardise/docs/` for missing files
2. Add `HUMANS.md` to signal standardisation
3. Ensure README links to all docs
4. Verify each doc ≤8k tokens
5. PR with `docs` and `agent` labels (if AI-generated)

**Never overwrite existing docs.** Existing docs represent the repo owner's decisions.

### Verification Checklist

- [ ] README links all docs
- [ ] Each doc ≤8k tokens
- [ ] No duplicate content
- [ ] GUIDELINES has what+why sections
- [ ] STANDARDS has what+why sections
- [ ] Conventional commits used

---

## 8. References

- [STANDARDS.md](STANDARDS.md) — Technical requirements
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contributor guide
- [Engineering Standards (skills)](https://github.com/ctr26/skills)
