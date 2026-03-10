# HUMANS.md

**If this file exists, this repository has been standardised.**

---

## What This Means

This repo follows a documentation system designed for both humans and AI agents.

### For Humans

You already know how to read docs. Nothing changes for you — README, CONTRIBUTING, GUIDELINES, and STANDARDS work exactly as you'd expect.

### For Agents

AI agents (Claude, Codex, Copilot, etc.) can use lightweight **skills** to navigate this repo:

| Skill | Purpose |
|-------|---------|
| `what` | Understand requirements and constraints |
| `why` | Understand rationale and design decisions |
| `where` | Find the right location for changes |
| `how` | Find workflows and processes |
| `who` | Find relevant reviewers |
| `standardise` | Set up or improve repo docs |

Skills are available via submodule at `.claude/skills/`.

**Agents should automatically:**
- Read these docs before writing code
- Write code that complies with STANDARDS
- Follow GUIDELINES workflows
- Use conventional commits
- Request appropriate reviewers

---

## Ecosystem Context

This is the **hub repo** for the Virtual Cells ecosystem. Standards defined here apply across all related repositories:

| Scope | Applies to |
|-------|-----------|
| Version control | All ecosystem repos |
| Security | All ecosystem repos |
| Documentation structure | All ecosystem repos |
| Code quality (Python) | Code-bearing repos (hooke-forge, HookeTx, vcb, etc.) |

Individual repos MAY override with their own STANDARDS.md, but defaults flow from here.

---

## Design Goals

| Goal | How |
|------|-----|
| **General** | Works for any language, any project |
| **Convenient** | Uses docs you already have (or provides templates) |
| **Pervasive** | Same system across all ecosystem repos |
| **Low lift** | Add once, benefit forever |

### Zero Configuration

Agents that understand this system will:
1. Detect HUMANS.md → know repo is standardised
2. Read root docs automatically
3. Write compliant code without being asked
4. Follow commit conventions
5. Create properly labeled PRs

### Optimised for Both Audiences

| For Humans | For Agents |
|------------|------------|
| Info in docs humans actually read | Each doc ≤8k tokens (fits context window) |
| Important stuff surfaces to top | Consistent structure for parsing |
| Less hunting through long files | Links instead of duplication |
| Single source of truth | No conflicting information |

---

## The System

```
┌─────────────────────────────────────────┐
│              Root Docs                  │
│  README → CONTRIBUTING → GUIDELINES     │
│              → STANDARDS                │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
   ┌────▼────┐        ┌─────▼─────┐
   │  Human  │        │   Agent   │
   │ (reads) │        │  (skill)  │
   └─────────┘        └───────────┘
```

One source of truth. Two audiences.

---

## If You're New Here

1. Start with **README.md** — project overview and repo index
2. Read **CONTRIBUTING.md** — how to contribute
3. Check **GUIDELINES.md** — workflows and processes
4. Reference **STANDARDS.md** — technical requirements

If you're an agent, use the skills at `.claude/skills/`. If you're human, just read the docs.

---

*Standardised by [engineering-standards](https://github.com/ctr26/skills)*
