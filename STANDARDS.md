# STANDARDS — Ecosystem Technical Standards

**Version:** 1.0.0
**Status:** Normative
**Scope:** Virtual Cells ecosystem

---

## 1. Definitions

| Term | Definition |
|------|------------|
| **SHALL** | Absolute requirement |
| **SHALL NOT** | Absolute prohibition |
| **SHOULD** | Recommended unless justified |
| **MAY** | Optional |

---

## 2. Scope

This repo (hooke) is the ecosystem hub. Standards here serve as **defaults** for all related repositories. Individual repos MAY override with their own STANDARDS.md.

| Standard | Applies to |
|----------|-----------|
| Version control (§3) | All repos |
| Security (§4) | All repos |
| Documentation (§5) | All repos |
| Code quality (§6) | Code-bearing repos only |

---

## 3. Version Control

### 3.1 Commit Messages

Format: `<type>: <description>`

| Type | Usage |
|------|-------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `test` | Test addition/modification |
| `chore` | Maintenance |
| `refactor` | Code restructure |

Constraints:
- Present tense ("add" not "added")
- Lowercase
- No trailing period
- Reference issue when applicable: `feat: add auth (#42)`

### 3.2 Branch Naming

Format: `<type>/issue-<N>-<description>`

| Type | Usage |
|------|-------|
| `feature` | New functionality |
| `bugfix` | Defect correction |
| `chore` | Maintenance |
| `docs` | Documentation |
| `experimental` | Exploratory work |

### 3.3 Pull Requests

| Requirement | Enforcement |
|-------------|-------------|
| Issue linkage | PR template |
| CI passage | Branch protection |
| Description | PR template |
| Merge method | Squash and merge |

### 3.4 PR Labels

PRs SHALL use conventional labels:

| Label | Usage |
|-------|-------|
| `agent` | AI-generated PR (required for agent PRs) |
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `chore` | Maintenance |
| `refactor` | Code restructure |
| `test` | Test changes |

### 3.5 Review Comments

Prefix format per [Conventional Comments](https://conventionalcomments.org/):

| Prefix | Meaning | Action Required |
|--------|---------|-----------------|
| `issue:` | Defect | Yes |
| `suggestion:` | Improvement | Optional |
| `question:` | Clarification needed | Response |
| `nitpick:` | Minor style | Optional |
| `praise:` | Positive feedback | None |

---

## 4. Security

| Requirement | Enforcement |
|-------------|-------------|
| No hardcoded credentials | `gitleaks` in CI |
| Secrets via environment | Review |
| Private repos by default | Repository settings |

---

## 5. Documentation

### 5.1 Token Budget

All documentation files SHALL be ≤8,000 tokens (~32KB, ~6,000 words).

| Reason | Benefit |
|--------|---------|
| LLM context limits | Docs fit in prompts |
| Cognitive load | Readable in one sitting |
| Forces concision | No rambling |

### 5.2 Structure

Each project SHOULD have:

| Document | Purpose | Target Size |
|----------|---------|-------------|
| README.md | Overview + links | 1-2k tokens |
| CONTRIBUTING.md | Contributor guide | 500-1k tokens |
| GUIDELINES.md | Workflows (what + why) | 3-6k tokens |
| STANDARDS.md | Requirements (what + why) | 3-6k tokens |
| HUMANS.md | Standardisation signal | 1-2k tokens |

### 5.3 Single Source

Content SHALL appear once, with pointers elsewhere:

```markdown
# Single source + pointer
GUIDELINES: "Use conventional commits..."
CONTRIBUTING: "See commit conventions in GUIDELINES.md"
```

### 5.4 What + Why Pattern

GUIDELINES and STANDARDS sections SHOULD include both the rule and the rationale.

---

## 6. Code Quality (for code-bearing repos)

### 6.1 Required Practices

Projects SHALL implement:

| Practice | Purpose |
|----------|---------|
| Type checking | Catch errors at analysis time |
| Linting | Consistent style, catch bugs |
| Unit tests | Verify individual components |
| Pre-commit checks | Catch issues before push |

### 6.2 Recommended Tools

| Tool | Purpose | Why |
|------|---------|-----|
| uv | Package management | 100x faster than pip, reproducible |
| ruff | Linting + formatting | Replaces 5 tools, instant |
| ty | Type checking | Faster than mypy, stricter |
| pytest | Testing | Standard, extensible |

### 6.3 Entry Points

Projects SHOULD implement consistent targets:

```bash
make install    # Setup
make lint       # Static analysis
make test       # Test execution
make prepush    # Pre-push validation
```

### 6.4 Type Safety

| Requirement | Enforcement |
|-------------|-------------|
| Type annotations on public APIs | Type checker |
| Avoid `any`/`Any` | Justify with comment if needed |

### 6.5 Architecture Constraints

| Constraint | Rationale |
|------------|-----------|
| Shallow inheritance | Easier to understand |
| Functions over classes | When stateless, prefer simplicity |
| Validate at boundaries | Fail fast, trust internal data |

---

## 7. Exceptions

| Context | Waivable Requirements |
|---------|----------------------|
| Prototype/spike | Coverage threshold |
| One-off script | Full CI |
| Vendor/generated code | All lint rules |

Document exceptions inline:
```python
# type: ignore[arg-type]  # Legacy API constraint
# pragma: no cover  # Production-only path
```

---

## 8. References

- [GUIDELINES.md](GUIDELINES.md) — Workflows and processes
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contributor guide
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Conventional Comments](https://conventionalcomments.org/)
- [Engineering Standards (skills)](https://github.com/ctr26/skills)
