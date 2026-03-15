# AGENTS.md - Engineering Standards

## Project Defaults

- **Visibility**: Private unless explicitly stated
- **Workflow**: Fork → clone (easier GHA management)
- **First commit**: PROJECT_SPEC.md + standards files
- **Multi-agent repos**: Git worktrees + explicit PRs
- **Never force push**: main, dev, trunk are protected — no `--force`
- **Close PRs, don't force push fixes**: If PR needs major changes, close it and open a new one
- **Only post issues on owned repos**: ctr26/* and craggles17/* only — never upstream
- **Tool-agnostic agents**: Rules/configs must work with Cursor AND Claude Code — no vendor lock-in

## Issue Writing Style

- **Sound human** — no robotic templates
- Write like a developer would, not an AI
- Brief context, clear ask, no boilerplate headers
- Example: "The PSF interpolation is missing GP — need to add it for the comparison table in the paper"
- NOT: "## Summary\n\n### Context\n\nThis issue tracks the implementation of..."

## Architecture Principles

- **Cheap**: Minimize hosting costs, use free tiers
- **Scalable**: Cloud-first, horizontal scaling
- **Low LOC**: Minimal code, maximum impact
- **Minimal deps**: Fewer dependencies = fewer vulnerabilities
- **Good OSS**: MIT/Apache when public, clear docs, contribution guidelines

## ML & Research Engineering

### Audit-Readiness
> "Behave as if we could be audited at any time."

Every file, checkpoint, and result should be regenerable from:
- A git commit
- A resolved config
- A data version
- A single command

### Core Principles

1. **Baselines First**
   - Start with the simplest model that could work
   - Establish performance floor before complexity
   - Plan ablations upfront to justify each component

2. **Compute-Aware Experiment Design**
   - Assume No QoS (Quality of Service)
   - Design for preemption: checkpoints must be resumable
   - Fail early: validate config/data before GPU allocation
   - Use submitit or equivalent for cluster management

3. **Bus Factor Reduction**
   - Typed interfaces (Pydantic, dataclasses) as contracts
   - Schema as pipe shapes — validates at boundaries
   - Document WHY, not just WHAT (use ADRs)
   - Reading docs is as important as writing them

4. **Configuration Management**
   - Hydra/Hydra-Zen for hierarchical configs
   - YAML per experiment with full reproducibility
   - Config lifecycle: development → validation → frozen
   - Store resolved configs with every checkpoint

5. **Experiment Tracking**
   - W&B (or MLflow) for all experiments
   - Log hyperparams, metrics, artifacts
   - Tag experiments with git commit SHA
   - Link to data versions explicitly

## Python

- **Modern**: 3.11+ for applications
- **Library support**: 3.9+ when publishing packages
- **Always**: Reusable, modular, type-hinted where helpful

## Code Quality

- Unit tests for core logic (aim 80%+ coverage)
- Schemas for data validation (Pydantic/Zod)
- Type hints where it helps
- CI for lint + test on every PR

## Governance

- Every 10 issues/PRs = leadership meeting
- PM writes issues to GitHub (not just notes)
- Principle Engineer reviews architecture + code quality
- Testing issues have highest priority

## Safety

### Disk Operations
- **DRY RUN ONLY** for cleanup operations
- Always use `--dry-run`, `-n`, or equivalent
- Never auto-delete; require explicit confirmation
- Log before executing; provide undo path
- **Prefer archive over delete** — risk of deletion is high
- Rename/move instead of rm where possible

### Destructive Actions
- Explicit user confirmation required
- Log intent before execution
- Reversible where possible

## Model Selection

| Task Type | Model | Alias |
|-----------|-------|-------|
| Architecture, code review, novel problems | claude-opus-4-5 | opus |
| Standard dev, features, docs | claude-sonnet-4-5 | sonnet |
| Bulk/repetitive tasks | Cheap tier (mistral-7b, llama-3) | dogs-body |
| Trivial tasks | Smallest available | kakapo |

**Budget rule**: Reserve 10% for critical Opus decisions.

## Issue Priority

1. CI/CD setup
2. Test infrastructure
3. Core functionality
4. Features
5. Documentation

## Workflow

```
PM creates issue → Spawn agent → Agent uses worktree → PR to main → Review → Merge
```

## Self-Improvement Meta-Rules

### Continuous Learning Protocol

**After every interaction, ask yourself:**
1. "Could a rule have prevented this mistake?"
2. "Should I update guidance for next time?"
3. "What did I learn that others should know?"

### When to Log Learnings

**Immediate capture** (don't wait):
- User corrects you: "no, actually...", "that's wrong"
- Non-obvious solution discovered (>10 min investigation)
- Error message was misleading
- Configuration differs from documentation
- Pattern worked but wasn't in existing rules

### Learning Quality Gates

**High-value learnings** (promote to docs):
- [ ] Reusable across contexts (not one-off)
- [ ] Non-trivial (requires discovery, not just docs)
- [ ] Specific trigger conditions
- [ ] Verified solution (actually works)
- [ ] No duplication (new insight)

### Promotion Hierarchy

```
.learnings/         ← Raw capture (errors, corrections, features)
AGENTS.md           ← Workflow patterns & delegation
docs/               ← Domain knowledge & decisions (ADRs)
```

### Self-Improvement Guardrails

**Anti-Drift Limits** (forbidden evolution):
- Don't add complexity to "look smart"
- Don't make unverifiable changes
- Don't use vague justifications ("intuition")
- Don't sacrifice stability for novelty

**Priority**: Stability > Explainability > Reusability > Scalability > Novelty

### Verify Before Reporting (VBR)

**When about to say "done", "complete", "finished"**:
1. STOP before typing that word
2. Test the feature from user's perspective
3. Verify the outcome (not just the output)
4. Check the mechanism (not just the text)
5. Only THEN report complete

## Git Rules

### Before ANY push, pre-flight check:
```bash
# 1. Verify remote shows correct account
git remote -v | grep -E "(ctr26|craggles17)/" || echo "WRONG ACCOUNT"

# 2. Test remote access
git ls-remote origin HEAD >/dev/null 2>&1 || echo "CANNOT ACCESS REMOTE"

# 3. Only then push
git push ...
```

### Context before working:
```bash
git log --oneline -20                    # Recent history
git log --oneline --grep="<keyword>"     # Find relevant commits
git show <commit>                        # Understand changes
```

## Credentials

This repo uses ctr26 credentials.
Verify with: `gh auth status`

If wrong account, switch: `gh auth switch`
