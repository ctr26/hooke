# <Project Name> - Project Specification

## Problem

<!-- What problem does this solve? Why does it matter? -->

## Solution

<!-- High-level approach. 2-3 sentences max. -->

## Architecture

```
<project>/
├── src/<package>/     # Core library
│   └── ...
├── tests/             # Test suite
├── configs/           # Configuration files (if applicable)
└── scripts/           # CLI entry points (if applicable)
```

### Tech Stack

- **Python**: 3.11+
- **Package manager**: `uv`
- **Linting/Format**: `ruff`
- **Type checking**: `ty`
- **Testing**: `pytest` + `pytest-cov` (80%+ coverage)

<!-- Add project-specific stack items below -->

## Milestones

1. **Infrastructure** (CI/CD, testing, type checking)
2. <!-- Core milestone -->
3. <!-- Feature milestone -->

## Non-Goals

<!-- What this project explicitly does NOT do -->

## Quality Standards

- **CI**: All PRs must pass lint + type check + tests
- **Coverage**: 80%+ on core logic
- **Type hints**: All public APIs
- **Documentation**: Docstrings on public functions

## Security

- **Private repo**: Default until production-ready
- **Credentials**: Never commit secrets, use env vars
- **Dependencies**: Minimal, audited for vulnerabilities

## References

- Engineering standards: `ctr26/hooke`

---

**First commit**: This file + standards files
**Next**: GitHub issues for each milestone
