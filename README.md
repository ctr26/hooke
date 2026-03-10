# Hooke

Entry point for sane navigation across hooke and its related repos.

See the paper: [Virtual Cells: Predict, Explain, Discover](https://www.valencelabs.com/publications/virtual-cells-predict-explain-discover/)

## Related Repositories

### Active

| Repo | Org | Description | Language |
|------|-----|-------------|----------|
| [hooke-forge](https://github.com/valence-labs/hooke-forge) | valence-labs | Train Diffusion Transformers on phenomics and transcriptomics data using flow matching. Provides `hooke-train`, `hooke-eval`, `hooke-infer` CLIs. | Python |
| [HookeTx](https://github.com/valence-labs/HookeTx) | valence-labs | Transcriptomics perturbation prediction. Uses Hydra configs and PyG. | Python |
| [vcb](https://github.com/valence-labs/vcb) | valence-labs | Virtual Cell Benchmark. Evaluation framework for virtual cell models. | Python |
| [hooke-predict](https://github.com/recursionpharma/hooke-predict) | recursionpharma | Unified multi-modal approach to predict outcomes of biological experiments. | Python |
| [hooke-explain](https://github.com/recursionpharma/hooke-explain) | recursionpharma | Framework for generating, verifying, and evaluating scientific explanations using LLMs. | Python |
| [hooke-explain-tooling](https://github.com/recursionpharma/hooke-explain-tooling) | recursionpharma | External tooling for Hooke Explain. | Python |

### Infrastructure (engineering, not research)

| Repo | Org | Description | Language |
|------|-----|-------------|----------|
| [bc-hooke](https://github.com/recursionpharma/bc-hooke) | recursionpharma | Bounded context repo: ArgoCD deployments, Terraform infrastructure, and Germ metadata. | HCL |

### Inactive

| Repo | Org | Description | Language |
|------|-----|-------------|----------|
| [TxPert](https://github.com/valence-labs/TxPert) | valence-labs | Graph-supported perturbation prediction with transcriptomic data. Paper reproduction repo; historical predecessor to HookeTx. | Python |
| [TxPert](https://github.com/recursionpharma/TxPert) | recursionpharma | Internal fork of TxPert. | Python |

## Engineering Standards

This repo follows the hooke 6-file standard:

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `CONTRIBUTING.md` | PR process, git workflow, commit conventions |
| `GUIDELINES.md` | Code patterns and anti-patterns |
| `STANDARDS.md` | Tooling: uv, ruff, ty, pytest |
| `AGENTS.md` | Agent rules, model selection, safety |
| `PROJECT.md` | Architecture spec |

## Templates

Ready-to-copy project scaffolding in `templates/`:

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python config with ruff, pytest, ty, hatchling |
| `Makefile` | Dev commands: install, test, lint, prepush |
| `.github/workflows/ci.yml` | CI: lint, format, typecheck, test, coverage |
| `PROJECT_SPEC.md` | Architecture spec template |

## Quick Start

```bash
# New project from hooke
gh repo create ctr26/<name> --private --clone && cd <name>
cp ~/projects/ctr26/hooke/AGENTS.md .
cp ~/projects/ctr26/hooke/templates/pyproject.toml .
cp ~/projects/ctr26/hooke/templates/Makefile .
cp -r ~/projects/ctr26/hooke/templates/.github .
cp ~/projects/ctr26/hooke/templates/PROJECT_SPEC.md .
cp -r ~/projects/ctr26/hooke/.claude .
cp -r ~/projects/ctr26/hooke/.cursor .
```

## Agent Configuration

| Location | Purpose |
|----------|---------|
| `.claude/rules/` | Language and domain rules (Python, TS, ML, engineering) |
| `.cursor/rules/` | Mirror of `.claude/rules/` for Cursor IDE |

## License

MIT
