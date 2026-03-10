# hooke

Hub repo for the Hooke ecosystem — a collection of tools and models for
building virtual cells that predict, explain, and discover biological
phenomena. This repo contains no source code; it exists as the single entry
point for navigating and understanding the ecosystem.

Based on the paper:
[Virtual Cells: Predict, Explain, Discover](https://www.valencelabs.com/publications/virtual-cells-predict-explain-discover/)

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add or update repository
entries, branch naming conventions, and PR guidelines.

## For AI Agents

See [AGENTS.md](AGENTS.md) for instructions on what this repo is, what belongs
here, and the conventions to follow. Cursor-specific rules are in
`.cursor/rules/`.
