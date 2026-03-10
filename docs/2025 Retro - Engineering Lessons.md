# 2025 Retro → Engineering Lessons

> Source: [Notion](https://www.notion.so/3070c0d8de70811e8adff8a9f0d159ea)
> Parent: HOOKE-1: Engineering

## Audit-Readiness Standard

- Every experiment must be regenerable from: **git commit + config + data version + single command**
- An experiment without a committed config is not reproducible
- A checkpoint without a logged run is not traceable
- A result without a data version is not trustworthy
- Practices in Engineering Practices are the minimum bar, not aspirational

---

## Retro Pain Points → Engineering Practices

Mapping from 2025 retrospective themes to concrete practices. Format: **pain point → applicable practices → remaining gap.**

---

### 1. Complexity Overkill / Premature Optimization

**Retro:** Built too-complex solutions too early. Long iteration cycles. Hard to isolate failures. (6 stop, 4 discuss)

**Practices:**

- Fewer LOC → smaller debugging surface (Code Standards)
- Opinionated config lifecycle: Pretrain → Eval → Infer → Finetune → Benchmark. Each stage works end-to-end before the next gets complex (Code Standards)
- Every design choice is a config parameter → isolate failures via ablation (Code Standards)
- Config per experiment → recorded artifact, not terminal history (Code Standards)
- Evolve schemas, don't fork. New fields get defaults, old results still load (Code Standards)

**Gap — baselines first:** Get a simple model working end-to-end before adding complexity. Never break the baseline.

---

### 2. Integration Bottleneck

**Retro:** Every workstream feeds one central module. One overwhelmed group does all integration. (7 stop, 2 discuss)

**Practices:**

- Schemas as pipe shapes → integration is mechanical, not artisanal (Code Standards)
- Functions emit friendly primitives (tensors, DataFrames, dataclasses) (Code Standards)
- Standard HuggingFace interface → anyone can plug in (Engineering Practices)
- Pydantic validation on configs → catch incompatibilities at config time (Code Standards)
- Datasets test against schema → upstream changes don't silently break downstream (Data Pipeline Standards)

**Gap — bus factor:** If one person is the only one who can integrate, the architecture is wrong. Typed interfaces make integration a mechanical process anyone can perform.

---

### 3. Fragmented Knowledge & Silos

**Retro:** Silos despite small team. Limited knowledge transfer. Specific people become bottlenecks. (8 stop, 2 discuss)

**Practices:**

- README per major directory → self-onboarding (Code Standards)
- Docstrings with shape annotations (Code Standards)
- Architecture Decision Records (see ADR section below) (Code Standards)
- 4-week release cadence → prevents version drift (Code Standards)
- Consistent style via ruff / PEP 8 (Code Standards)
- Clear and explicit logging (Code Standards)

**Gap — reading docs:** Writing docs is necessary but not sufficient. Self-onboarding is an expectation. First step when joining a workstream: read the README, ADRs, and docstrings.

---

### 4. GPU Constraints & Preemption

**Retro:** Resource reallocation, preemption disrupting long runs. (2 discuss)

**Practices:**

- Assume No QoS: checkpoint aggressively, make runs resumable, fail in first 30 seconds (Code Standards)
- `submitit` for job submission → handles requeuing natively (Code Standards)
- Fail loudly and early → validate everything before burning GPU time (Code Standards)
- Re-queueing first mentality:
  - Strong checkpoints
  - **Idempotency, and determinism**

**Gap — compute-aware experiment design:** Know cost per run. Prioritise cheap experiments first. Treat compute as a budget.

---

### 5. Isolation / Silos

**Retro:** Feeling isolated. Absence of guidance. Working in a vacuum. (7 stop, 2 discuss)

**Practices:**

- W&B tracking with programmatic grouping → work is visible (Experiment Tracking)
- Config per experiment → self-documenting artifacts (Code Standards)
- Typed configs make intent explicit (Code Standards)
- Clear logging → colleagues understand runs without interrupting you (Code Standards)

**Note:** Engineering practices make work legible but don't replace communication and alignment.

---

### 7. Generalisation Trade-off

**Retro:** Prioritised generality (multimodal/LLM) too early at expense of scalability. (3 stop, 1 discuss)

**Practices:**

- Opinionated lifecycle → invest rigour proportional to usage frequency (Code Standards)
- Plan ablations as part of engineering → "does the LLM help?" is a one-field config change (Code Standards)
- Minor versions superset → add generality without breaking the simpler path (Code Standards)

**Gap:** Same as §1. Simple model working end-to-end across all data > general model working on a subset.

---

## Summary: Gaps to Fill

1. **Baselines first.** Simple end-to-end before complexity. Never break the baseline.
2. **Compute-aware design.** Cost per run. Cheap experiments first. Compute is a budget.
3. **Bus factor reduction via typed interfaces.** Integration should be mechanical, not person-dependent.
4. **Reading docs = writing docs.** Self-onboarding is an expectation. README → ADRs → docstrings first.
5. **Audit-readiness is daily.** Git commit + config + data version + single command. Always.
6. **Single-command pipelines.** One entry point, automatic caching. Manual step-chaining is fragile: forgotten re-runs, stale outputs, undocumented ordering. Single entry point + cache = both speed and correctness.

---

## Typed Pipeline Contracts & Config Architecture

Three patterns addressing integration bottleneck, fragmented knowledge, and complexity overkill.

---

### Pattern 1: Pydantic Schemas as Pipe Shapes

**Principle:** Output of stage N is a typed field on stage N+1's config. Not a dict key, not a YAML interpolation.

```python
class TaskGenOutput(BaseModel):
    task_parquet_path: Path
    modalities: frozenset[Modality]
    n_samples: int
    outer_fold: int       # carried explicitly for downstream
    inner_fold: int
    split: Split

class HookeDatasetConfig(BaseModel):
    task: TaskGenOutput   # ← stage N output IS a field on stage N+1 config
    max_seq_len: int = 2048
    augment: bool = False

class DatasetOutput(BaseModel):
    dataset: object
    n_samples: int
    modalities: frozenset[Modality]
    split: Split
    outer_fold: int       # passthrough for sampler
    inner_fold: int

class DataloaderConfig(BaseModel):
    dataset: DatasetOutput  # ← stage N output IS a field
    batch_size: int = 32
    num_workers: int = 4
```

**Properties:**

- Passthrough args (e.g. `outer_fold`) visible in schema — not buried in dicts
- Missing field that downstream needs → type error at construction, not runtime `KeyError`
- Builder functions in `src/` are the only place wiring logic lives
- Config files specify values only

---

### Pattern 2: Hydra-Zen over Hydra YAML

**Problem with Hydra YAML:**

- `_target_:` is a string → rename class = silent breakage
- `${hooke1.size.hidden_dim}` resolved at runtime → no IDE, no type checking
- `activation: banana` accepted until instantiation
- Import paths in YAML → refactoring tools don't see them

**Hydra-Zen equivalent:**

```python
from hydra_zen import builds, store
from hooke.models.adapters.mlp_adapter import MlpAdapter

MlpAdapterConfig = builds(
    MlpAdapter,                    # real import — tracked by IDE
    input_dim=256,
    output_dim=576,                # explicit, not ${interpolation}
    hidden_dim=512,
    activation=Activation.GELU,    # enum — invalid values rejected
    populate_full_signature=True,
)
store(MlpAdapterConfig, group="hooke1/encoders", name="mol_molgps_v2")
```

**Comparison:**

| Dimension | Hydra YAML | Hydra-Zen |
| --- | --- | --- |
| Import path | `_target_:` string | Real Python import |
| Type safety | None | Full (enum, int, Path) |
| IDE support | None | Autocomplete, go-to-def, rename |
| Refactoring | Rename class → silent breakage | Rename class → IDE updates |
| Cross-references | `${interpolation}` strings | Python attributes |

**Migration:** Hydra-Zen and YAML configs coexist in the same config store. Migrate one group at a time.

---

### Pattern 3: Registry Pattern Replaces YAML Interpolation

**Problem:**

```yaml
encoder_adapters:
  tx: ${hooke1.encoders.tx}
  phx: ${hooke1.encoders.phx}
  mol: ${hooke1.encoders.mol}
```

- Adding a modality → edit YAML in 2+ places
- Interpolation strings are untyped
- Wiring logic in config files instead of Python

**Solution:** Modality specs own their adapter configs. Top-level config lists active modalities. Python resolves the rest.

```python
MODALITY_REGISTRY: dict[Modality, ModalitySpec] = {
    Modality.TX: ModalitySpec(
        embedding_dim=39_754,
        special_token="<TX_CTRL_EXP>",
        adapter=AdapterSpec(input_dim=39_754, hidden_dim=1024, num_layers=3),
    ),
    Modality.MOL_MOLGPS_V2: ModalitySpec(
        embedding_dim=256,
        special_token="<MOL_MOLGPS_V2_EMB>",
        adapter=AdapterSpec(input_dim=256),
    ),
}
```

```python
@dataclass
class Hooke1Config:
    active_modalities: frozenset[Modality] = frozenset({Modality.TX, Modality.PHX, Modality.MOL})
    hidden_dim: int = 576
```

```python
def build_encoder_adapters(config: Hooke1Config) -> dict[Modality, MlpAdapter]:
    specs = get_specs(config.active_modalities)
    return {
        m: MlpAdapter(
            input_dim=spec.adapter.input_dim,
            output_dim=spec.adapter.output_dim(config.hidden_dim),
        )
        for m, spec in specs.items()
    }
```

**Result:** Ablation is a one-field CLI override:

```bash
uv run python scripts/train_hooke1.py active_modalities='[tx,phx,mol_molgps_v2]'
```

No YAML created or edited. Registry resolves names → specs → modules.

---

### Config Rules

1. **Config = WHAT, Python = HOW.** Config: `active_modalities: [tx, phx, mol]`. Python: knows TX needs a 3-layer adapter with 39,754 input dim.
2. **Config values = things a human changes between runs.** Batch size: config. TX gene count (39,754): registry. Facts in registry, choices in config.
3. **Schemas are pipe shapes.** `TaskGenOutput` → `HookeDatasetConfig.task` → `DatasetOutput` → `DataloaderConfig.dataset`. Each arrow is a typed field. Type-checks → fits.

---

## Dependency Policy

Five questions before adding any library.

### 1. Does it reduce code, or just relocate it?

- **Add:** `polars` replaces hundreds of lines of DataFrame logic you'd write worse
- **Skip:** config library that replaces 40 lines of dataclasses with 40 lines of decorators + a migration guide
- **Test:** delete the library, estimate replacement LOC. If the answer is "a small utility module the team understands" → skip

### 2. Usage fraction

- Using 3 functions from a 400-function library → smell
- Full dependency surface (version pins, transitive deps, security patches) for a sliver of value
- **Exception:** those 3 functions are genuinely hard to reimplement (crypto, GPU kernels, numerical optimisation)
- Otherwise → vendor the specific functions or reimplement

### 3. Maintenance risk

- Funded org or one maintainer's weekend project?
- Last release? Last commit?
- Open issues with "breaking change"?
- If maintainer ghosts → can you fork or replace in a day?
- Core deps (torch, polars, zarr): large ecosystem, fast fixes. Acceptable.
- Niche (<500 stars, one maintainer): you will own this code eventually. Vendor or rewrite.

### 4. Human cost

- **Learning curve:** does the team need to learn new idioms? `hydra-zen`: worth it (replaces something worse). Fancy logging wrapper: not worth it.
- **Environment friction:** adds time to `uv sync`? Conflicts with pins? Requires `apt install`?
- **Debugging opacity:** can the team debug it, or do they file GitHub issues and wait? Metaprogramming / code generation libraries are worst here.
- **Onboarding tax:** new member already learns PyTorch, Hydra, Hooke architecture, Recursion data. Every library multiplies this.
- **Rule of thumb:** if the "how to use X" doc is longer than the code X replaces → X is not pulling its weight

### 5. Critical path vs convenience path

- **Critical path** (training loop, data loading, forward pass, checkpointing): must be rock-solid, well-understood, hard to hand-roll. `torch`, `deepspeed`, `zarr` earn their place.
- **Convenience path** (progress bars, coloured output, CLI sugar): lightweight, unobtrusive. Must never block a release, require pin gymnastics, or appear in a stack trace.
- If a convenience dep appears in bug reports → it has failed its only job.

### Decision Matrix

| Signal | Add | Skip |
| --- | --- | --- |
| Reduces code by >100 LOC | ✓ | — |
| Hard to reimplement correctly | ✓ | — |
| Team already knows it | ✓ | — |
| Org-backed, active releases | ✓ | — |
| Using <10% of surface area | — | ✓ |
| Adds >1 min to env setup | — | ✓ |
| Requires tutorial doc for team | — | Think hard |
| One maintainer, stale commits | — | ✓ |
| Critical path, no escape hatch | — | ✓ (unless irreplaceable) |

### Hooke Examples

- **`hydra-zen`** → Add. Replaces fragile YAML with typed Python. Team already uses Hydra.
- **`polars`** → Add. Faster than pandas, stricter types. Replaces code we'd write badly.
- **W&B wrapper** → Skip. W&B API is already simple. Wrapper adds indirection, breaks on updates.
- **Dataclass-simplification library** → Skip. Dataclasses are already simple. Saving 3 LOC per class is not worth the dependency.

**Meta-principle:** A dependency should make the hard things possible, not make the easy things slightly more concise.

---

## Architecture Decision Records (ADRs)

**What:** Short document capturing a non-obvious design choice: what, why, consequences, alternatives rejected.

**Where:** `docs/decisions/` in repo.

**Why:** Code shows *what* was built. ADRs show *why* and *what was rejected*.

**Format:**

```
# ADR-003: Flow matching over DDPM for diffusion decoders

## Status
Accepted (2025-01-15)

## Context
Need diffusion decoder for TX and PHX. DDPM requires ~1000 sampling steps.
Flow matching allows fewer steps with comparable quality.

## Decision
Flow matching with linear interpolation. ODE integration t=0 → t=1.

## Consequences
- Faster inference (fewer steps)
- Simpler training (v_t = x_1 - x_0)
- Less community tooling
- Custom sampling infrastructure required

## Alternatives Rejected
- DDPM: sampling speed
- Score-based SDE: implementation complexity
```

**When to write:** Decision is non-obvious AND hard to reverse. Choosing between valid architectures, adopting a hard-to-remove dependency, changing a data format, deciding *not* to do something obvious.

**When not to write:** Obvious choices ("we use PyTorch"), trivial decisions, easily reversible changes.

**Time budget:** 10–15 minutes.
