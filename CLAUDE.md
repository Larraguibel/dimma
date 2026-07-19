# CLAUDE.md

Operational instructions for AI agents working on dimma.
Read `CONTEXT.md` first — it defines the domain language all code and prompts must use.

---

## Build and test

```bash
# Install in editable mode with all dev extras
pip install -e ".[dev,examples]"

# Run the full test suite
pytest

# Run a single test file
pytest tests/test_accounting.py -v

# Run with coverage
pytest --cov=dimma --cov-report=term-missing
```

## Project layout

```
src/dimma/          # library source (src/ layout, Hatchling backend)
tests/              # pytest suite; includes regression tests vs. reference impls
examples/private_spiderboost/    # Private SpiderBoost demos: notebooks/, lib/ (shared model.py + viz), scripts/, figs/
docs/
  adr/              # Architecture Decision Records — hard-to-reverse, non-obvious choices
  agents/           # Agent skill documentation (issue tracker, triage, domain)
  writing-guide.md  # Rules for writing mkdocs pages — read before editing mkdocs/
mkdocs/             # Published documentation site
  algorithms/       # One folder per implemented algorithm
specs/              # What to build: one file per topic of concern
```

## Documentation

**Before editing any file under `mkdocs/`, read `docs/writing-guide.md`.** It defines the rules for notation, page scope, admonitions, and where to put new content.

## Key conventions

- **Never pin the JAX version.** dimma declares only `python >= 3.10`. Users manage their own JAX/CUDA environment. Do not add `jax==x.y.z` to any requirements file.
- **`src/` layout is intentional.** Do not move modules to the repo root.
- **Architecture-agnostic by design.** `train()` accepts any JAX pytree for params and any per-sample loss function — the *algorithm* must not hard-code a specific architecture. This is **not** model-free: dimma ships reference models (Flax/optax are core deps) under `src/dimma/models/` so researchers have a testing model in hand. Model code belongs in `src/dimma/models/`, never in `src/dimma/algorithms/`. See `docs/adr/0002-thm-b3-config-resolver.md` for related structure.
- **Paper notation in the API.** Parameter names (`L0`, `L1`, `b1`, `b2`, `T`, `q`) follow Arora et al. 2023. Do not rename them to be "clearer".
- **Three noise scales, not one.** The algorithm requires `σ₁`, `σ₂`, `σ₂_hat`. Any change to noise injection must account for all three. See `CONTEXT.md → noise scale`.
- **Poisson subsampling is load-bearing for privacy.** The privacy accounting assumes i.i.d. Poisson inclusion. Do not substitute fixed-size batching without a corresponding accounting change.

## Agent skills

### Issue tracker
Issues live in GitHub Issues: `https://github.com/Larraguibel/dimma`.
See `docs/agents/issue-tracker.md`.

### Triage labels
`needs-triage` · `needs-info` · `ready-for-agent` · `ready-for-human` · `wontfix`
See `docs/agents/triage-labels.md`.

### Domain docs
Single-context, core + per-algorithm glossary layout — a universal `CONTEXT.md` at the repo root, per-algorithm glossaries under `docs/glossaries/<algo>.md`, plus `docs/adr/` for architectural decisions.
See `docs/agents/domain.md`.

## Architecture decisions

Before adding a new algorithm, privacy accountant, or sampling strategy, check `docs/adr/` for prior decisions. If your change is hard to reverse or non-obvious, write a new ADR.

---

## Learned rules

<!-- This section compounds. After every agent mistake, add one rule. Format: what not to do → why. -->

- Do not call `jax.grad` inside the training loop without `jax.vmap` — dimma uses per-sample gradients; batching without vmap silently computes the wrong thing.
