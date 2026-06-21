# Domain Documentation

## Layout

dimma uses a **core + per-algorithm glossary layout**: one universal glossary that
holds the language every DP-SGD variant shares, plus a small glossary per algorithm
for its own vocabulary.

| File | Purpose |
|---|---|
| `CONTEXT.md` | Universal glossary — the canonical definition of every term shared by all algorithms |
| `docs/glossaries/<algo>.md` | Per-algorithm glossary — terms specific to one algorithm (step names, noise-scale layout, hyperparameters, entry point) |
| `CLAUDE.md` | Operational instructions: build/test/run commands, conventions, learned rules |
| `docs/adr/` | Architecture Decision Records for hard-to-reverse, non-obvious choices |
| `specs/` | Feature specifications: what to build, one file per topic |

> **Why the split.** The project ships many algorithms over time, of which Private
> SpiderBoost is the first. A single flat glossary made it impossible to tell which
> terms were laws of DP and which were one paper's vocabulary. `CONTEXT.md` now holds
> only the shared language; each algorithm's terms live with that algorithm.

## How to use the glossaries

`CONTEXT.md` defines the universal language; `docs/glossaries/<algo>.md` defines an
algorithm's own terms. A glossary is the canonical naming *contract* — the pages
under `mkdocs/algorithms/` *explain* the terms but must not redefine them. When
writing code, comments, issues, or prompts:

- Use the term defined in the glossary, not a synonym
- The `_Avoid_:` list under each term names synonyms that have caused confusion — don't use them
- If you need a term that isn't defined, add it first — to `CONTEXT.md` if it is shared
  by all algorithms, otherwise to the relevant per-algorithm glossary
- When you implement a new algorithm, create `docs/glossaries/<name>.md` and add a row
  to the table in `CONTEXT.md → Algorithm-specific glossaries`

## How to use docs/adr/

An ADR is warranted when a decision is:
- Hard to reverse (changing the public API, switching accountants)
- Non-obvious (why we don't pin JAX, why we use Poisson not fixed-size batching)
- Likely to be re-questioned (by a future agent or contributor)

Template: `docs/adr/NNNN-short-title.md`

```markdown
# NNNN — Short title

## Status
Accepted

## Context
What problem are we solving and why does it arise?

## Decision
What did we decide?

## Consequences
What becomes easier? What becomes harder?
```

## How to use specs/

A spec file answers: *what does this component do?* Not how — that's the code.

Each spec is a complete vertical slice: it describes behavior from the user's perspective, references the relevant `CONTEXT.md` terms, and lists acceptance criteria that map to tests.

One spec per topic of concern. The title should be answerable in one sentence without the word "and".

## Updating domain docs

- **CONTEXT.md**: update when a *shared* term is renamed, a new cross-algorithm abstraction is introduced, or an `_Avoid_` entry proves wrong. Never add implementation details, and never add a term that belongs to a single algorithm — that goes in `docs/glossaries/<algo>.md`.
- **docs/glossaries/<algo>.md**: update when an algorithm's own vocabulary changes. If a term that started algorithm-specific turns out to be shared by a second algorithm, promote it to `CONTEXT.md` and remove it from the per-algorithm glossaries.
- **CLAUDE.md → Learned rules**: add one rule after every agent mistake. Format: what not to do → why.
- **ADRs**: immutable after acceptance. If a decision is reversed, write a new ADR that supersedes the old one; do not edit the original.
