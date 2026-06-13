# Domain Documentation

## Layout

dimma uses a **single-context layout**:

| File | Purpose |
|---|---|
| `CONTEXT.md` | Ubiquitous language glossary — the canonical definition of every domain term |
| `CLAUDE.md` | Operational instructions: build/test/run commands, conventions, learned rules |
| `docs/adr/` | Architecture Decision Records for hard-to-reverse, non-obvious choices |
| `specs/` | Feature specifications: what to build, one file per topic |

## How to use CONTEXT.md

`CONTEXT.md` defines the language of the domain. When writing code, comments, issues, or prompts:

- Use the term defined in `CONTEXT.md`, not a synonym
- The `_Avoid_:` list under each term names synonyms that have caused confusion — don't use them
- If you need a term that isn't in `CONTEXT.md`, add it before implementing

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

- **CONTEXT.md**: update when a term is renamed, a new abstraction is introduced, or an `_Avoid_` entry proves wrong. Never add implementation details.
- **CLAUDE.md → Learned rules**: add one rule after every agent mistake. Format: what not to do → why.
- **ADRs**: immutable after acceptance. If a decision is reversed, write a new ADR that supersedes the old one; do not edit the original.
