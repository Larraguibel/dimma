# MkDocs writing guide

Rules for adding or editing pages under `mkdocs/`. Agents must read this before touching any file in `mkdocs/`.

---

## 1. Define every symbol on first use

Any letter or abbreviation that refers to an algorithm parameter, mathematical quantity, or code variable must be defined the first time it appears on a page — even if it seems obvious from context. Use a short parenthetical:

> `b1` (the anchor-step batch size)
> `q` (the anchor frequency — an anchor step fires every `q` steps)
> `F0 = F(w0; S) − min_w F(w; S)`, the initial suboptimality

Do **not** assume the reader has just read the paper or another page. Each page must be interpretable on its own.

This rule applies to:
- Algorithm parameters (`b1`, `b2`, `T`, `q`, `L0`, `L1`, `σ₁`, `σ₂`, `σ̂₂`, `c`, `δ`, `ε`, `d`, `n`, `F0`, …)
- Code identifiers used in prose (`params_random`, `verify_epsilon`, `control_rng`, …)
- Theorem or section references (`Theorem B.3`, `Algorithm 2`, …)

**Do not over-specify.** One phrase is enough. The goal is to give the reader a foothold, not to duplicate the paper.

---

## 2. One concern per page

Each page should answer one question. Examples of correctly scoped pages:

- *Differences between theory and implementation* — what the code does that the paper doesn't guarantee
- *The q-invariance of params_random* — one specific algorithmic property and its consequences

If a page starts needing more than one `##` section that aren't sub-parts of the same question, consider splitting it.

---

## 3. Use admonitions for status

Use MkDocs admonitions to signal the epistemic status of a claim:

```markdown
!!! warning
    **Status: open.** This has not been resolved.

!!! note
    Background context the reader needs but that isn't the main point.
```

Do not bury open questions in prose where they look like settled facts.

---

## 4. Where to put new content

```
mkdocs/
  index.md                    # landing page — update the "Structure" section when adding pages
  algorithms/
    index.md                  # one-row-per-algorithm table — add a row when implementing a new algorithm
    spiderboost/              # one folder per algorithm
      index.md                # algorithm overview: what it does in prose, entry point in code
      theory-vs-implementation.md
      implementation-notes.md
      <other sub-pages>
    <new-algorithm>/          # same structure
```

When adding a new algorithm:
1. Create the algorithm's glossary at `docs/glossaries/<name>.md` (per `docs/agents/domain.md`) and add a row to the table in `CONTEXT.md → Algorithm-specific glossaries`. The narrative pages below *explain* those terms; they do not redefine them.
2. Create `mkdocs/algorithms/<name>/index.md` with a one-sentence description and the `dimma` entry point.
3. Add a row to `mkdocs/algorithms/index.md`.
4. Add the algorithm to the `nav` in `mkdocs.yml` under the `Algorithms` section.
5. Update `mkdocs/index.md` → "Structure of this section" to mention it.

---

## 5. Language

All pages are in English.

---

## 6. Links

Use relative links between pages (e.g., `../library.md`). Do not hardcode `https://larraguibel.github.io/dimma/` URLs inside the docs themselves.
