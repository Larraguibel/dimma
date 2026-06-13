# Triage Labels

## Label vocabulary

| Label | When to apply |
|---|---|
| `needs-triage` | Default for all new issues. Remove once assessed. |
| `needs-info` | Issue is underspecified — missing reproduction, version info, or expected behavior. Comment explaining what is needed before removing. |
| `ready-for-agent` | Issue is scoped, reproducible, and the correct behavior is unambiguous. Safe to assign to an AI agent. |
| `ready-for-human` | Requires human judgment before implementation can begin. See below. |
| `wontfix` | Intentionally out of scope. Always add a comment explaining why. |

## What requires a human (`ready-for-human`)

Apply `ready-for-human` when the issue involves:

- **Privacy accounting correctness** — any question about whether `compute_noise_scales` produces valid (ε, δ) guarantees. The math must be reviewed against the paper before an agent touches it.
- **API design** — adding, renaming, or removing public symbols (`train`, `TrainConfig`, `NoiseScales`, etc.). These are hard to reverse once published.
- **Algorithm extensions** — adding a new DP optimizer. Requires verifying the convergence and privacy analysis before implementation.
- **Paper interpretation** — any ambiguity about what Arora et al. 2023 specifies.
- **License or attribution** — anything touching the Criteo dataset license or paper citation.

## Triage checklist

When triaging a new issue:

1. Is it reproducible? If not → `needs-info`
2. Is the correct behavior obvious from code alone? If not → `ready-for-human`
3. Does it touch privacy accounting or the public API? → `ready-for-human`
4. Otherwise → `ready-for-agent`
5. Is it clearly out of scope (general JAX question, unrelated ML task)? → `wontfix` with explanation
