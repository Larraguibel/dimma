# Issue Tracker

Issues live at: `https://github.com/Larraguibel/dimma/issues`

## When to open an issue

Open an issue for:
- A bug with a reproducible test case or error traceback
- A feature that requires design discussion before implementation
- A privacy accounting question where the correct behavior is non-obvious
- A failing example notebook

Do not open an issue for work-in-progress experiments or notebook exploration that hasn't been reduced to a clear problem statement.

## Issue anatomy

```
Title: [component] short imperative description
       e.g. [accounting] compute_noise_scales raises on q > 0.5

Body:
- What you expected
- What happened (traceback or incorrect output)
- Minimal reproduction (code snippet or test)
- dimma version and JAX version
```

## Workflow states (via labels)

| Label | Meaning |
|---|---|
| `needs-triage` | Newly opened; not yet assessed |
| `needs-info` | Blocked on more information from the reporter |
| `ready-for-agent` | Scoped, reproducible, safe to hand to an AI agent |
| `ready-for-human` | Requires human judgment (privacy accounting, API design, paper interpretation) |
| `wontfix` | Intentionally out of scope; comment explains why |

## What makes an issue agent-ready

An issue is `ready-for-agent` when it has:
1. A clear expected vs. actual behavior
2. A failing test or reproduction script
3. No ambiguity about the correct mathematical behavior (if it's an accounting issue, a human must resolve the math first)

If the correct behavior requires interpreting the Arora et al. 2023 paper, label it `ready-for-human` instead.
