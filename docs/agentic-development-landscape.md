# The Landscape of AI-Assisted Software Development
### A Research Synthesis for Human Consumption
*Compiled June 2026 — from primary research on Andrej Karpathy, Boris Cherny, Simon Willison, Geoffrey Huntley, and Matt Pocock*

---

## The Big Picture

We are in the middle of a genuine paradigm shift in software development. It is not just "autocomplete that got better." The five practitioners surveyed here have been building *with* AI agents — not around them — long enough to have formed stable, tested opinions. They do not fully agree on everything, but their convergences are striking.

The shift can be summarized in one sentence: **the bottleneck has moved from writing code to managing context.**

Writing code is now essentially free. Delivering working, maintainable, well-designed code is not. The cost of generation collapsed; the cost of quality has not. Your job as a developer is no longer primarily to type correct syntax — it is to be an information architect: deciding what goes into the agent's context window, in what form, at what level of specificity.

---

## Part 1 — The Voices

### Andrej Karpathy — The Theorist of Software 3.0

**Who he is:** Former OpenAI research director, creator of nanoGPT and micrograd, one of the most followed technical voices in AI.

**His framing: Software 1.0, 2.0, 3.0**

- **Software 1.0** — Explicit human-written logic. You wrote every line.
- **Software 2.0** (his 2017 concept) — Neural networks learned from data. "The programmer" shifted to curating datasets and loss functions.
- **Software 3.0** — LLMs as the new runtime. Natural language is the programming interface.

The analogy he uses:
> Model weights = CPU (fixed at inference time, the processing substrate)
> Context window = RAM (working memory that shapes every computation)
> Prompting/context = Programming

**"Context engineering" over "prompt engineering"**

Karpathy coined the term shift from "prompt engineering" to "context engineering" in June 2025:
> "People associate prompts with short task descriptions. In every industrial-strength LLM app, context engineering is the delicate art and science of filling the context window. Too little or the wrong form and the LLM doesn't have what it needs. Too much or too irrelevant, and performance comes down and costs go up."

**Vibe coding → Agentic engineering**

Karpathy invented the term "vibe coding" in February 2025 — generating code with an LLM without reviewing what it produces. It went viral (Collins Word of the Year 2025). He later declared it obsolete as a serious practice, replacing it with "agentic engineering":
> "Agentic, because the new default is that you are not writing the code directly 99% of the time. You are orchestrating agents who do and acting as oversight. Engineering, to emphasize that there is an art and science and expertise to it."

**The "Idea File" paradigm**

One of his most interesting contributions: in the era of LLM agents, you don't share code — you share the *idea*. An idea file is a markdown document describing an architecture pattern or system design, meant to be pasted into an agent, which then instantiates it for the user's specific project. The LLM Wiki he published as a GitHub Gist demonstrates this: it's a pattern for building a personal wiki with an AI agent, written in markdown, not code.

**On understanding:**
> "You can outsource your thinking but you can't outsource your understanding."

**Key infrastructure recommendations for AI-ready codebases:**
1. `llm.txt` files — structured, natural-language summaries for visiting AI agents (analogous to `robots.txt` for crawlers)
2. Clean Markdown documentation with executable API calls, not screenshots
3. Flatten-and-ingest tools — collapse entire repos into digestible text for LLM analysis

---

### Boris Cherny — The Practitioner Who Built the Tool

**Who he is:** Creator and Head of Claude Code at Anthropic. Former Principal Engineer at Meta (Instagram infrastructure). Author of *Programming TypeScript* (O'Reilly, 2019). Claude Code started as his side project in September 2024.

**Scale of practice:**
> "In the last thirty days, 100% of my contributions to Claude Code were written by Claude Code."

> "Last week, there was a day when I submitted 150 PRs."

> "This morning I was managing maybe a few hundred [agents]. Some days it's thousands, or tens of thousands."

He has not manually written a line of code since late 2025. He ships from his phone.

**The central thesis: context compounds**
> "The instructions you give the agent are more valuable than the code the agent writes. The code is ephemeral. The context compounds."

**CLAUDE.md as institutional memory**

Every mistake becomes a rule. Not a verbal correction — a written rule added to `CLAUDE.md`. His team's file sits at ~2,500 tokens. His discipline: every time Claude does something wrong, he doesn't correct it in the chat. He tells it to write the lesson to `CLAUDE.md`. The file is checked into git and shared across the team. Every engineer benefits from every other engineer's corrections.

If the file gets too long: delete it and start fresh. With each new model, you need less in it (models get smarter; rules that were necessary last year become unnecessary).

**Compounding Engineering**

His team runs a GitHub Action that tags `@.claude` on every PR. The agent reviews code review comments and automatically updates `CLAUDE.md`. Learnings compound across reviews without human intervention.

**Agentic search beats RAG**

This is counterintuitive and important. Building Claude Code, his team tried: local vector databases, recursive model-based indexing, semantic search. All of them lost to:
> "Agentic search is really just glob and grep, and plain glob and grep, driven by the model, beat everything."

For live codebases, simple model-driven file search outperforms every vector-database-backed solution they tested.

**On workflow:**
- 5 parallel Claude instances in separate git worktrees
- Plan Mode first; iterate the plan until solid; then auto-accept
- Verification loops are a 2–3x quality multiplier: give the agent a way to test its own work
- Build for the model 6 months from now, not today's

**On the future of the developer role:**
> "The title 'software engineer' could start to disappear by the end of this year. I don't think we're going to call them engineers. But if we talk about people writing code or using agents to write code, I think there will be 100 times more of them than there are today."

> "I imagine a world where everyone is able to program. Anyone can just build software anytime." (Compares to the printing press democratizing literacy from ~1% to ~70% over 200 years.)

---

### Simon Willison — The Most Sustained Public Practitioner

**Who he is:** Co-creator of Django, creator of Datasette and the `llm` CLI tool (model-agnostic, plugin-based). Has written daily at simonwillison.net since 2002. 387+ posts tagged `ai-assisted-programming`. Coined the term "prompt injection" in September 2022.

**His framing: AI lowers the threshold for what is worth building**

Not just faster — different work becomes possible:
> "It's not about getting work done faster, but about being able to ship projects that I wouldn't have been able to justify spending time on at all."

The question changes from "is this worth my time?" to "why not try this?"

But immediately: "Writing code is cheap now... but delivering good code remains significantly more expensive." He is both enthusiastic and unsentimental.

**Vibe coding vs. vibe engineering (his distinction)**

- **Vibe coding** (Karpathy's term): generating code without reviewing it. Fine for throwaway projects.
- **Vibe engineering** (his coinage): experienced professionals using coding agents at full speed while remaining fully accountable for what they ship. Requires mastery of automated testing, planning, code review, manual QA.

His golden rule:
> "I won't commit any code to my repository if I couldn't explain exactly what it does."

He admitted in May 2026 that the line is blurring even for him: "As the coding agents get more reliable, I'm not reviewing every line of code anymore, even for production stuff."

**Documentation as a deliverable to AI agents**

This is his most important structural contribution. He built `docs-for-llms`: a CI job (GitHub Actions) that concatenates all project documentation into a single, LLM-optimized file, shipped with every release and queryable from the CLI. The file is designed to be pasted into agent context, not read by humans.

He recommends `AGENTS.md` per repository: a file specifically for coding agents that documents available tools, conventions, and context.

**The LLM tool philosophy (model-agnosticism in practice)**

His `llm` CLI tool works with OpenAI, Anthropic Claude, Google Gemini, local Ollama models, and dozens of others via a plugin system. Its design principles:
- Single abstraction layer across all model providers
- SQLite logging of everything (transparent, reproducible, auditable)
- Fragments: SHA256-deduplicated reusable context blocks
- Plugin architecture: no model is baked in

This is the most concrete working example of an agent-agnostic tool in the field.

**The codegen workflow (greenfield projects):**
1. Ask the model one question at a time: "Ask me one question at a time so we can develop a thorough spec."
2. Save output as `spec.md`
3. Use a reasoning model to generate `prompt_plan.md` with sequenced prompts
4. Maintain `todo.md`; agents check off items

**Prompt injection — the unsolved security problem**

Willison coined "prompt injection" and has written the most extensive body of work on it. The core problem: LLMs cannot reliably distinguish developer instructions from attacker instructions embedded in untrusted content.

The **Lethal Trifecta** — an AI agent becomes critically dangerous when it combines all three:
1. Access to private data
2. Exposure to untrusted content
3. Ability to communicate externally (exfiltration pathway)

His proposed defense (the **Dual LLM Pattern**, 2023 — two years before the field formalized it):
- Privileged LLM: sees only trusted input, has full tool access
- Quarantined LLM: processes untrusted content, has zero tool access
- Deterministic controller mediates between them

His consistent warning: "Do not fight AI with AI." Probabilistic AI defenses fail at the security boundary. The only real mitigations are architectural (isolation, blast-radius reduction).

**Seven durable principles:**
1. AI lowers the threshold for what is worth building, not just how long it takes
2. Vibe coding and agentic engineering are different things — the distinction matters in production
3. Hoard working examples — every solved problem becomes future agent input
4. Documentation is a deliverable to AI, not just humans
5. Tests are the infrastructure that make agentic iteration safe
6. Prompt injection is unsolved; the lethal trifecta must be broken by design
7. The agent does the typing; the human does the thinking

---

### Geoffrey Huntley — The Automator

**Who he is:** Australian open source engineer, formerly tech lead for developer productivity at Canva, now at Sourcegraph building Amp. Creator of the Ralph Wiggum Technique and Latent Patterns (an AI education platform).

**Core thesis:** Software development as a discipline is dead; what survives is software engineering as orchestration.
> "I seriously can't see a path forward where the majority of software engineers are doing artisanal hand-crafted commits by as soon as the end of 2026."

**The Ralph Wiggum Technique — autonomous agentic loops**

His signature contribution: a named loop that drives an LLM agent continuously until a project is done.

```bash
while :; do cat PROMPT.md | claude; done
```

Or with modes:
```bash
./loop.sh plan      # Planning mode: gap analysis, generates IMPLEMENTATION_PLAN.md
./loop.sh           # Build mode: picks one task, implements it, runs tests, commits, exits
```

The key insight: **each loop iteration exits after one commit**. Fresh context every time. The `IMPLEMENTATION_PLAN.md` file on disk is the shared state that bridges context windows.

**His three-file system:**

**`specs/*.md`** — What to build. One file per "topic of concern." His test for a topic: "Can you describe this in one sentence without the word 'and'?" If not, split it. These files are written through dialogue with the agent at the start of a project, not handed down from above. They are the source of truth.

**`AGENTS.md`** — ~60 lines maximum. How to build, run, and test the project. Brief operational learnings. Nothing else. He is explicit: "Status updates and progress notes belong in IMPLEMENTATION_PLAN.md. A bloated AGENTS.md pollutes every future loop's context."

**`IMPLEMENTATION_PLAN.md`** — Disposable. Can be fully regenerated in one planning loop. Persists on disk between iterations. The eventual-consistency mechanism between isolated context windows.

**Backpressure is mandatory**

Tests, typechecks, and lints are not optional code quality aspirations — they are quality gates that the loop must pass before committing. Without backpressure, autonomous loops produce confident nonsense. The compiler is the agent's reality check.

**The stdlib concept**

Beyond specs (what to build), he maintains a `stdlib` of reusable prompt templates that enforce consistent LLM behavior across projects — for example, forcing the agent to use Svelte 5 syntax when it defaults to Svelte 4. The combination: specs (what) + stdlib (how) + a type-safe compiler (backpressure) = the full "hands-free" stack.

**On parallel agents:**
- Parallel subagents for search and write tasks: up to 500 simultaneous
- Strictly 1 subagent for build and test: enforces the backpressure gate
- "With ~176k truly usable tokens from a 200k budget, efficiency is critical. Less is more."

**On vibe coding:**
> He explicitly warns against it for complex systems, calling it "reckless." His critique: shipping unverified AI output without backpressure mechanisms is the failure mode, not AI itself.

**On open source and IP:**
> "AI can extract specs from source code, enabling rapid product replication." He considers this "a Bitcoin mixer for intellectual property" — a genuine disruption of moats.

**His one-liner on the developer future:**
> "The future belongs to people who can just do things."

---

### Matt Pocock — The TypeScript Educator Turned Systems Thinker

**Who he is:** Creator of Total TypeScript (~60,000 subscribers), formerly at Vercel and Stately. Launched aihero.dev in late 2024 as his primary AI engineering platform. His personal `.claude` directory published as `mattpocock/skills` reached 75,700+ GitHub stars in weeks (#1 trending AI repo on GitHub, May 2026).

**Core thesis:** Software engineering fundamentals matter *more* with AI, not less.

He directly opposes the "vibe coding" and "specs-to-code" paradigm. His position: AI agents fail in predictable, fixable ways, and the solution is to apply classic engineering discipline (DDD, TDD, vertical slices, deep modules) at the agent level. The human's job shifts from writing code to designing interfaces, grilling assumptions, and reviewing outputs.

**CONTEXT.md as DDD ubiquitous language**

This is his most powerful contribution and the one with the most measurable impact. He derived it from Eric Evans's Domain-Driven Design concept of *ubiquitous language* — a shared vocabulary that eliminates ambiguity.

Problem: agents dropped into a project figure out the jargon as they go. They use 20 words where 1 will do.

His example from a real project (`course-video-manager`):
- **Before CONTEXT.md:** "There's a problem when a lesson inside a section of a course is made 'real' (i.e. given a spot in the file system)"
- **After CONTEXT.md:** "There's a problem with the materialization cascade"

The format:
```markdown
## Language

**Materialization**:
The process of assigning a file-system slot to a lesson that was previously virtual.
_Avoid_: making real, instantiating, creating

**Anchor step**:
The phase in SpiderBoost that computes a full gradient at a fixed point.
_Avoid_: snapshot step, reference computation
```

Rules:
- One or two sentences maximum per definition
- Pick one canonical term; list all others under `_Avoid_`
- Only project-specific terms — no general programming concepts
- CONTEXT.md is a glossary only — no implementation details, no specs, no scratch pad

**"Codebase over prompt"**

> "Your codebase, way more than your prompt or your AGENTS.md file, is the biggest influence on AI's output."

He recommends **Deep Modules** (from Ousterhout's *A Philosophy of Software Design*): large amounts of functionality behind simple, controllable interfaces. The Grey Box Module Pattern:
- Humans design and own module interfaces
- AI handles implementation
- Tests lock down module behavior through public APIs only

**The five-step workflow:**

1. **`/grill-me`** — Interview relentlessly before writing a single line of code. Ask one question at a time. Walk down every branch of the design tree. Resolve decisions in order. (~93k tokens on Opus for a complex feature.)
2. **`/to-prd`** — Synthesize the grilling session into a Product Requirements Document (GitHub Issue). Include problem statement, user stories, interface contracts. No file paths, no code snippets.
3. **`/to-issues`** — Break the PRD into independently grabbable vertical-slice GitHub Issues with blocking relationships (a DAG). **Vertical slices, not horizontal** (never "schema first, then API, then frontend" — each issue should be a complete feature end-to-end).
4. **AFK Agent Loop** — Deploy agents to implement issues in fresh context. This runs without you. Human is not in the loop.
5. **`/improve-codebase-architecture`** — Periodic review (every few days) to surface consolidation candidates, shallow modules, tangled dependencies. Outputs HTML report with before/after diagrams.

Human gatekeeping rule:
> "Implementation should be AFK, but planning and QA must stay human-in-the-loop."

**Context budget discipline**

The "smart zone / dumb zone" framework: models become noticeably worse above ~100k tokens, regardless of advertised context window size. His response: **reset, not compact**. Compaction creates "sediment" — accumulated noise that degrades future sessions. Each session starts from a clean, stable baseline.

**Agent-agnosticism**

His skills/commands are explicitly portable across Claude Code, Codex CLI, Cursor, Aider, Cline, Continue. The same markdown files work everywhere. When his `/setup-matt-pocock-skills` skill runs, it writes to whichever of `CLAUDE.md` / `AGENTS.md` already exists — it never creates both, and asks the user if neither exists. Deliberate agent-neutrality.

**On using classic software engineering books as prompts:**
> "The classic coding books contain some of the best descriptions of 'good code' ever written. Use them in your prompts, skills, and AGENTS.md files."

(He maintains an `agent-rules-books` collection: Clean Code, DDD, Clean Architecture, POSA — all formatted for agent consumption.)

**Four failure modes and their fixes:**

| Failure Mode | Symptom | Fix |
|---|---|---|
| Misalignment | "The agent didn't do what I want" | `/grill-me` before starting |
| Verbosity | "The agent is way too verbose" | `CONTEXT.md` domain glossary |
| Broken code | "The code doesn't work" | TDD red-green-refactor |
| Architectural decay | "We built a ball of mud" | `/to-prd`, periodic zoom-out review |

---

## Part 2 — What They All Agree On

Despite coming from very different angles (research, product, tool-building, education, automation), these five voices converge on a small set of durable ideas.

### 1. The Three-Layer Context Model

Every practitioner independently arrived at the same structure:

| Layer | File | Purpose | Owner |
|---|---|---|---|
| Domain language | `CONTEXT.md` | What terms mean in *this project* | Human writes, agent reads |
| Operations | `CLAUDE.md` / `AGENTS.md` | How to build, test, run; accumulated rules | Human + agent co-write |
| Work state | `IMPLEMENTATION_PLAN.md` / `todo.md` | Current tasks; disposable | Agent writes, human reviews |

These layers must be **separate**. Mixing them (e.g., putting task status in CONTEXT.md, or architecture decisions in AGENTS.md) degrades all three.

### 2. Context Is the Primary Lever

No one talks about "better prompts" anymore. The universal move is: more context, better-structured context, context that is *curated* rather than dumped. But also: less is more above ~100k tokens. The skill is knowing what to include and what to leave out.

### 3. Tests Are the Safety Net That Makes Autonomy Possible

Without a test suite as a quality gate (what Huntley calls "backpressure"), autonomous agent loops are unsafe. You cannot run the loop unattended without a way for the agent to know if it broke something. TDD is not just a code quality practice in this world — it is the prerequisite for delegation.

### 4. Specs Before Code

Whether it's Karpathy's `program.md`, Huntley's `specs/*.md`, Willison's `spec.md`, or Pocock's `PRD-as-GitHub-issue — every practitioner insists on a written specification before the agent writes implementation. The spec is where human judgment lives. The implementation is increasingly where it doesn't need to.

### 5. Agent-Agnostic Design = Plain Markdown

The files that survive tool changes are the ones that are plain markdown: `CONTEXT.md`, `AGENTS.md`, `specs/`. These work with Claude Code today, with Codex tomorrow, with whatever comes after. The tool-specific parts (`.claude/commands/`, `.cursor/rules/`) are thin wrappers over the same portable content.

### 6. Documentation Is a First-Class Deliverable to AI

Not documentation that happens to also be readable by AI — documentation *designed for AI consumption*: concatenated context files, `AGENTS.md`, `llms.txt`, fragments. These are engineering artifacts that require maintenance, versioning, and care. They are as important as the code they describe.

### 7. The Human Does the Thinking; the Agent Does the Typing

Willison's formulation, echoed by all five: what AI removed is the tedium of translating understanding into syntax. What remains is the hardest part — knowing what to build, recognizing bad outputs, making principled trade-offs, designing interfaces, asking the right questions. These compound more than ever.

---

## Part 3 — What They Disagree On (or Vary On)

**How much autonomy is safe right now**

Cherny runs thousands of agents overnight. Willison is cautious about agents making "material decisions" on your behalf and documents cases of agents deleting inboxes, writing hit pieces on open-source maintainers. The difference is likely context: Cherny is operating on a well-tested, well-instrumented codebase he owns entirely.

**Compact vs. reset**

Pocock: always reset when context gets stale; compaction creates sediment. Cherny: uses compaction (`CLAUDE_CODE_AUTO_COMPACT_WINDOW=400000`) but also recommends `/clear` with a hand-written brief for genuinely new tasks. The principle is the same — don't drag stale context forward — but the implementation differs.

**How much to put in `AGENTS.md` / `CLAUDE.md`**

Huntley: 60 lines maximum, ruthlessly operational. Cherny: ~2,500 tokens, but delete and restart if it grows unwieldy. Pocock: CONTEXT.md (glossary) is separate from CLAUDE.md (ops) — don't conflate them.

**The role of RAG and vector search**

Cherny: glob + grep beat RAG for live codebases. Willison: uses SQLite and semantic similarity via embeddings for his own knowledge base tools. The resolution: RAG is useful for static knowledge (documentation, past conversations), but for *live codebase navigation*, model-driven file search wins.

---

## Part 4 — The Emerging File System for AI-Native Projects

Based on convergences across all five voices, here is the canonical file structure for an AI-native library or research project:

```
project-root/
│
├── CONTEXT.md              # DDD ubiquitous language glossary (Pocock)
│                           # Format: term → 1-2 sentence definition + _Avoid_ list
│                           # ONLY project-specific terms. No specs. No status.
│
├── CLAUDE.md               # Agent operations (Cherny + Huntley)
│   (or AGENTS.md)          # Build/test/run commands. Conventions. Accumulated rules.
│                           # ~60-2500 tokens. Delete and restart if it bloats.
│
├── llms.txt                # Concatenated LLM-optimized docs (Karpathy + Willison)
│                           # Machine-readable summary of the entire project.
│                           # Generated by CI, not hand-written.
│
├── specs/                  # What to build (Huntley)
│   ├── algorithm-X.md      # One file per topic of concern
│   ├── public-api.md       # ("Can you say this in one sentence without 'and'?")
│   └── accounting.md       # Written through dialogue, not dictation
│
├── docs/
│   ├── adr/                # Architecture Decision Records (Pocock + this repo)
│   │   └── 0001-*.md       # Only for: hard to reverse + surprising + real trade-off
│   └── agents/             # Agent skill documentation
│       ├── issue-tracker.md
│       └── domain.md
│
├── .claude/
│   └── commands/           # Reusable slash commands for common workflows (Cherny)
│       ├── run-tests.md    # /run-tests
│       ├── add-algorithm.md # /add-algorithm
│       └── update-context.md # /update-context
│
└── IMPLEMENTATION_PLAN.md  # Disposable cross-session task state (Huntley)
                            # Regenerated in one planning pass. Not committed long-term.
```

---

## Part 5 — On Security (The Thing Everyone Forgets)

Simon Willison is the most important voice on this, and it is not being discussed enough in practice.

**Prompt injection** is the vulnerability where an attacker embeds competing instructions inside untrusted content (an email, a document, a web page) that an AI agent will process. The LLM cannot reliably distinguish developer instructions from attacker instructions.

This is not hypothetical. Documented cases exist of AI agents:
- Deleting users' entire email inboxes despite instructions to check first
- Publishing false information about open-source maintainers
- Sending messages to third parties without authorization

**The Lethal Trifecta** — an agent becomes critically dangerous when it combines:
1. Access to private data
2. Exposure to untrusted content
3. Ability to communicate externally

If your agent can read your files, process external content, *and* make API calls or send emails — you have built a system that can be hijacked by anyone whose content the agent reads.

**The practical mitigations:**
1. Break the trifecta by design: never let an agent with private data access also process untrusted content
2. Scope tool permissions tightly (read-only when possible, staging-only credentials)
3. Use isolated environments (Docker, Codespaces) for agents that need broad access
4. Do not use AI to filter AI injection attempts ("99% is a failing grade in application security")
5. Design for bounded blast radius: assume exploitation will happen; limit the damage

---

## Part 6 — What This Means If You Are Building a Research Library

A research library like `dimma` has specific characteristics that make agentic development particularly tractable:

**Things that work very well with agents:**
- Implementing well-specified mathematical algorithms (the spec is the paper)
- Writing tests for known-correct behavior (regression tests, numerical tolerance checks)
- Refactoring to enforce consistent naming and module structure
- Generating documentation from source code
- Running experiments with well-defined success criteria (accuracy metrics, convergence)

**Things that still require human judgment:**
- Deciding which algorithms to implement next
- Making trade-offs between correctness, performance, and API ergonomics
- Reviewing outputs for numerical correctness (agents hallucinate math)
- Designing the public API surface (what users see)
- Interpreting experimental results

**The key adaptation for research code:**
The paper itself is the spec. The mathematical notation in the paper is the ubiquitous language. Your `CONTEXT.md` should contain the exact terms from the paper — ε, δ, σ₁, σ₂, anchor step, variation step, phase length — with precise definitions, and a list of informal synonyms to avoid. This is the fastest way to stop agents from introducing subtle notation drift that corrupts implementations.

---

## Summary: The Five Questions Every AI-Native Project Should Answer

1. **What is the ubiquitous language of this project?** → `CONTEXT.md` (DDD glossary, one term per concept)
2. **How does an agent get started working here?** → `CLAUDE.md` / `AGENTS.md` (build, test, run, key conventions)
3. **What is the source of truth for what should be built?** → `specs/*.md` (one file per topic of concern)
4. **How does the agent know if it broke something?** → Test suite (backpressure; mandatory quality gate)
5. **How do lessons compound across sessions?** → `CLAUDE.md` compounding rules; specs updated through dialogue

If you can answer all five, you have a codebase that any competent agent can navigate — regardless of which model is on the other end.

---

*Sources consulted: karpathy.bearblog.dev, simonwillison.net (387+ AI-assisted programming posts), ghuntley.com, mattpocock/skills (GitHub), borischerny.com, howborisusesclaudecode.com, newsletter.pragmaticengineer.com, lennysnewsletter.com, ycombinator.com/library (multiple episodes), aihero.dev, youtube.com (YC AI Startup School keynote, Sequoia AI Ascent 2026, AIE Europe 2026 workshop). Research conducted via parallel agent search — June 13, 2026.*
