# Launch Announcement Assets

Canonical metrics (from `benchmark_results.json`, seed 42/2026): **81.4% cost reduction** vs
always-frontier routing · **0.138 µs** mean decision latency (0.13–0.14 µs across reruns) ·
**489× faster** than a decision-tree router · Q = 0.5909 (82.0% of frontier quality) · OOD cost
reduction rises to **85.5%** · champion `22e2a1f5`, 12 lines of Python, frontier tier never called.

---

## 1. X / Twitter post (single shot, <280 chars)

> We evolved an LLM router in 12 lines of Python.
>
> 81.4% cost cut vs always-frontier. 0.138µs per decision — 490× faster than a decision-tree router. Under distribution shift, savings RISE to 85.5%.
>
> It never calls the frontier tier. No embeddings. No GPU. A 3-gate oracle + MAP-Elites over code.
>
> — Thomas Gia Vy Day (@giavytday · giavytday@gmail.com)

## 2. LinkedIn post

> **We replaced our learned LLM router with a 12-line Python function discovered by evolution.**
>
> The economics of LLM serving are brutal: frontier-tier output costs 20× more than small-tier
> output, yet most queries don't need it. The standard answer is a neural router over embeddings —
> which adds tens of milliseconds of latency and a model-serving cost center directly on the
> critical path of time-to-first-token.
>
> We tried the opposite abstraction: treat the router as a program and evolve it.
>
> — A deterministic 3-gate oracle (AST security → boundary smoke tests → vectorized benchmark
>   with a hard 100µs latency penalty) scores every candidate.
> — A 3-island MAP-Elites search mutates routing code: threshold jitter, comparison flips,
>   branch rotation, condition inversion, grammar synthesis, crossover.
> — Every evaluation, parent link, and mutation "thought" is persisted to SQLite. Full lineage,
>   fully replayable, ~5 seconds per search on a laptop CPU.
>
> **Results (10,000-query benchmark, seed-controlled):**
> → 81.4% cost reduction vs always-frontier routing
> → 0.138µs mean decision latency — 489× faster than a decision-tree router of comparable quality
> → 82.0% of frontier quality retained; OOD shift *improves* savings to 85.5%
> → The champion never calls the frontier tier: math→medium, short→medium, long→small
>
> The supervised baseline is the cautionary tale: trained to imitate the quality-optimal tier, it
> collapses to near always-frontier behavior while costing 67.6µs per query — dominated on every axis.
>
> Paper (PDF + LaTeX), figures, ablation suite, and one-command reproduction (`./reproduce.sh`)
> are MIT-licensed in the repo. Comments and critique welcome — especially on the live-trace
> validation protocol in Section 6.
>
> — Thomas Gia Vy Day · Independent Researcher · giavytday@gmail.com

## 3. Blog post (~500 words) — Hugging Face Papers / Substack format

### We Evolved an LLM Router in a Dozen Lines of Python. It Cut Inference Cost 81.4%.

Every team serving LLMs at scale faces the same economic bug: a frontier model with a 20× price
premium answering queries that a small model could handle trivially. The standard fix is routing —
send each query to the cheapest tier that preserves quality. The standard recipe is a learned
classifier over neural embeddings. We think that is the wrong abstraction, and we built a fully
reproducible experiment to test the alternative: evolve the router as code.

Our setup is deliberately austere. A synthetic corpus of 10,000 queries spans chat, code, math,
and analysis tasks, each annotated with per-tier quality scores and token counts priced at
$0.15/$0.80/$3.00 per million input tokens across small/medium/frontier tiers. A candidate router
is a pure Python function, `select_model(q)`, mapping query features to a tier. Before earning a
benchmark run, it must survive a three-gate deterministic oracle: an AST security pass rejecting
imports, `exec`, and dangerous builtins; eight boundary smoke tests; and a vectorized benchmark
computing relative quality Q, cost reduction ΔC against always-frontier routing, and per-decision
latency — with a hard fitness penalty above 100 microseconds.

Search is an island-model genetic algorithm: three populations of 14 policies, tournament
selection, cyclic migration every two generations, and a 6×6 MAP-Elites grid archiving the fittest
policy per (quality × cost-reduction) niche so diversity never collapses. Mutation is semantic —
threshold jitter, comparison flips, branch rotation, condition inversion, grammar-guided synthesis,
two-parent crossover — each wrapped in a simulated `<thought>`/`<code>` exchange with a
deterministic offline fallback. Every evaluation and parent link is persisted to SQLite; the whole
240-evaluation search runs in about five seconds on a laptop CPU.

The result is embarrassingly simple. The champion, found in generation ten, is twelve lines long
and never calls the frontier tier: math-symbol queries go to medium, non-math queries up to 367
tokens go to medium, everything longer goes to small. That structure retains 82.0% of frontier
quality at 18.6% of frontier cost — an **81.4% cost reduction** — and under distribution shift the
savings *rise* to 85.5%. Decision latency is **0.138 microseconds**, about 490× faster than the
scikit-learn decision-tree router we trained for comparison. That baseline is the cautionary tale:
trained to imitate the quality-optimal tier, it collapses to near always-frontier behavior (0.5%
cost reduction) at 67.6µs per query — dominated on every axis.

We replicated every ablation over five seeds. The honest headline: mean peak fitness is
statistically indistinguishable across topology, archive, and operator variants (all means within
0.17 fitness) — the framework, not the configuration, does the heavy lifting. The seed-robust
signals: semantic mutation is the only convergence effect (95%-of-final in 1.0±0.0 generations vs
1.6±1.3 for random jitter), and island topology cuts frontier-size variance by 65%.

The caveats are real: the quality model is synthetic, latency measures the router rather than the
model, and the corpus is a single synthetic draw. But the pipeline — oracle, search, lineage,
figures, even the LaTeX paper — regenerates from one command (`./reproduce.sh`), which is exactly
what you want before spending money on live traces: replay production queries, recalibrate,
re-evolve in seconds, canary at 1–5% of traffic.

Code, paper, and full lineage database are MIT-licensed. Next time someone proposes a neural
router, ask a simple question first: what would evolution find?

*— Thomas Gia Vy Day (Independent Researcher) · giavytday@gmail.com*
