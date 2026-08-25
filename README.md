# Dynamic Multi-Model LLM Routing via Evolutionary Code Search

Executable research artifact: an island-model MAP-Elites genetic search that **evolves Python
routing policies** assigning each query to the cheapest capable LLM tier (`small` / `medium` /
`frontier`), validated by a deterministic 3-gate oracle (AST security → boundary smoke tests →
vectorized benchmark with a hard 100 µs latency penalty). No LLM-in-the-loop calls, no GPU,
full SQLite lineage, end-to-end reproducible in ~1 minute on a laptop CPU.

**Router design as program search.** Instead of training a supervised router on preference
data, we evolve executable `select_model(q)` functions over a synthetic 10,000-query corpus
with per-tier quality and token-price models ($0.15/$0.80/$3.00 per 1M input tokens across
tiers), then benchmark the champion against constant, random, hand-engineered, and
scikit-learn baselines, plus a budget-controlled 3-axis ablation (topology, archive,
operators).

---

## Core Results

```
┌────────────────────────────────────────────────────────────────────────┐
│  CHAMPION 22e2a1f5 (gen 10, island 2)                                  │
│                                                                        │
│  Cost reduction vs always-frontier ......... 81.4%   (OOD: 85.5%)      │
│  Mean decision latency ..................... 0.138 µs/query            │
│  Quality retention vs frontier ............. 82.0%  (Q = 0.5909)       │
│  Speedup vs decision-tree router ........... 489×  (67.6 µs)           │
│  Effective parameters ...................... 2 thresholds, 0.43 KB     │
│  Search budget ............................. 240 evals / ~5 s          │
└────────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ dataset.py                                                       │
│ 10,000 synthetic queries · difficulty-conditional quality model  │
│ small $0.15/$0.60 · medium $0.80/$3.20 · frontier $3.00/$15.00   │
└──────────────┬───────────────────────────────────────────────────┘
               │ train 6,000 · test 4,000 ──► OOD slice 832
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ mutator.py ◄── INITIAL_BASELINE_CODE                             │
│ <thought>/<code> extraction · 6 semantic operators ·             │
│ deterministic offline fallback                                   │
└──────────────┬───────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ oracle.py · deterministic 3-gate evaluation                      │
│  G1 AST security ► G2 boundary smoke ► G3 vectorized benchmark   │
│  F = 100·Q + 0.6·ΔC − 0.05·L − 300·𝟙[L > 100 µs]                 │
└──────────────▲───────────────────────────────────────────────────┘
               │ fitness · Q · ΔC · latency
┌──────────────┴───────────────────────────────────────────────────┐
│ engine.py                                                        │
│ Island model (3 × 14) · cyclic migration every 2 generations     │
│ MAP-Elites 6×6 grid (Q × ΔC) · SQLite lineage · Pareto frontier  │
└──────────────┬───────────────────────────────────────────────────┘
               ▼
  run.py ──► evolution_search.db ──► champion 22e2a1f5
               │
               ├── benchmark_comparison.py ──► Table 1 · benchmark_results.json
               ├── ablation.py             ──► Table 2 · ablation_table.json
               ├── visualize.py            ──► pareto_frontier.png · fitness_trajectory.png
               ├── inspect_champion.py     ──► top-3 lineage + OOD metrics
               └── generate_paper.py       ──► paper.tex · PAPER.md
```

## Table 1 · Main Baseline Comparison

| Method | Q (ID) | ΔC% (ID) | Q (OOD) | ΔC% (OOD) | L (µs) | Params | Mem (KB) |
|---|---|---|---|---|---|---|---|
| Frontier | 0.7208 | 0.00 | 0.7763 | −32.74 | 0.100 | 1 | 0.23 |
| Medium | 0.6072 | 77.56 | 0.6544 | 69.69 | 0.097 | 1 | 0.23 |
| Small | 0.4490 | 95.79 | 0.4851 | 94.32 | 0.097 | 1 | 0.23 |
| Random | 0.5888 | 58.82 | 0.6266 | 46.79 | 0.419 | 0 | 0.32 |
| Heuristic | 0.5512 | 82.32 | 0.6661 | 60.85 | 0.250 | 6 | 0.81 |
| Hand-Written | 0.5560 | 72.35 | 0.7292 | 32.77 | 0.174 | 5 | 0.85 |
| ML-Tree | 0.7203 | 0.47 | 0.7761 | −32.56 | 67.624 | 133 | 12.65 |
| **Evolved Champion** | **0.5909** | **81.36** | **0.5912** | **85.46** | **0.138** | **1** | **0.43** |

ID: train split (n=6,000); OOD: top-decile shift slice (n=832). Reference cost: always-frontier.
Latency = mean wall-clock `select_model(q)` call (±10% run-to-run). The supervised decision tree
collapses to always-frontier imitation — quality-optimal labels carry no cost pressure — while
costing 489× more inference than the champion for a strictly dominated operating point.

## Table 2 · Ablation Study

| Axis | Variant | Peak Fitness | Gen→95% | Gen>Base | Elites | Cells |
|---|---|---|---|---|---|---|
| A (topology) | Single population | 107.87 ± 0.36 | 1.0 ± 0.0 | 1.0 ± 0.0 | 220.0 ± 7.3 | 12.6 ± 0.9 |
| A (topology) | 3-island + migration (full) | 107.85 ± 0.30 | 1.0 ± 0.0 | 1.0 ± 0.0 | 223.2 ± 7.7 | 12.8 ± 0.5 |
| B (archive) | Greedy fitness replacement | 107.89 ± 0.26 | 1.0 ± 0.0 | 1.0 ± 0.0 | 222.6 ± 6.7 | 0.0 ± 0.0 |
| B (archive) | 2D MAP-Elites parents | 107.79 ± 0.34 | 1.0 ± 0.0 | 1.0 ± 0.0 | 222.6 ± 5.3 | 12.2 ± 0.8 |
| C (operators) | Random constant jitter | 107.72 ± 0.43 | 1.6 ± 1.3 | 1.0 ± 0.0 | 244.0 ± 0.0 | 10.6 ± 0.6 |
| C (operators) | Semantic thought-guided (full) | 107.85 ± 0.30 | 1.0 ± 0.0 | 1.0 ± 0.0 | 223.2 ± 7.7 | 12.8 ± 0.5 |

Budget-controlled (24 evals/generation), aggregated as mean ± std over 5 seeds (42, 123, 2026,
777, 999). Findings: mean peak fitness is statistically indistinguishable across all variants
(all means within 0.17, σ ≤ 0.43) — the framework, not the configuration, does the heavy lifting.
Seed-robust signals: **semantic mutation** is the only convergence-axis effect (Gen→95% 1.0±0.0
vs 1.6±1.3; +1.40 mean fitness at generation 1; tighter archive — 223±8 vs 244±0 elites, since
jitter-only offspring are all novel but none better); **islands** leave peak fitness neutral
(Δ=−0.03) but cut frontier-size variance by **65%** (σ 28.2 → 9.9); **MAP-Elites parents** are
fitness-equivalent (Δ=−0.10), their value being the inspectable quality-diversity grid itself.

## Champion Policy `22e2a1f5`

```python
def select_model(q):
    if not (not q["has_math_symbols"]):
        return "medium"
    if not (q["estimated_tokens"] > 367):
        return "medium"
    return "small"
```

Simplified semantics: **math → medium · non-math ≤ 367 tokens → medium · long non-math → small ·
frontier never used.**

- **Frontier elimination** — the 20× price premium is never paid (ΔC ≈ 81%).
- **Difficulty-aware escalation** — math queries carry the largest frontier–small quality gap;
  medium captures most of it at a quarter of the frontier price.
- **Difficulty inversion** — long non-math (chat/analysis) queries are easy, so bulk traffic
  rides the small tier almost free.
- The double negations are mutational scars of condition-inversion + comparison-flip lineage;
  threshold 367 is a `tweak_constants` child of its parent's 522-token threshold one generation
  after a branch rotation — neutral drift followed by exploitative refinement.

## Reproduction

```bash
pip install -r requirements.txt   # matplotlib, scikit-learn only
./reproduce.sh
```

| Step | Script | Output |
|---|---|---|
| 1. Discovery (10 gens × 3 islands) | `run.py` | `evolution_search.db`, live Pareto table |
| 2. Baselines vs champion | `benchmark_comparison.py` | Table 1, `benchmark_results.json`, `baseline_vs_champion.png` |
| 3. Ablations A/B/C | `ablation.py` | Table 2, `ablation_table.json` |
| 4. Figures | `visualize.py` | `pareto_frontier.png`, `fitness_trajectory.png` |
| 5. Paper | `generate_paper.py` | `paper.tex`, `PAPER.md` |

Requires Python ≥ 3.9 (stdlib-only core; matplotlib + scikit-learn for Phase 2+). Champion hash
`22e2a1f5` and all quality/cost metrics are deterministic (seed 42/2026); latency figures vary
±10% run-to-run.

## Repository Layout

```
dataset.py  oracle.py  mutator.py  engine.py  run.py     # core pipeline (stdlib only)
inspect_champion.py  baselines.py  benchmark_comparison.py
ablation.py  visualize.py  generate_paper.py
reproduce.sh  requirements.txt  LICENSE  README.md
paper.tex  PAPER.md  *.png  *.json  *.db                  # generated artifacts
```

## Author

**Thomas Gia Vy Day** — Independent Researcher
Contact: [giavytday@gmail.com](mailto:giavytday@gmail.com)

## Citation

```bibtex
@article{evorouting2026,
  title   = {Dynamic Multi-Model {LLM} Routing via Evolutionary Code Search},
  author  = {Day, Thomas Gia Vy},
  journal = {Preprint},
  year    = {2026},
  email   = {giavytday@gmail.com},
  note    = {Champion policy 22e2a1f5: 81.4\% cost reduction at 0.138\,$\mu$s decision latency}
}
```

## License

MIT — see [LICENSE](LICENSE).
