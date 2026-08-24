#!/usr/bin/env bash
# Full reproduction pipeline: discovery -> baselines -> ablations -> figures -> paper.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python

banner() { echo; echo "=============================================================="; echo " $1"; echo "=============================================================="; }

banner "STEP 1/5 · Discovery search (run.py)"
"$PY" run.py

banner "STEP 2/5 · Baseline comparison (benchmark_comparison.py)"
"$PY" benchmark_comparison.py

banner "STEP 3/5 · Ablation suite (ablation.py)"
"$PY" ablation.py

banner "STEP 4/5 · Figures (visualize.py)"
"$PY" visualize.py

banner "STEP 5/5 · Paper compilation (generate_paper.py)"
"$PY" generate_paper.py

banner "REPRODUCTION COMPLETE"
echo " artifacts:"
ls -la evolution_search.db ablation_*.db benchmark_results.json ablation_table.json \
       pareto_frontier.png fitness_trajectory.png baseline_vs_champion.png \
       paper.tex PAPER.md 2>/dev/null || true
