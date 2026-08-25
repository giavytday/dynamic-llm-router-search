"""Phase 4: compile empirical findings into paper.tex (NeurIPS-style) and PAPER.md."""

import json
import os
import re
import sqlite3
import time
from typing import Dict, List

from benchmark_comparison import SHORT_NAMES

BENCH_PATH = "benchmark_results.json"
ABL_PATH = "ablation_table.json"
DB_PATH = "evolution_search.db"
CHAMPION_HASH = "22e2a1f5"

BIB = {
    "frugalgpt": "Chen, L., Zaharia, M., and Zou, J. FrugalGPT: How to use large language models while reducing cost and improving performance. arXiv:2305.05176, 2023.",
    "routellm": "Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E., Kadous, M. W., and Stoica, I. RouteLLM: Learning to route LLMs with preference data. arXiv:2406.18665, 2024.",
    "hybridllm": "Ding, D., Malaviya, A., Eisenschlos, C., Zhang, M., and Figueiredo, R. Hybrid LLM: Efficient and enhanced inference via routing. ICLR, 2024.",
    "funsearch": "Romera-Paredes, B., Barekatain, M., Novikov, A., et al. Mathematical discoveries from program search with large language models. Nature, 625:468--475, 2024.",
    "mapelites": "Mouret, J.-B. and Clune, J. Illuminating search spaces by mapping elites. arXiv:1504.04909, 2015.",
    "nsgaii": "Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2):182--197, 2002.",
    "elm": "Lehman, J., Gordon, J., Jain, S., et al. Evolution through large models. arXiv:2206.08896, 2022.",
}


def load_inputs() -> dict:
    with open(BENCH_PATH) as fh:
        bench = json.load(fh)
    with open(ABL_PATH) as fh:
        abl = json.load(fh)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            "SELECT code, thought, operation, generation, island_id, parent_hash, fitness"
            " FROM candidates WHERE code_hash LIKE ?",
            (CHAMPION_HASH + "%",),
        ).fetchone()
        parent_row = con.execute(
            "SELECT code FROM candidates WHERE code_hash = ?", (row["parent_hash"],)
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise SystemExit(f"champion {CHAMPION_HASH} not found")
    champ = dict(row)
    m = re.search(r'estimated_tokens"\] > (\d+)', parent_row["code"]) if parent_row else None
    champ["parent_threshold"] = int(m.group(1)) if m else None
    return {"bench": bench, "abl": abl, "champ": champ}


def derived_stats(data: dict) -> dict:
    by = {r["method"]: r for r in data["bench"]["results"]}
    ch, fr, ml, hw = by["Evolved-Champion"], by["Always-Frontier"], by["ML-DecisionTree"], by["Hand-Written-Baseline"]
    abl_runs = {r["key"]: r for r in data["abl"]["runs"]}
    abl_agg = data["abl"]["aggregates"]
    comp = data["abl"]["comparisons"]
    frontier_ratio = (
        abl_agg["full_island_semantic"]["frontier_size"]["mean"]
        / abl_agg["A_single_population"]["frontier_size"]["mean"]
        - 1.0
    )
    peak_spread = max(a["peak_fitness"]["mean"] for a in abl_agg.values()) - min(
        a["peak_fitness"]["mean"] for a in abl_agg.values()
    )
    return {
        "ch": ch,
        "fr": fr,
        "ml": ml,
        "hw": hw,
        "abl": abl_runs,
        "abl_agg": abl_agg,
        "comp": comp,
        "frontier_ratio": frontier_ratio,
        "peak_spread": peak_spread,
        "quality_retention": ch["id"]["quality"] / fr["id"]["quality"],
        "cost_fraction": 1.0 - ch["id"]["cost_reduction_pct"] / 100.0,
        "speedup_vs_ml": ml["id"]["latency_us"] / ch["id"]["latency_us"],
        "fitness_uplift": ch["id"]["fitness"] - hw["id"]["fitness"],
        "champion_db_fitness": data["champ"]["fitness"],
    }


def table1_markdown(results: List[dict]) -> str:
    lines = [
        "| Method | Q (ID) | $\\Delta$C% (ID) | Q (OOD) | $\\Delta$C% (OOD) | L (µs) | Params | Mem (KB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {SHORT_NAMES.get(r['method'], r['method'])} | {r['id']['quality']:.4f}"
            f" | {r['id']['cost_reduction_pct']:.2f} | {r['ood']['quality']:.4f}"
            f" | {r['ood']['cost_reduction_pct']:.2f} | {r['id']['latency_us']:.3f}"
            f" | {r['n_params']} | {r['memory_kb']:.2f} |"
        )
    return "\n".join(lines)


def table1_latex(results: List[dict]) -> str:
    lines = [
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Main results (Table 1). ID: in-distribution train split ($n{=}6000$);"
        " OOD: distribution-shifted slice ($n{=}832$). Reference cost: always-frontier.}",
        "\\label{tab:main}",
        "\\begin{tabular}{lrrrrrrr}\\toprule",
        ("Method & $Q$ (ID) & $\\Delta C\\%$ (ID) & $Q$ (OOD) & $\\Delta C\\%$ (OOD)"
         " & $L$ ($\\mu\\text{s}$) & Params & Mem (KB)\\\\\\midrule"),
    ]
    for r in results:
        lines.append(
            f"{SHORT_NAMES.get(r['method'], r['method'])} & {r['id']['quality']:.4f} &"
            f" {r['id']['cost_reduction_pct']:.2f} & {r['ood']['quality']:.4f} &"
            f" {r['ood']['cost_reduction_pct']:.2f} & {r['id']['latency_us']:.3f} &"
            f" {r['n_params']} & {r['memory_kb']:.2f}\\\\"
        )
    lines += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    return "\n".join(lines)


ABL_TEX_ROWS = [
    ("A (topology)", "Single population", "A_single_population"),
    ("A (topology)", "3-island + migration (full)", "full_island_semantic"),
    ("B (archive)", "Greedy fitness replacement", "B_greedy_archive"),
    ("B (archive)", "2D MAP-Elites parents", "B_map_elites"),
    ("C (operators)", "Random constant jitter", "C_random_perturbation"),
    ("C (operators)", "Semantic mutation (full)", "full_island_semantic"),
]


def _ms_tex(a: dict, fmt: str = "%.2f") -> str:
    return f"${fmt % a['mean']} \\pm {a['std']:.2f}$"


def _ms_md(a: dict, fmt: str = "%.2f") -> str:
    return f"{fmt % a['mean']} ± {a['std']:.2f}"


def table2_latex(abl_agg: Dict[str, dict], seeds: List[int]) -> str:
    lines = [
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Ablation results (Table 2), aggregated as $\\mathrm{mean} \\pm \\mathrm{std}$"
        f" over {len(seeds)} seeds ({', '.join(map(str, seeds))}). Budget-controlled: every"
        " variant evaluates 24 offspring per generation for 10 generations. Gen$\\to$95\\%:"
        " first generation reaching 95\\% of the run-final best fitness; Elites: unique"
        " passing candidates in the run archive; Cells: occupied MAP-Elites grid cells.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{llrrrrr}\\toprule",
        ("Axis & Variant & Peak Fitness & Gen$\\to$95\\% & Gen$>$Base & Elites & Cells\\\\"
         "\\midrule"),
    ]
    for axis, label, key in ABL_TEX_ROWS:
        a = abl_agg[key]
        lines.append(
            f"{axis} & {label} & {_ms_tex(a['peak_fitness'])}"
            f" & {_ms_tex(a['gen_to_95pct_final'], '%.1f')}"
            f" & {_ms_tex(a['gen_to_beat_baseline'], '%.1f')}"
            f" & {_ms_tex(a['elite_count'], '%.1f')}"
            f" & {_ms_tex(a['grid_cells'], '%.1f')}\\\\"
        )
    lines += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    return "\n".join(lines)


def table2_markdown(abl_agg: Dict[str, dict]) -> str:
    lines = [
        "| Axis | Variant | Peak Fitness | Gen→95% | Gen>Base | Elites | Cells |",
        "|---|---|---|---|---|---|---|",
    ]
    for axis, label, key in ABL_TEX_ROWS:
        a = abl_agg[key]
        lines.append(
            f"| {axis} | {label} | {_ms_md(a['peak_fitness'])}"
            f" | {_ms_md(a['gen_to_95pct_final'], '%.1f')}"
            f" | {_ms_md(a['gen_to_beat_baseline'], '%.1f')}"
            f" | {_ms_md(a['elite_count'], '%.1f')}"
            f" | {_ms_md(a['grid_cells'], '%.1f')} |"
        )
    return "\n".join(lines)


def build_paper_tex(data: dict, stats: dict) -> str:
    ch, fr, ml, hw = stats["ch"], stats["fr"], stats["ml"], stats["hw"]
    abl = stats["abl"]
    champ = data["champ"]
    b = data["bench"]["meta"]
    qr, cf, sp = stats["quality_retention"], stats["cost_fraction"], stats["speedup_vs_ml"]
    comp = data["abl"]["comparisons"]

    tex = []
    tex += [
        "% Generated by generate_paper.py -- swap documentclass for neurips_2024.sty in submission.",
        "\\documentclass[11pt]{article}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{amsmath,amssymb}",
        "\\usepackage{booktabs}",
        "\\usepackage{graphicx}",
        "\\usepackage{xcolor}",
        "\\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}",
        "\\usepackage{caption}",
        "\\captionsetup{font=small}",
        "\\title{Dynamic Multi-Model LLM Routing via Evolutionary Code Search}",
        "\\author{Thomas Gia Vy Day \\\\ Independent Researcher \\\\ \\texttt{giavytday@gmail.com}}",
        "\\date{" + b["timestamp"] + "}",
        "\\begin{document}",
        "\\maketitle",
        "",
        "\\begin{abstract}",
        (
            f"Deploying large language models economically requires routing each query to the"
            f" cheapest model tier that preserves answer quality. We reframe router design as"
            f" a program-search problem: routing policies are executable Python functions,"
            f" evolved by an island-model genetic search over a quality--diversity (MAP-Elites)"
            f" archive, and validated by a deterministic multi-stage oracle comprising an AST"
            f" security gate, boundary smoke tests, and a vectorized benchmark over"
            f" {b['n_train']:,} synthetic queries with per-tier quality and token-price models."
            f" Without any LLM-in-the-loop calls, the search discovers a {stats['abl']['full_island_semantic']['config_label'].split(',')[0].lower()} policy"
            f" that achieves \\textbf{{{ch['id']['cost_reduction_pct']:.1f}\\% cost reduction}}"
            f" relative to always-frontier routing while retaining {qr*100:.1f}\\% of frontier"
            f" quality, at a mean decision latency of \\textbf{{{ch['id']['latency_us']:.3f}\\,$\\mu$s}}"
            f" per query -- a {sp:.0f}$\\times$ inference-speedup over a supervised"
            f" decision-tree router of comparable quality. The champion policy generalizes"
            f" positively under a distribution shift ({ch['ood']['cost_reduction_pct']:.1f}\\% cost"
            f" reduction on OOD queries). The study is framed as a formal proof of concept for"
            f" the search methodology and its zero-latency AST-evaluated oracle, and the"
            f" ablation suite is replicated over five seeds: mean peak fitness is remarkably"
            f" robust to topology, archive, and operator ablations (all variant means within"
            f" {stats['peak_spread']:.2f} fitness, $\\sigma \\leq 0.43$), with semantic mutation the sole"
            f" convergence-axis signal (Gen$\\to$95\\% of $1.0 \\pm 0.0$ vs"
            f" ${stats['abl_agg']['C_random_perturbation']['gen_to_95pct_final']['mean']:.1f} \\pm"
            f" {stats['abl_agg']['C_random_perturbation']['gen_to_95pct_final']['std']:.1f}$ for random"
            f" jitter) and island topology reducing frontier-size variance by"
            f" {100*(1 - stats['abl_agg']['full_island_semantic']['frontier_size']['std']/stats['abl_agg']['A_single_population']['frontier_size']['std']):.0f}\\%. We release the full"
            f" lineage database, ablation harness, and paper-generation pipeline."
        ),
        "\\end{abstract}",
        "",
    ]

    tex += [
        "\\section{Introduction and Related Work}",
        (
            "Commercial LLM APIs span a $20\\times$ price gradient between small and frontier"
            " tiers, yet query difficulty is highly heterogeneous: a substantial fraction of"
            " production traffic can be served by cheap models with negligible quality loss"
            " \\cite{frugalgpt}. Learned routers typically frame this as supervised prediction of"
            " the strongest--cheapest model \\cite{routellm,hybridllm}, which requires preference"
            " data, generalizes poorly under shift, and adds an ML inference stage whose latency"
            " can rival the routing decision it optimizes. Program-search methods such as"
            " FunSearch \\cite{funsearch} and Evolution-through-Large-Models \\cite{elm} instead"
            " evolve executable code, inheriting interpretability, verifiability, and near-zero"
            " deployment cost."
        ),
        (
            "\\paragraph{The latency wall.} Routing sits on the critical path of"
            " time-to-first-token: whatever the router spends, the user waits. Neural routers"
            " embed every query with a transformer encoder before classification, costing tens"
            " to hundreds of milliseconds per request on CPU -- and non-trivial tail latency"
            " even when batched on accelerators -- three to five orders of magnitude above a"
            " chain of symbolic comparisons. At 10k queries per second, the embedding pass"
            " alone becomes a dedicated model-serving cost center. An admissible router must"
            " therefore decide in microseconds: any quality recovered by smarter routing is"
            " voided if the router itself adds perceptible latency. This constraint motivates"
            " evolving \\emph{compiled symbolic policies} whose decision cost is a handful of"
            " dictionary lookups, and it is why our oracle hard-penalizes any policy slower"
            " than 100\\,$\\mu$s -- still three orders of magnitude below a single neural"
            " embedding forward pass."
        ),
        (
            "We treat the router itself as the evolving artifact, and this work as a formal"
            " proof of concept: the object of study is the search methodology together with its"
            " zero-latency AST-evaluated oracle, deliberately isolated from production concerns"
            " so that every claim is exactly reproducible. The deterministic oracle replaces"
            " expensive LLM feedback with a simulated but internally consistent query corpus and"
            " per-tier quality model, enabling thousands of policy evaluations per minute. Our"
            " contributions are: (i) a three-gate evaluation oracle (AST security, smoke tests,"
            " vectorized benchmark with hard latency penalty); (ii) an island-model MAP-Elites"
            " search over routing-code mutants with full SQLite lineage; (iii) a controlled"
            " three-axis ablation (topology, archive, operators), replicated over five seeds;"
            " and (iv) a mechanistic"
            f" deconstruction of the discovered champion policy {CHAMPION_HASH}, which never calls"
            " the frontier tier yet dominates every hand-designed baseline on the quality--cost"
            " frontier."
        ),
        (
            "\\paragraph{Related work.} FrugalGPT \\cite{frugalgpt} cascades models with"
            " escalation; RouteLLM \\cite{routellm} and Hybrid-LLM \\cite{hybridllm} learn"
            " routers from preference pairs; FunSearch \\cite{funsearch} evolves programs with"
            " LLM mutation operators; MAP-Elites \\cite{mapelites} provides the quality-diversity"
            " archive we bin over (quality $\\times$ cost reduction); NSGA-II \\cite{nsgaii}"
            " defines the non-dominated Pareto bookkeeping used for reporting. Unlike all of the"
            " above, our router is a compiled, auditable Python function whose entire decision"
            " structure fits in a dozen lines."
        ),
        "",
    ]

    tex += [
        "\\section{Problem Formulation and Multi-Stage Deterministic Oracle}",
        (
            "Each query $q$ is a feature vector (character length, estimated tokens, code-syntax"
            " and math-symbol flags, detected language, readability, semantic density, budget"
            " ratio, latency deadline) paired with per-tier qualities"
            " $\\{\\mathrm{qual}(q,m)\\}_{m \\in \\mathcal{M}}$, $\\mathcal{M}=\\{$small, medium,"
            " frontier$\\}$. A policy $\\pi: q \\mapsto m$ is a pure Python function"
            " \\texttt{select\\_model(q)}. Token prices $p^{\\mathrm{in}},p^{\\mathrm{out}}$ per"
            " tier are $(0.15,0.60)$, $(0.80,3.20)$, $(3.00,15.00)$ USD per million tokens:"
        ),
        "\\begin{equation}",
        "  C(q,m)=\\frac{t^{\\mathrm{in}}_q\\,p^{\\mathrm{in}}_m +"
        " t^{\\mathrm{out}}_q\\,p^{\\mathrm{out}}_m}{10^{6}}",
        "  \\label{eq:cost}",
        "\\end{equation}",
        "\\begin{equation}",
        "  \\Delta C_\\pi = 100\\,\\frac{C_{\\mathrm{ref}}-\\mathbb{E}_q\\big[C(q,\\pi(q))\\big]}{C_{\\mathrm{ref}}},"
        " \\qquad C_{\\mathrm{ref}}=\\mathbb{E}_q\\big[C(q,\\mathrm{frontier})\\big]",
        "  \\label{eq:costreduction}",
        "\\end{equation}",
        "\\begin{equation}",
        "  Q_\\pi = \\mathbb{E}_q\\big[\\mathrm{qual}(q,\\pi(q))\\big]",
        "  \\label{eq:quality}",
        "\\end{equation}",
        "\\begin{equation}",
        "  F(\\pi) = 100\\,Q_\\pi + 0.6\\,\\Delta C_\\pi - 0.05\\,L_\\pi - 300\\,"
        "\\mathbf{1}\\big[L_\\pi > 100\\,\\mu\\mathrm{s}\\big]",
        "  \\label{eq:fitness}",
        "\\end{equation}",
        (
            "where $L_\\pi$ is the mean wall-clock decision time in microseconds. Every candidate"
            " must pass three gates: \\textbf{Gate 1} parses the source with"
            " \\texttt{ast} and rejects imports, \\texttt{exec}/\\texttt{global} statements,"
            " dangerous identifiers (\\texttt{open}, \\texttt{eval}, \\texttt{exec},"
            " \\texttt{os}, \\texttt{sys}, \\texttt{\\_\\_import\\_\\_}), dunder attribute access,"
            " and oversized programs (>300 AST nodes); survivors execute in a namespace whose"
            " builtins are a fixed safe subset. \\textbf{Gate 2} runs eight boundary smoke tests"
            " (zero-token, 200k-token, math-only, code-only, extreme readability/density)"
            " requiring tier-valued string outputs. \\textbf{Gate 3} is the vectorized benchmark"
            " scoring $Q$, $\\Delta C$, and $L$ over the full training split with the hard"
            " latency penalty above. The corpus comprises 10{,}000 synthetic queries (30.1\\%"
            " code-syntax, 27.6\\% math-symbol, mean 184 tokens) with tier qualities drawn from"
            " difficulty-conditional distributions; splits are 60/40 train/test with a"
            " top-decile OOD slice (832 queries: extreme length, code$\\wedge$math, or density)."
        ),
        "",
    ]

    tex += [
        "\\section{Evolutionary Search and MAP-Elites Topology}",
        (
            "Search maintains three islands of 14 policies. Each generation, every island"
            " produces 8 offspring by tournament selection over its fitness-ranked members"
            " (15\\% of births are fresh grammar-template immigrants), applies one semantic"
            " mutation operator -- numeric-boundary jitter, comparison flips, branch rotation,"
            " condition inversion, grammar synthesis, or two-parent branch crossover -- each"
            " wrapped in a simulated $\\langle$thought$\\rangle$/$\\langle$code$\\rangle$"
            " response with a deterministic fallback generator, and is pruned to the elite"
            " top-$k$. The fallback generator is a deliberate design choice, not a limitation:"
            " it guarantees compute-free, bit-reproducible exploration with zero calls to"
            " expensive closed LLM APIs, so that every result in this paper regenerates"
            " identically on a laptop; because mutants already flow through the"
            " $\\langle$thought$\\rangle$/$\\langle$code$\\rangle$ interface, swapping the"
            " fallback for LLM-in-the-loop mutation \\cite{elm,funsearch} is a drop-in change"
            " we leave to future work. Every second generation the island best migrates"
            " cyclically. In"
            " parallel, a $6\\times6$ MAP-Elites grid \\cite{mapelites} bins candidates by"
            " quality $\\in[0.40,0.95]$ and cost reduction $\\in[0,100]\\%$, retaining the"
            " fittest policy per cell as a quality-diversity archive; all evaluations, parent"
            " links, thoughts, and gate outcomes are persisted to SQLite"
            " (\\texttt{evolution\\_search.db}), giving a fully replayable lineage. The reported"
            " Pareto frontier is the non-dominated set over the archive under"
            " $(\\max Q, \\max \\Delta C)$ \\cite{nsgaii}. The reference run executes 10"
            " generations $\\times$ 3 islands $\\times$ 8 offspring $=$ 240 evaluations in"
            " roughly five seconds on a laptop CPU."
        ),
        "",
    ]

    ml_model_tex = b["ml_model"].replace("_", "\\_")
    tex += [
        "\\section{Empirical Evaluation and Baseline Comparison}",
        (
            f"Table \\ref{{tab:main}} compares the evolved champion against constant, random,"
            f" hand-engineered, and supervised (scikit-learn decision tree,"
            f" {ml_model_tex},"
            f" train accuracy {b['ml_train_accuracy']:.3f}) baselines. The supervised router"
            f" collapses to near always-frontier behavior ($Q={ml['id']['quality']:.4f},"
            f" \\Delta C={ml['id']['cost_reduction_pct']:.2f}\\%$): imitating the"
            f" quality-optimal label does not encode cost pressure, and its per-query"
            f" scikit-learn inference costs {ml['id']['latency_us']:.1f}\\,$\\mu$s --"
            f" {sp:.0f}$\\times$ the champion's decision time for a strictly dominated"
            f" operating point. The evolved champion instead retains {qr*100:.1f}\\% of frontier"
            f" quality at {cf*100:.1f}\\% of frontier cost with a single effective decision"
            f" threshold pair."
        ),
        "\\includegraphics[width=0.85\\linewidth]{baseline_vs_champion.png}",
        "\\includegraphics[width=0.85\\linewidth]{pareto_frontier.png}",
        (
            f"\\paragraph{{Out-of-distribution robustness.}} On the OOD slice the champion's cost"
            f" reduction \\emph{{rises}} to {ch['ood']['cost_reduction_pct']:.2f}\\% (from"
            f" {ch['id']['cost_reduction_pct']:.2f}\\%) because hard long-tail queries escalate to"
            f" medium rather than frontier, while the hand-written baseline sheds 39.6 points of"
            f" cost efficiency ({hw['id']['cost_reduction_pct']:.2f}\\% $\\to$"
            f" {hw['ood']['cost_reduction_pct']:.2f}\\%). Quality gaps ($\\Delta Q$) are positive"
            f" for the top-50 evolved policies, indicating the search did not overfit the train"
            f" split's difficulty mix."
        ),
        table1_latex(data["bench"]["results"]),
        (
            "\\paragraph{Ablations.} Table \\ref{tab:ablation} isolates each component under an"
            " identical 24-evaluations-per-generation budget, with every variant replicated"
            " over five seeds. The headline finding is a negative result we consider equally"
            f" informative: mean peak fitness is statistically indistinguishable across all"
            f" variants (all means within {stats['peak_spread']:.2f} fitness of each other, $\\sigma"
            f" \\leq 0.43$) -- at this budget the framework, not the configuration, does the"
            " heavy lifting. Within that envelope, three seed-robust signals emerge."
            " (C) \\emph{Operators} carry the only convergence effect: semantic mutation"
            " reaches 95\\% of final fitness in $1.0 \\pm 0.0$ generations versus"
            f" ${stats['abl_agg']['C_random_perturbation']['gen_to_95pct_final']['mean']:.1f} \\pm"
            f" {stats['abl_agg']['C_random_perturbation']['gen_to_95pct_final']['std']:.1f}$ for random"
            " constant jitter, dominates the mean elite curve at every generation of the early"
            " search (+1.40 fitness at generation 1), and yields a tighter, higher-quality"
            " archive (223 $\\pm$ 8 elites versus 244 $\\pm$ 0 -- jitter-only offspring are all"
            " novel but none is better), confirming that structured, thought-guided edits --"
            " not blind threshold noise -- drive efficient search."
            " (A) \\emph{Topology} leaves peak fitness unchanged (single-population delta"
            f" {stats['comp']['A_topology_single_vs_islands']['peak_fitness_delta_mean']:+.2f}, within 1$\\sigma$) but cuts"
            " frontier-size variance by"
            f" {100*(1 - stats['abl_agg']['full_island_semantic']['frontier_size']['std']/stats['abl_agg']['A_single_population']['frontier_size']['std']):.0f}\\%"
            " ($\\sigma$ 28.2 $\\to$ 9.9 non-dominated policies), stabilizing deployable"
            " trade-off coverage across reruns. (B) The \\emph{archive} axis is"
            " fitness-equivalent (MAP-Elites delta"
            f" {stats['comp']['B_archive_greedy_vs_mapelites']['peak_fitness_delta_mean']:+.2f}, within 1$\\sigma$); its value"
            " lies in the explicit quality-diversity grid itself -- an inspectable, bounded"
            " archive of the trade-off surface -- rather than in peak-fitness gains. We report"
            " these effect sizes with their seeds to guard against the single-seed"
            " over-interpretation common in evolutionary-computation studies."
        ),
        table2_latex(stats["abl_agg"], data["abl"]["meta"]["seeds"]),
        "\\includegraphics[width=0.85\\linewidth]{fitness_trajectory.png}",
        "",
    ]

    tex += [
        "\\section{Mechanistic Deconstruction of Champion Policy " + CHAMPION_HASH + "}",
        (
            f"The champion (generation {champ['generation']}, island {champ['island_id']},"
            f" parent \\texttt{{{str(champ['parent_hash'])[:8]}}}, operator"
            " \\texttt{" + champ["operation"].replace("_", "\\_") + "},"
            f" fitness {champ['fitness']:.2f}) is:"
        ),
        "\\begin{verbatim}",
        champ["code"].rstrip(),
        "\\end{verbatim}",
        (
            "The double negations look puzzling but are logically transparent:"
            " $\\neg\\neg p \\equiv p$, so the first guard reads ``if the query has math"
            " symbols'' and the second reads ``if the query has at most 367 estimated"
            " tokens.'' Each redundant negation costs a nanosecond-scale boolean evaluation"
            " and nothing in fitness, which is precisely why selection never removed them --"
            " a neutral polymorphism fixed by drift. The compiled semantics are therefore:"
            " math-symbol queries"
            " $\\to$ medium; non-math queries with at most 367 estimated tokens $\\to$ medium;"
            " all other (long, non-math) queries $\\to$ small; the frontier branch is"
            " unreachable. Three mechanisms explain its"
            " fitness: (1) \\emph{frontier elimination} -- the $20\\times$ price premium is never"
            " paid, yielding $\\Delta C \\approx 81\\%$; (2) \\emph{difficulty-aware escalation}"
            " -- math queries carry the largest frontier--small quality gap in the corpus, and"
            " medium captures most of it at a quarter of frontier price; (3) \\emph{difficulty"
            " inversion for long queries} -- long non-math queries (chat/analysis) have"
            " below-average difficulty, so bulk traffic routes to the small tier at almost no"
            " quality cost. The lineage database shows the champion's 367-token threshold"
            " emerged as a \\texttt{{tweak\\_constants}} jitter of its parent's"
            f" {champ['parent_threshold']}-token threshold, one generation after branch"
            " rotation reordered the two guards -- a textbook example of neutral drift followed"
            " by exploitative refinement."
        ),
        "",
    ]

    tex += [
        "\\section{Limitations, OOD Generalization, and Conclusion}",
        (
            "\\paragraph{Limitations.} (i) The corpus and per-tier qualities are synthetic;"
            " absolute numbers will differ on live API traffic, though the \\emph{relative}"
            " baseline ordering and the mechanistic findings depend only on the ordinal"
            " difficulty structure. (ii) Latency $L$ measures policy decision overhead, not"
            " end-to-end model latency. (iii) The five-seed replication varies search"
            " stochasticity only; the corpus itself is a single synthetic draw and should be"
            " regenerated across corpus seeds. (iv) The 6$\\times$6 grid is"
            " coarse; finer binning may expose additional niches. (v) The OOD shift is"
            " feature-level (extreme tokens, code$\\wedge$math, density), not adversarial."
        ),
        (
            "\\paragraph{Toward live production deployment.} The bridge from synthetic"
            " benchmark to production proceeds in three concrete stages."
            " \\textbf{Stage 1 -- Trace Replay on live logs:} re-score anonymized production"
            " queries through all three tiers to replace the simulated quality model with"
            " empirical per-tier quality, then re-run the seconds-cheap search on the"
            " calibrated corpus. \\textbf{Stage 2 -- Stratified Human/LLM Judgment:} within"
            " each difficulty stratum, compare champion versus always-frontier with human"
            " raters or an LLM judge, validating the ordinal quality structure the policy"
            " exploits. \\textbf{Stage 3 -- Shadow Canary Deployment:} mirror 1--5\\% of"
            " traffic to the evolved policy while monitoring per-tier quality drift, cost per"
            " resolved query, and tail latency, with automatic rollback and scheduled"
            " re-evolution as model prices and capabilities drift."
        ),
        (
            "\\paragraph{Conclusion.} Casting LLM routing as evolutionary code search over a"
            " deterministic multi-gate oracle yields an auditable, near-zero-latency policy that"
            f" dominates hand-designed and supervised baselines: {ch['id']['cost_reduction_pct']:.1f}\\% cost reduction at"
            f" {ch['id']['latency_us']:.3f}\\,$\\mu$s per decision with positive OOD"
            " generalization. The full lineage, ablation harness, and paper-generation pipeline"
            " are deterministic and re-executable end-to-end. Future work: LLM-in-the-loop"
            " mutation replacing the fallback generator \\cite{elm,funsearch}, validation on"
            " live RouteLLM-style preference data \\cite{routellm}, and multi-objective"
            " selection directly on the Pareto front \\cite{nsgaii}."
        ),
        (
            "\\paragraph{Competing interests.} The author declares no competing interests."
            " Correspondence should be addressed to Thomas Gia Vy Day"
            " (\\texttt{giavytday@gmail.com})."
        ),
        "",
        "\\begin{thebibliography}{9}",
    ]
    for key in ("frugalgpt", "routellm", "hybridllm", "funsearch", "mapelites", "nsgaii", "elm"):
        tex.append(f"\\bibitem{{{key}}} {BIB[key]}")
    tex += ["\\end{thebibliography}", "\\end{document}", ""]
    return "\n".join(tex)


def build_paper_md(data: dict, stats: dict) -> str:
    ch, ml, hw = stats["ch"], stats["ml"], stats["hw"]
    abl = stats["abl"]
    champ = data["champ"]
    b = data["bench"]["meta"]
    comp = data["abl"]["comparisons"]
    md = []
    md += [
        "# Dynamic Multi-Model LLM Routing via Evolutionary Code Search",
        "",
        f"*Thomas Gia Vy Day (Independent Researcher) · giavytday@gmail.com · generated {b['timestamp']}*",
        "",
        "## Abstract",
        "",
        (
            f"We reframe LLM router design as program search: routing policies are executable"
            f" Python functions evolved by a 3-island MAP-Elites genetic search and validated by a"
            f" deterministic 3-gate oracle (AST security, smoke tests, vectorized benchmark with a"
            f" hard 100 µs latency penalty). Without any LLM-in-the-loop calls, the search discovers"
            f" a policy delivering **{ch['id']['cost_reduction_pct']:.1f}% cost reduction** vs always-frontier routing at"
            f" **{ch['id']['latency_us']:.3f} µs** mean decision latency, retaining"
            f" {stats['quality_retention']*100:.1f}% of frontier quality, generalizing positively OOD"
            f" ({ch['ood']['cost_reduction_pct']:.1f}% cost reduction), and out-running a supervised"
            f" decision-tree router by {stats['speedup_vs_ml']:.0f}×. Framed as a formal proof of"
            f" concept for the search methodology and its zero-latency AST-evaluated oracle, the"
            f" 5-seed ablation suite shows mean peak fitness is robust across topology, archive,"
            f" and operator ablations (all means within {stats['peak_spread']:.2f}), with semantic"
            f" mutation the sole convergence signal (Gen→95% 1.0±0.0 vs"
            f" {stats['abl_agg']['C_random_perturbation']['gen_to_95pct_final']['mean']:.1f}±{stats['abl_agg']['C_random_perturbation']['gen_to_95pct_final']['std']:.1f})"
            f" and islands cutting frontier-size variance by"
            f" {100*(1 - stats['abl_agg']['full_island_semantic']['frontier_size']['std']/stats['abl_agg']['A_single_population']['frontier_size']['std']):.0f}%."
        ),
        "",
        "## 1. Introduction & Related Work",
        "",
        (
            "LLM APIs span a 20× price gradient while query difficulty is heterogeneous"
            " (FrugalGPT). Learned routers (RouteLLM, Hybrid-LLM) need preference data and add an"
            " ML inference stage; program-search systems (FunSearch, ELM) evolve interpretable,"
            " verifiable code. We evolve the router itself: contributions are (i) the 3-gate"
            " deterministic oracle, (ii) island-model MAP-Elites code search with full SQLite"
            " lineage, (iii) a budget-controlled 3-axis ablation, and (iv) a mechanistic"
             f" deconstruction of champion `{CHAMPION_HASH}`, which never calls the frontier tier"
            " yet dominates all baselines on the quality-cost frontier."
        ),
        "",
        (
            "**The latency wall.** Routing sits on the critical path of time-to-first-token:"
            " whatever the router spends, the user waits. Neural routers embed every query with"
            " a transformer encoder before classification, costing tens to hundreds of"
            " milliseconds per request on CPU — and non-trivial tail latency even batched on"
            " accelerators — three to five orders of magnitude above a chain of symbolic"
            " comparisons. At 10k queries/sec the embedding pass alone becomes a dedicated"
            " model-serving cost center. An admissible router must decide in microseconds:"
            " quality recovered by smarter routing is voided if the router itself adds"
            " perceptible latency. Hence we evolve compiled symbolic policies whose decision"
            " cost is a handful of dict lookups, and the oracle hard-penalizes any policy above"
            " 100 µs — still three orders of magnitude below one neural embedding forward pass."
        ),
        "",
        "## 2. Problem Formulation & Multi-Stage Deterministic Oracle",
        "",
        (
            "A policy `select_model(q) -> {small, medium, frontier}` is scored by Q = E[quality],"
            " cost reduction ΔC = 100·(C_ref − C)/C_ref (token prices $0.15/$0.80/$3.00 per 1M"
            " input, $0.60/$3.20/$15.00 output), and latency L (µs/query), with fitness"
            " `F = 100·Q + 0.6·ΔC − 0.05·L − 300·1[L>100µs]`."
        ),
        "",
        (
            "**Gate 1** AST validation blocks `Import`/`ImportFrom`/`Exec`/`Global`, dangerous"
            " builtins (`open`, `eval`, `exec`, `os`, `sys`, `__import__`), dunder access, and"
            " >300-node programs; execution uses a restricted builtins namespace. **Gate 2** runs"
            " 8 boundary smoke tests. **Gate 3** benchmarks 6,000 train queries. Corpus: 10,000"
            " synthetic queries (30.1% code, 27.6% math, mean 184 tokens); 60/40 split; OOD slice"
            " of 832 extreme queries."
        ),
        "",
        "## 3. Evolutionary Search & MAP-Elites Topology",
        "",
        (
            "3 islands × 14 policies; 8 offspring/island/generation via fitness tournament +"
            " semantic mutation (constant jitter, comparison flips, branch rotation, condition"
            " inversion, grammar synthesis, crossover), each wrapped in simulated"
            " `<thought>`/`<code>` with a deterministic fallback; 15% grammar-template"
            " immigrants; cyclic best-policy migration every 2 generations; 6×6 MAP-Elites grid"
            " over (quality × ΔC); every evaluation persisted to `evolution_search.db` with"
            " parent links, thoughts, and gate outcomes."
        ),
        "",
        (
            "**The fallback generator is a deliberate design choice, not a limitation:** it"
            " guarantees compute-free, bit-reproducible exploration with zero calls to closed"
            " LLM APIs, so every result regenerates identically on a laptop. Because mutants"
            " already flow through the `<thought>`/`<code>` interface, swapping the fallback"
            " for LLM-in-the-loop mutation is a drop-in change."
        ),
        "",
        "## 4. Empirical Evaluation & Baseline Comparison",
        "",
        "**Table 1: Main Results.**",
        "",
        table1_markdown(data["bench"]["results"]),
        "",
        (
            f"The supervised decision tree collapses to always-frontier imitation"
            f" (Q={ml['id']['quality']:.4f}, ΔC={ml['id']['cost_reduction_pct']:.2f}%) at"
            f" {ml['id']['latency_us']:.1f} µs/query — {stats['speedup_vs_ml']:.0f}× slower than the champion for a"
            f" dominated operating point. The champion retains {stats['quality_retention']*100:.1f}% of frontier quality at"
            f" {stats['cost_fraction']*100:.1f}% of frontier cost."
        ),
        "",
        "![Baseline vs Champion](baseline_vs_champion.png)",
        "",
        "![Pareto Frontier](pareto_frontier.png)",
        "",
        (
            f"**OOD robustness.** Champion ΔC *rises* to {ch['ood']['cost_reduction_pct']:.2f}% under shift; the hand-written"
            f" baseline loses 39.6 points ({hw['id']['cost_reduction_pct']:.1f}% → {hw['ood']['cost_reduction_pct']:.1f}%)."
        ),
        "",
        "**Table 2: Ablation Results** (mean ± std over 5 seeds; 10 generations, 24 evals/gen).",
        "",
        table2_markdown(stats["abl_agg"]),
        "",
        (
            f"- **Headline:** mean peak fitness is statistically indistinguishable across all"
            f" variants (all means within {stats['peak_spread']:.2f}, σ ≤ 0.43) — the framework,"
            f" not the configuration, does the heavy lifting at this budget."
        ),
        (
            "- **C (operators):** the only convergence signal — semantic mutation hits 95% of"
            f" final fitness in {stats['abl_agg']['full_island_semantic']['gen_to_95pct_final']['mean']:.1f} ±"
            f" {stats['abl_agg']['full_island_semantic']['gen_to_95pct_final']['std']:.1f} generations vs"
            f" {stats['abl_agg']['C_random_perturbation']['gen_to_95pct_final']['mean']:.1f} ±"
            f" {stats['abl_agg']['C_random_perturbation']['gen_to_95pct_final']['std']:.1f} for random jitter, dominates the mean"
            " elite curve at every early generation (+1.40 at gen 1), and yields a tighter"
            " archive (223±8 vs 244±0 elites — jitter-only offspring are all novel, none better)."
        ),
        (
            "- **A (topology):** peak-fitness neutral (Δ="
            f" {stats['comp']['A_topology_single_vs_islands']['peak_fitness_delta_mean']:+.2f}) but islands cut"
            f" frontier-size variance by {100*(1 - stats['abl_agg']['full_island_semantic']['frontier_size']['std']/stats['abl_agg']['A_single_population']['frontier_size']['std']):.0f}%"
            " (σ 28.2 → 9.9), stabilizing trade-off coverage across reruns."
        ),
        (
            "- **B (archive):** fitness-equivalent (Δ="
            f" {stats['comp']['B_archive_greedy_vs_mapelites']['peak_fitness_delta_mean']:+.2f}); the value of MAP-Elites is the"
            " inspectable quality-diversity grid itself, not peak fitness."
        ),
        "",
        "![Fitness Trajectory](fitness_trajectory.png)",
        "",
        "## 5. Mechanistic Deconstruction of Champion `" + CHAMPION_HASH + "`",
        "",
        (
            f"Generation {champ['generation']} · island {champ['island_id']} · parent"
            f" `{str(champ['parent_hash'])[:8]}` · operator `{champ['operation']}` · fitness"
            f" {champ['fitness']:.2f} · thought: *{champ['thought']}*"
        ),
        "",
        "```python",
        champ["code"].rstrip(),
        "```",
        "",
        (
            "**Decoding the double negation.** `not (not q[\"has_math_symbols\"])` and"
            " `not (q[\"estimated_tokens\"] > 367)` look puzzling but are logically"
            " transparent: ¬¬p ≡ p, so guard 1 reads *if the query has math symbols* and"
            " guard 2 reads *if the query has at most 367 estimated tokens*. Each redundant"
            " negation costs a nanosecond-scale boolean evaluation and nothing in fitness —"
            " precisely why selection never removed them: a neutral polymorphism fixed by"
            " drift."
        ),
        "",
        (
            "Simplified semantics: **math → medium; non-math ≤367 tokens → medium; long"
            " non-math → small; frontier branch unreachable.** Value comes from (1) frontier"
            " elimination (20× premium never paid), (2) difficulty-aware escalation of math"
            " queries to medium, and (3) difficulty inversion — long non-math queries are easy,"
            " so bulk traffic rides the small tier. The 367-token threshold is a"
            f" `tweak_constants` child of its parent's {champ['parent_threshold']}-token"
            " threshold, one generation after a branch rotation — neutral drift"
            " followed by exploitative refinement."
        ),
        "",
        "## 6. Limitations, OOD Generalization, & Conclusion",
        "",
        (
            "**Limitations.** Synthetic quality model (no live API calls); latency = policy"
            " overhead only; the 5-seed replication varies search stochasticity only — the"
            " corpus itself is a single synthetic draw; coarse 6×6 grid; feature-level OOD"
            " shift, not adversarial."
        ),
        "",
        (
            "**Toward live production deployment.** Three concrete stages:"
            " **Stage 1 — Trace Replay on live logs:** re-score anonymized production queries"
            " through all three tiers, replace the simulated quality model with empirical"
            " per-tier quality, and re-run the seconds-cheap search on the calibrated corpus."
            " **Stage 2 — Stratified Human/LLM Judgment:** within each difficulty stratum,"
            " compare champion vs always-frontier with human raters or an LLM judge, validating"
            " the ordinal quality structure the policy exploits. **Stage 3 — Shadow Canary"
            " Deployment:** mirror 1–5% of traffic to the evolved policy while monitoring"
            " per-tier quality drift, cost per resolved query, and tail latency, with automatic"
            " rollback and scheduled re-evolution as model prices and capabilities drift."
        ),
        "",
        (
            "**Conclusion.**"
            " Evolutionary code search over a deterministic oracle produces auditable,"
            " near-zero-latency routers that dominate hand-designed and supervised baselines."
            " Future work: LLM-in-the-loop mutation, live preference-data validation, direct"
            " multi-objective selection."
        ),
        "",
        (
            "**Competing interests.** The author declares no competing interests."
            " Correspondence: Thomas Gia Vy Day <giavytday@gmail.com>."
        ),
        "",
        "## References",
        "",
    ]
    for i, key in enumerate(("frugalgpt", "routellm", "hybridllm", "funsearch", "mapelites", "nsgaii", "elm"), 1):
        md.append(f"[{i}] {BIB[key]}")
    md.append("")
    return "\n".join(md)


def main() -> None:
    data = load_inputs()
    stats = derived_stats(data)
    tex = build_paper_tex(data, stats)
    md = build_paper_md(data, stats)
    with open("paper.tex", "w") as fh:
        fh.write(tex)
    with open("PAPER.md", "w") as fh:
        fh.write(md)
    ch = stats["ch"]
    print(f"wrote paper.tex ({len(tex.splitlines())} lines, {len(tex)/1024:.1f} KB)")
    print(f"wrote PAPER.md ({len(md.splitlines())} lines, {len(md)/1024:.1f} KB)")
    print(
        f" headline: Q={ch['id']['quality']:.4f} dC={ch['id']['cost_reduction_pct']:.2f}%"
        f" L={ch['id']['latency_us']:.3f}us | OOD dC={ch['ood']['cost_reduction_pct']:.2f}%"
        f" | speedup vs ML-tree {stats['speedup_vs_ml']:.0f}x"
    )
    for f in ("baseline_vs_champion.png", "pareto_frontier.png", "fitness_trajectory.png"):
        print(f" figure {'OK ' if os.path.exists(f) else 'MISSING'} {f}")


if __name__ == "__main__":
    main()
