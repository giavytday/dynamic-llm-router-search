"""Table 1: comparative benchmark of baselines vs the evolved champion policy."""

import json
import sqlite3
import subprocess
import sys
import time
from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")

from dataset import generate_synthetic_dataset, split_dataset
from engine import SearchEngine  # noqa: F401  (ensures schema compatibility on import)
from mutator import INITIAL_BASELINE_CODE
from oracle import EvaluationOracle
from run import build_ood

import baselines

DB_PATH = "evolution_search.db"
CHAMPION_HASH = "22e2a1f5"
SHORT_NAMES = {
    "Always-Frontier": "Frontier",
    "Always-Medium": "Medium",
    "Always-Small": "Small",
    "Random-Uniform": "Random",
    "Heuristic-Threshold": "Heuristic",
    "Hand-Written-Baseline": "Hand-Written",
    "ML-DecisionTree": "ML-Tree",
    "Evolved-Champion": "Champion",
}


def ensure_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("scikit-learn not found -> installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "scikit-learn"])
        except subprocess.CalledProcessError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])


def load_champion(oracle: EvaluationOracle):
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute(
            "SELECT code, generation, island_id, fitness FROM candidates WHERE code_hash LIKE ?",
            (CHAMPION_HASH + "%",),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise SystemExit(f"champion {CHAMPION_HASH} not found in {DB_PATH}")
    fn, reason = oracle.compile_candidate(row[0])
    if fn is None:
        raise SystemExit(f"champion failed to compile: {reason}")
    return fn, row[0], {"generation": row[1], "island_id": row[2], "db_fitness": row[3]}


def main() -> None:
    ensure_sklearn()
    import sklearn

    records = generate_synthetic_dataset(10000, 42)
    train, test = split_dataset(records, 0.6)
    ood = build_ood(test)
    oracle = EvaluationOracle(train)

    champion_fn, champion_code, champion_meta = load_champion(oracle)
    ml_router = baselines.ml_classifier_router(train, test)
    baseline_router_fn, _ = oracle.compile_candidate(INITIAL_BASELINE_CODE)

    methods: List[Dict[str, object]] = [
        ("Always-Frontier", baselines.always_frontier, 1),
        ("Always-Medium", baselines.always_medium, 1),
        ("Always-Small", baselines.always_small, 1),
        ("Random-Uniform", baselines.random_router, 0),
        ("Heuristic-Threshold", baselines.heuristic_threshold_router,
         baselines.count_numeric_params(baselines.heuristic_threshold_router)),
        ("Hand-Written-Baseline", baseline_router_fn,
         baselines.count_numeric_params(INITIAL_BASELINE_CODE)),
        ("ML-DecisionTree", ml_router, ml_router.n_params),
        ("Evolved-Champion", champion_fn,
         baselines.count_numeric_params(champion_code)),
    ]

    results = []
    for name, fn, n_params in methods:
        if callable(fn):
            res_id = oracle.evaluate_callable(fn, train)
            res_ood = oracle.evaluate_callable(fn, ood)
            if name == "ML-DecisionTree":
                mem_kb = ml_router.mem_bytes / 1024.0
            else:
                mem_kb = baselines.code_memory_bytes(fn) / 1024.0
        results.append(
            {
                "method": name,
                "n_params": int(n_params),
                "memory_kb": round(mem_kb, 3),
                "id": res_id,
                "ood": res_ood,
            }
        )
        print(
            f" benchmarked {name:<22} ID: Q={res_id['quality']:.4f} dC={res_id['cost_reduction_pct']:6.2f}%"
            f" L={res_id['latency_us']:8.3f}us | OOD: Q={res_ood['quality']:.4f}"
            f" dC={res_ood['cost_reduction_pct']:6.2f}%"
        )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "champion_hash": CHAMPION_HASH,
            "champion_generation": champion_meta["generation"],
            "champion_island": champion_meta["island_id"],
            "n_train": len(train),
            "n_test": len(test),
            "n_ood": len(ood),
            "sklearn_version": sklearn.__version__,
            "ml_model": "DecisionTreeClassifier(max_depth=8, min_samples_leaf=20)",
            "ml_train_accuracy": round(ml_router.train_accuracy, 4),
            "ml_test_accuracy": round(ml_router.test_accuracy, 4),
            "latency_definition": "mean wall-clock per select_model(q) call, microseconds (ID split)",
            "memory_definition": "code-object bytes for policies; pickled model bytes for sklearn",
        },
        "results": results,
    }
    with open("benchmark_results.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nsaved benchmark_results.json")

    print("\n**Table 1: Main Results.** Comparative evaluation of routing strategies")
    print(f" (ID: n={len(train)}; OOD shift: n={len(ood)}; reference cost = always-frontier).\n")
    header = (
        "| Method | Q (ID) | $\\Delta C$\\% (ID) | Q (OOD) | $\\Delta C$\\% (OOD)"
        " | $L$ (µs) | Params | Mem (KB) |"
    )
    print(header)
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['method']} | {r['id']['quality']:.4f} | {r['id']['cost_reduction_pct']:.2f}"
            f" | {r['ood']['quality']:.4f} | {r['ood']['cost_reduction_pct']:.2f}"
            f" | {r['id']['latency_us']:.3f} | {r['n_params']} | {r['memory_kb']:.2f} |"
        )

    print("\n% ---- LaTeX (booktabs) ----")
    print("\\begin{table}[t]\\centering\\small")
    print("\\caption{Main results: routing quality, cost reduction, inference latency,")
    print("and model size. ID = in-distribution train split; OOD = distribution-shifted slice.}")
    print("\\label{tab:main}")
    print("\\begin{tabular}{lrrrrrrr}\\toprule")
    print("Method & \\multicolumn{2}{c}{ID} & \\multicolumn{2}{c}{OOD} & $L$ ($\\mu$s) & Params & Mem (KB)\\\\")
    print(" & $Q$ & $\\Delta C$\\% & $Q$ & $\\Delta C$\\% & & & \\\\\\midrule")
    for r in results:
        print(
            f"{SHORT_NAMES.get(r['method'], r['method'])} & {r['id']['quality']:.4f} &"
            f" {r['id']['cost_reduction_pct']:.2f} & {r['ood']['quality']:.4f} &"
            f" {r['ood']['cost_reduction_pct']:.2f} & {r['id']['latency_us']:.3f} &"
            f" {r['n_params']} & {r['memory_kb']:.2f}\\\\"
        )
    print("\\bottomrule\\end{tabular}\\end{table}")

    chart(results, "baseline_vs_champion.png")
    print("saved baseline_vs_champion.png")


def chart(results: List[Dict[str, object]], out_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = [SHORT_NAMES[r["method"]] for r in results]
    champion_idx = names.index("Champion")
    x = np.arange(len(names))
    w = 0.38

    def bar_colors(ood: bool) -> List[str]:
        return [
            "#c62828" if i == champion_idx else ("#ef9a9a" if ood else "#90caf9")
            for i in range(len(names))
        ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    id_q = [r["id"]["quality"] for r in results]
    ood_q = [r["ood"]["quality"] for r in results]
    ax1.bar(x - w / 2, id_q, w, color=bar_colors(False), label="ID (train)")
    ax1.bar(x + w / 2, ood_q, w, color=bar_colors(True), label="OOD shift")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=28, ha="right", fontsize=9)
    ax1.set_ylabel(r"Relative Quality $Q$")
    ax1.set_ylim(0, 0.85)
    ax1.set_title("Routing Quality")
    ax1.legend(fontsize=9)

    id_c = [r["id"]["cost_reduction_pct"] for r in results]
    ood_c = [r["ood"]["cost_reduction_pct"] for r in results]
    ax2.bar(x - w / 2, id_c, w, color=bar_colors(False))
    ax2.bar(x + w / 2, ood_c, w, color=bar_colors(True))
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=28, ha="right", fontsize=9)
    ax2.set_ylabel(r"Cost Reduction $\Delta C$ (%)")
    ax2.set_title("Cost Reduction vs Always-Frontier")
    ax2.axhline(0.0, color="grey", lw=0.8)

    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Baseline Routers vs Evolved Champion (Table 1)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
