"""Phase 3: multi-seed ablation suite over topology, archive, and mutation-operator axes."""

import json
import os
import statistics
import time
from typing import Dict, List, Optional

from dataset import generate_synthetic_dataset, split_dataset
from engine import SearchEngine
from mutator import SemanticMutator

SEEDS = (42, 123, 2026, 777, 999)
GENERATIONS = 10
RUNS_DIR = "ablation_runs"


class RandomPerturbationMutator(SemanticMutator):
    """Ablation C: uniform numeric jitter only; no semantic operator guidance."""

    def mutate(self, parent_code: str, co_parent_code: Optional[str] = None):
        child = parent_code
        for _ in range(3):
            child = self._tweak_constants(parent_code)
            if child.strip() != parent_code.strip():
                break
        thought = "[random-perturbation] uniform numeric jitter; no semantic reasoning applied."
        return thought, child, "random_perturbation"


RUNS: Dict[str, dict] = {
    "A_single_population": {
        "axis": "A (topology)",
        "config_label": "Single population, no migration",
        "num_islands": 1,
        "offspring_per_island": 24,
        "migrate_every": 10**9,
    },
    "full_island_semantic": {
        "axis": "A/C reference",
        "config_label": "3-island + migration, semantic ops (full system)",
        "num_islands": 3,
        "offspring_per_island": 8,
        "migrate_every": 2,
    },
    "B_greedy_archive": {
        "axis": "B (archive)",
        "config_label": "Greedy fitness replacement, fitness parents",
        "num_islands": 3,
        "offspring_per_island": 8,
        "migrate_every": 2,
        "use_grid": False,
        "grid_parents": False,
    },
    "B_map_elites": {
        "axis": "B (archive)",
        "config_label": "2D MAP-Elites archive, elite-grid parents",
        "num_islands": 3,
        "offspring_per_island": 8,
        "migrate_every": 2,
        "use_grid": True,
        "grid_parents": True,
    },
    "C_random_perturbation": {
        "axis": "C (operators)",
        "config_label": "Random constant jitter only",
        "num_islands": 3,
        "offspring_per_island": 8,
        "migrate_every": 2,
        "mutator": "random",
    },
}

TABLE_ROWS = [
    ("A (topology)", "Single population", "A_single_population"),
    ("A (topology)", "3-island + migration (full)", "full_island_semantic"),
    ("B (archive)", "Greedy fitness replacement", "B_greedy_archive"),
    ("B (archive)", "2D MAP-Elites parents", "B_map_elites"),
    ("C (operators)", "Random constant jitter", "C_random_perturbation"),
    ("C (operators)", "Semantic thought-guided (full)", "full_island_semantic"),
]


def _clean(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)


def run_ablation(key: str, cfg: dict, train, seed: int) -> dict:
    os.makedirs(RUNS_DIR, exist_ok=True)
    db_path = os.path.join(RUNS_DIR, f"{key}_s{seed}.db")
    _clean(db_path)
    engine_kwargs: Dict[str, object] = dict(
        num_islands=cfg["num_islands"],
        offspring_per_island=cfg["offspring_per_island"],
        migrate_every=cfg["migrate_every"],
    )
    if "use_grid" in cfg:
        engine_kwargs["use_grid"] = cfg["use_grid"]
    if "grid_parents" in cfg:
        engine_kwargs["grid_parents"] = cfg["grid_parents"]
    if cfg.get("mutator") == "random":
        engine_kwargs["mutator"] = RandomPerturbationMutator(seed=seed + 13)

    engine = SearchEngine(
        train_records=train,
        db_path=db_path,
        seed=seed,
        population_size=14,
        **engine_kwargs,
    )
    engine.seed_population()
    engine.search(GENERATIONS)

    curve = [round(log["best_fitness"], 4) for log in engine.generation_logs]
    final_best = max(curve)
    thr95 = 0.95 * final_best
    gen_to_95 = next(
        (g for g, v in zip(range(1, GENERATIONS + 1), curve) if v >= thr95), GENERATIONS + 1
    )
    baseline_fitness = max(
        c.fitness for c in engine.archive.values() if c.origin == "baseline"
    )
    gen_to_beat = next(
        (g for g, v in zip(range(1, GENERATIONS + 1), curve) if v > baseline_fitness),
        GENERATIONS + 1,
    )
    frontier = engine.pareto_frontier()
    return {
        "key": key,
        "seed": seed,
        "axis": cfg["axis"],
        "config_label": cfg["config_label"],
        "db_path": db_path,
        "baseline_fitness": round(baseline_fitness, 4),
        "final_best_fitness": round(final_best, 4),
        "best_fitness_curve": curve,
        "gen_to_95pct_final": gen_to_95,
        "gen_to_beat_baseline": gen_to_beat,
        "frontier_size": len(frontier),
        "grid_cells": len(engine.grid) if engine.use_grid else 0,
        "archive_size": len(engine.archive),
        "champion_hash": engine.champion().code_hash,
    }


def _ms(values: List[float]) -> dict:
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4),
    }


def aggregate(records: List[dict]) -> dict:
    return {
        "n_seeds": len(records),
        "peak_fitness": _ms([r["final_best_fitness"] for r in records]),
        "gen_to_95pct_final": _ms([float(r["gen_to_95pct_final"]) for r in records]),
        "gen_to_beat_baseline": _ms([float(r["gen_to_beat_baseline"]) for r in records]),
        "elite_count": _ms([float(r["archive_size"]) for r in records]),
        "grid_cells": _ms([float(r["grid_cells"]) for r in records]),
        "frontier_size": _ms([float(r["frontier_size"]) for r in records]),
        "mean_curve": [
            round(statistics.mean(r["best_fitness_curve"][g] for r in records), 4)
            for g in range(GENERATIONS)
        ],
        "per_seed": {str(r["seed"]): r for r in records},
    }


def _cell(a: dict, fmt: str = "{:.2f}") -> str:
    return f"{fmt.format(a['mean'])} ± {a['std']:.2f}"


def _print_mean_curves(agg: Dict[str, dict], keys: List[str], title: str) -> None:
    print(f"\n  mean elite fitness curves across {len(SEEDS)} seeds · {title}")
    header = "   gen |" + "".join(f" {k[:18]:>20}" for k in keys)
    print(header)
    print("   " + "-" * (len(header) - 3))
    for g in range(GENERATIONS):
        row = f"   {g + 1:>3} |"
        for k in keys:
            row += f" {agg[k]['mean_curve'][g]:>20.2f}"
        print(row)


def main() -> None:
    t0 = time.perf_counter()
    records = generate_synthetic_dataset(10000, 42)
    train, _ = split_dataset(records, 0.6)
    print(
        f"ABLATION STUDY · {len(SEEDS)} seeds {list(SEEDS)} · {GENERATIONS} generations"
        f" · 24 evals/gen · train n={len(train)}"
    )
    print(f" total: {len(SEEDS) * len(RUNS)} independent runs, budget-controlled per seed\n")

    runs: Dict[str, List[dict]] = {key: [] for key in RUNS}
    for key, cfg in RUNS.items():
        for seed in SEEDS:
            rec = run_ablation(key, cfg, train, seed)
            runs[key].append(rec)
            print(
                f" [{key} s{seed}] peakFit={rec['final_best_fitness']:7.2f}"
                f" gen95={rec['gen_to_95pct_final']:2d} genBase={rec['gen_to_beat_baseline']:2d}"
                f" frontier={rec['frontier_size']:3d} cells={rec['grid_cells']:2d}"
                f" elites={rec['archive_size']:3d}"
            )

    aggregates = {key: aggregate(recs) for key, recs in runs.items()}
    full = aggregates["full_island_semantic"]

    def delta(a_key: str, b_key: str) -> dict:
        a, b = aggregates[a_key], aggregates[b_key]
        return {
            "peak_fitness_delta_mean": round(
                b["peak_fitness"]["mean"] - a["peak_fitness"]["mean"], 4
            ),
            "gen_to_95_delta_mean": round(
                b["gen_to_95pct_final"]["mean"] - a["gen_to_95pct_final"]["mean"], 4
            ),
            "winner": b_key if b["peak_fitness"]["mean"] > a["peak_fitness"]["mean"] else a_key,
        }

    comparisons = {
        "A_topology_single_vs_islands": delta(
            "A_single_population", "full_island_semantic"
        ),
        "B_archive_greedy_vs_mapelites": delta(
            "B_greedy_archive", "B_map_elites"
        ),
        "C_operators_random_vs_semantic": delta(
            "C_random_perturbation", "full_island_semantic"
        ),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seeds": list(SEEDS),
            "generations": GENERATIONS,
            "evals_per_generation": 24,
            "n_train": len(train),
            "population_size": 14,
            "convergence_metric": "first generation reaching 95% of run-final best fitness",
            "elite_count_definition": "unique passing candidates in run archive",
            "std_definition": "sample standard deviation across seeds (n-1)",
        },
        "runs": [rec for recs in runs.values() for rec in recs],
        "aggregates": aggregates,
        "comparisons": comparisons,
    }
    with open("ablation_table.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nsaved ablation_table.json")

    _print_mean_curves(aggregates, ["A_single_population", "full_island_semantic"], "Ablation A: topology")
    _print_mean_curves(aggregates, ["B_greedy_archive", "B_map_elites"], "Ablation B: archive")
    _print_mean_curves(aggregates, ["C_random_perturbation", "full_island_semantic"], "Ablation C: operators")

    print(f"\n**Table 2: Ablation Results.** mean ± std over {len(SEEDS)} seeds"
          f" {list(SEEDS)}; 10 generations, 24 evals/gen.\n")
    print(
        "| Axis | Variant | Peak Fitness | Gen→95% | Gen>Base | Elites | Cells |"
    )
    print("|---|---|---|---|---|---|---|")
    for axis, label, key in TABLE_ROWS:
        a = aggregates[key]
        print(
            f"| {axis} | {label} | {_cell(a['peak_fitness'])}"
            f" | {_cell(a['gen_to_95pct_final'], '{:.1f}')}"
            f" | {_cell(a['gen_to_beat_baseline'], '{:.1f}')}"
            f" | {_cell(a['elite_count'], '{:.1f}')}"
            f" | {_cell(a['grid_cells'], '{:.1f}')} |"
        )
    print("\nPairwise deltas (mean over seeds):")
    for name, d in comparisons.items():
        print(
            f" {name}: peak-fitness delta = {d['peak_fitness_delta_mean']:+.2f},"
            f" gen→95% delta = {d['gen_to_95_delta_mean']:+.2f} (winner: {d['winner']})"
        )
    print(f"\n total wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
