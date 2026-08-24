"""Phase 3: systematic ablations over topology, archive, and mutation-operator axes."""

import json
import os
import time
from typing import Callable, Dict, List, Optional

from dataset import generate_synthetic_dataset, split_dataset
from engine import SearchEngine
from mutator import SemanticMutator

SEED = 2026
GENERATIONS = 10
BASELINE_DB = "evolution_search.db"


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


def _clean_db(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)


def run_ablation(key: str, cfg: dict, train, verbose: bool = True) -> dict:
    db_path = f"ablation_{key}.db"
    _clean_db(db_path)
    engine_kwargs = dict(
        num_islands=cfg["num_islands"],
        offspring_per_island=cfg["offspring_per_island"],
        migrate_every=cfg["migrate_every"],
    )
    if cfg.get("use_grid") is not None or "use_grid" in cfg:
        engine_kwargs["use_grid"] = cfg.get("use_grid", True)
    if "grid_parents" in cfg:
        engine_kwargs["grid_parents"] = cfg["grid_parents"]
    if cfg.get("mutator") == "random":
        engine_kwargs["mutator"] = RandomPerturbationMutator(seed=SEED + 13)

    engine = SearchEngine(
        train_records=train,
        db_path=db_path,
        seed=SEED,
        population_size=14,
        **engine_kwargs,
    )
    if verbose:
        engine.on_generation = lambda s: print(
            f"   [{key}] g{s['generation']:02d} bestFit={s['best_fitness']:7.2f}"
            f" Q={s['best_quality']:.3f} dC={s['best_cost_reduction']:5.1f}%"
            f" cells={s['grid_cells']:2d} arch={s['archive']:3d}"
        )
    engine.seed_population()
    logs = engine.search(GENERATIONS)

    curve = [round(log["best_fitness"], 4) for log in logs]
    final_best = max(curve)
    thr95 = 0.95 * final_best
    gen_to_95 = next((g for g, v in zip(range(1, GENERATIONS + 1), curve) if v >= thr95), 11)
    baseline_fitness = max(
        c.fitness for c in engine.archive.values() if c.origin == "baseline"
    )
    gen_to_beat = next(
        (g for g, v in zip(range(1, GENERATIONS + 1), curve) if v > baseline_fitness), 11
    )
    frontier = engine.pareto_frontier()
    return {
        "key": key,
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


def _print_curves(runs: Dict[str, dict], keys: List[str], title: str) -> None:
    print(f"\n  elite fitness curves · {title}")
    header = "   gen |" + "".join(f" {k[:18]:>20}" for k in keys)
    print(header)
    print("   " + "-" * (len(header) - 3))
    for g in range(1, GENERATIONS + 1):
        row = f"   {g:>3} |"
        for k in keys:
            row += f" {runs[k]['best_fitness_curve'][g - 1]:>20.2f}"
        print(row)


def main() -> None:
    t0 = time.perf_counter()
    records = generate_synthetic_dataset(10000, 42)
    train, _ = split_dataset(records, 0.6)
    print(f"ABLATION STUDY · {GENERATIONS} generations · seed={SEED} · train n={len(train)}")
    print(" budget-controlled: every variant evaluates 24 offspring per generation\n")

    runs: Dict[str, dict] = {}
    for key, cfg in RUNS.items():
        print(f" running {key} ({cfg['axis']} · {cfg['config_label']})")
        runs[key] = run_ablation(key, cfg, train)

    full = runs["full_island_semantic"]

    def delta(a: dict, b: dict) -> dict:
        return {
            "final_fitness_delta": round(b["final_best_fitness"] - a["final_best_fitness"], 4),
            "gen_to_95_delta": b["gen_to_95pct_final"] - a["gen_to_95pct_final"],
            "winner": b["key"] if b["final_best_fitness"] > a["final_best_fitness"] else a["key"],
        }

    comparisons = {
        "A_topology_single_vs_islands": delta(runs["A_single_population"], full),
        "B_archive_greedy_vs_mapelites": delta(runs["B_greedy_archive"], runs["B_map_elites"]),
        "C_operators_random_vs_semantic": delta(runs["C_random_perturbation"], full),
    }

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": SEED,
            "generations": GENERATIONS,
            "evals_per_generation": 24,
            "n_train": len(train),
            "population_size": 14,
            "convergence_metric": "first generation reaching 95% of run-final best fitness",
        },
        "runs": list(runs.values()),
        "comparisons": comparisons,
    }
    with open("ablation_table.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nsaved ablation_table.json")

    _print_curves(runs, ["A_single_population", "full_island_semantic"], "Ablation A: topology")
    _print_curves(runs, ["B_greedy_archive", "B_map_elites"], "Ablation B: archive")
    _print_curves(runs, ["C_random_perturbation", "full_island_semantic"], "Ablation C: operators")

    print("\n**Table 2: Ablation Results.** 10 generations, 24 evals/gen, seed=2026.\n")
    print(
        "| Axis | Variant | Final Fitness | Gen→95% | Gen>Base | Frontier |E| | Grid Cells | Archive |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for axis, label, key in TABLE_ROWS:
        r = runs[key]
        print(
            f"| {axis} | {label} | {r['final_best_fitness']:.2f} | {r['gen_to_95pct_final']}"
            f" | {r['gen_to_beat_baseline']} | {r['frontier_size']} | {r['grid_cells']}"
            f" | {r['archive_size']} |"
        )
    print("\nPairwise deltas:")
    for name, d in comparisons.items():
        print(
            f" {name}: final-fitness delta = {d['final_fitness_delta']:+.2f},"
            f" gen→95% delta = {d['gen_to_95_delta']:+d} (winner: {d['winner']})"
        )
    print(f"\n total wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
