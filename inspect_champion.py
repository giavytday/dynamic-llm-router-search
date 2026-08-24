"""Inspect top-3 champions: lineage, mutation thought, full source, train/OOD metrics."""

import sqlite3
from typing import List

from dataset import QueryRecord, generate_synthetic_dataset, split_dataset
from oracle import EvaluationOracle
from run import build_ood

DB_PATH = "evolution_search.db"
WIDTH = 78


def load_top(db_path: str, k: int = 3):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(
            "SELECT * FROM candidates WHERE passed = 1 ORDER BY fitness DESC LIMIT ?",
            (k,),
        ).fetchall()
    finally:
        con.close()


def main() -> None:
    rows = load_top(DB_PATH, 3)
    if not rows:
        raise SystemExit("no evaluated candidates found in database")
    records = generate_synthetic_dataset(10000, 42)
    train, test = split_dataset(records, 0.6)
    ood = build_ood(test)
    oracle = EvaluationOracle(train)

    print(f"TOP {len(rows)} CHAMPIONS · {DB_PATH}")
    for rank, row in enumerate(rows, 1):
        ood_res = oracle.evaluate_on_records(row["code"], ood)
        ood_fitness = oracle.fitness(
            ood_res.quality, ood_res.cost_reduction_pct, ood_res.latency_us
        )
        print("\n" + "=" * WIDTH)
        print(
            f" RANK #{rank} · hash={row['code_hash'][:8]} · fitness={row['fitness']:.4f}"
            f" · gen={row['generation']} · island={row['island_id']}"
        )
        print("=" * WIDTH)
        print(f" origin      : {row['origin']}")
        print(f" operation   : {row['operation']}")
        print(f" parent_hash : {row['parent_hash']}")
        print(f" thought     : {row['thought']}")
        print("-" * WIDTH)
        print(" policy source:")
        print("-" * WIDTH)
        print(row["code"].rstrip())
        print("-" * WIDTH)
        print(f" {'metric':<16}| {'train (n=%d)' % len(train):<22}| ood (n=%d)" % len(ood))
        print(" " + "-" * 60)
        print(
            f" {'quality Q':<16}| {row['quality']:<22.6f}| {ood_res.quality:.6f}"
        )
        print(
            f" {'cost reduction':<16}| {row['cost_reduction_pct']:<21.6f}|"
            f" {ood_res.cost_reduction_pct:.6f}"
        )
        print(f" {'latency_us':<16}| {row['latency_us']:<22.6f}| {ood_res.latency_us:.6f}")
        print(f" {'fitness':<16}| {row['fitness']:<22.4f}| {ood_fitness:.4f}")


if __name__ == "__main__":
    main()
