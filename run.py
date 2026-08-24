"""Main asynchronous entrypoint: dataset -> baseline -> island evolution -> Pareto/OOD report."""

import asyncio
import os
import time
from typing import List

from dataset import COST_PER_M_TOKENS, QueryRecord, generate_synthetic_dataset, split_dataset
from engine import SearchEngine
from mutator import INITIAL_BASELINE_CODE

WIDTH = 78


def rule(title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(f" {title}")
    print("=" * WIDTH)


def build_ood(test: List[QueryRecord]) -> List[QueryRecord]:
    tokens = sorted(r.estimated_tokens for r in test)
    tok_cut = tokens[int(0.90 * (len(tokens) - 1))]
    densities = sorted(r.semantic_density for r in test)
    den_cut = densities[int(0.90 * (len(densities) - 1))]
    pool = [
        r
        for r in test
        if r.estimated_tokens >= tok_cut
        or (r.has_code_syntax and r.has_math_symbols)
        or r.semantic_density >= den_cut
    ]
    return pool[:1500]


async def main() -> None:
    t_start = time.perf_counter()
    print("DYNAMIC MULTI-MODEL LLM ROUTING VIA EVOLUTIONARY CODE SEARCH")

    rule("STAGE 1 · SYNTHETIC DATASET GENERATION")
    records = await asyncio.to_thread(generate_synthetic_dataset, 10000, 42)
    train, test = split_dataset(records, 0.6)
    pct_code = 100.0 * sum(1 for r in records if r.has_code_syntax) / len(records)
    pct_math = 100.0 * sum(1 for r in records if r.has_math_symbols) / len(records)
    mean_tokens = sum(r.estimated_tokens for r in records) / len(records)
    mean_q_frontier = sum(r.quality_frontier for r in records) / len(records)
    print(f" samples={len(records)} | train={len(train)} | test={len(test)}")
    print(
        f" mean_tokens={mean_tokens:.0f} | code={pct_code:.1f}% | math={pct_math:.1f}%"
        f" | E[quality_frontier]={mean_q_frontier:.3f}"
    )
    tariffs = " | ".join(
        f"{k}: ${v['input']:.2f}in/${v['output']:.2f}out per 1M"
        for k, v in COST_PER_M_TOKENS.items()
    )
    print(f" cost table: {tariffs}")

    engine = SearchEngine(train_records=train, db_path="evolution_search.db", seed=2026)

    rule("STAGE 2 · BASELINE EVALUATION (GATES 1-3)")
    baseline = await asyncio.to_thread(engine.oracle.evaluate, INITIAL_BASELINE_CODE)
    print(
        f" gates: AST={'PASS' if baseline.gate1_passed else 'FAIL'}"
        f" SMOKE={'PASS' if baseline.gate2_passed else 'FAIL'}"
        f" BENCH={'PASS' if baseline.gate3_passed else 'FAIL'}"
    )
    print(
        f" baseline: Q={baseline.quality:.4f}"
        f" | mean_cost=${baseline.mean_cost:.6f}/query"
        f" | dC={baseline.cost_reduction_pct:.2f}%"
        f" | latency={baseline.latency_us:.3f}us"
        f" | fitness={baseline.fitness:.2f}"
    )
    if not baseline.passed:
        raise SystemExit("baseline failed evaluation: " + str(baseline.failure_reason))

    rule("STAGE 3 · EVOLUTIONARY SEARCH · 3 ISLANDS × 10 GENERATIONS")

    def on_generation(s: dict) -> None:
        print(
            f" g{s['generation']:02d} eval={s['evaluated']:3d} pass={s['passed']:3d}"
            f" rej={s['rejected']:2d} | bestFit={s['best_fitness']:7.2f}"
            f" Q={s['best_quality']:.3f} dC={s['best_cost_reduction']:5.1f}%"
            f" | {s['best_hash'][:8]} {s['best_op']:<16}"
            f" | cells={s['grid_cells']:2d} arch={s['archive']:3d}"
        )

    engine.on_generation = on_generation
    await asyncio.to_thread(engine.seed_population)
    logs = await asyncio.to_thread(engine.search, 10)
    champion = engine.champion()
    assert champion is not None
    print(
        f"\n champion fitness={champion.fitness:.2f} vs baseline={baseline.fitness:.2f}"
        f" (uplift +{champion.fitness - baseline.fitness:.2f})"
    )

    rule("STAGE 4 · PARETO FRONTIER OUTPUT")
    pareto = await asyncio.to_thread(engine.pareto_frontier)
    print(
        f" {'rank':<5}{'hash':<10}{'gen':>4}{'isl':>4}  {'operation':<18}"
        f"{'Q':>8}{'dC%':>9}{'lat_us':>9}{'fitness':>10}"
    )
    print(" " + "-" * 71)
    for i, cand in enumerate(pareto, 1):
        print(
            f" {i:<5}{cand.code_hash[:8]:<10}{cand.generation:>4}{cand.island_id:>4}"
            f"  {cand.operation:<18}{cand.quality:>8.4f}{cand.cost_reduction_pct:>8.1f}%"
            f"{cand.latency_us:>9.3f}{cand.fitness:>10.2f}"
        )
    print(f"\n pareto size: {len(pareto)} non-dominated policies (maximize Q, maximize dC)")

    rule("STAGE 5 · OUT-OF-DISTRIBUTION GENERALIZATION")
    ood = build_ood(test)
    print(f" ood slice: {len(ood)} queries (top-decile tokens / code+math / density)")
    header = (
        f" {'policy':<10}{'Q_test':>8}{'dC_test':>9}{'us':>7}"
        f" | {'Q_ood':>8}{'dC_ood':>9}{'us':>7} | {'dQ_gap':>8}"
    )
    print(header)
    print(" " + "-" * (len(header) - 2))

    def row(label: str, rt, ro) -> str:
        gap = ro.quality - rt.quality
        return (
            f" {label:<10}{rt.quality:>8.4f}{rt.cost_reduction_pct:>8.1f}%{rt.latency_us:>6.2f}"
            f" | {ro.quality:>8.4f}{ro.cost_reduction_pct:>8.1f}%{ro.latency_us:>6.2f}"
            f" | {gap:>+8.4f}"
        )

    bt = await asyncio.to_thread(engine.oracle.evaluate_on_records, INITIAL_BASELINE_CODE, test)
    bo = await asyncio.to_thread(engine.oracle.evaluate_on_records, INITIAL_BASELINE_CODE, ood)
    print(row("baseline", bt, bo))
    for cand in pareto:
        rt = await asyncio.to_thread(engine.oracle.evaluate_on_records, cand.code, test)
        ro = await asyncio.to_thread(engine.oracle.evaluate_on_records, cand.code, ood)
        print(row(cand.code_hash[:8], rt, ro))

    rule("VERIFICATION")
    stats = engine.db_stats()
    gens_ok = stats["generations"] == 10 and len(logs) == 10
    print(f" db_file        : {os.path.abspath(stats['path'])} ({stats['size_kb']:.1f} KB)")
    print(
        f" candidates     : {stats['candidates_total']} total"
        f" | {stats['candidates_passed']} passed"
        f" | {stats['candidates_rejected']} rejected by gates"
    )
    print(
        f" generations    : {stats['generations']}/10 logged [{'OK' if gens_ok else 'ERROR'}]"
    )
    print(f" map_elites     : {stats['grid_cells']}/36 cells occupied")
    print(f" pareto_marked  : {stats['pareto_marked']} rows flagged is_pareto=1")
    print(" lineage        : parent_hash chains rooted at baseline + grammar templates")
    if not gens_ok:
        raise SystemExit("generation log incomplete")
    print(f"\n total wall time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
