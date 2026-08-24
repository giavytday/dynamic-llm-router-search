"""Island-model MAP-Elites evolutionary search with SQLite lineage tracking and Pareto output."""

import os
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from dataset import QueryRecord
from mutator import INITIAL_BASELINE_CODE, SemanticMutator
from oracle import EvaluationOracle, EvaluationResult, hash_code

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code_hash TEXT UNIQUE,
    generation INTEGER,
    island_id INTEGER,
    parent_hash TEXT,
    origin TEXT,
    operation TEXT,
    thought TEXT,
    code TEXT,
    gate1 INTEGER,
    gate2 INTEGER,
    gate3 INTEGER,
    passed INTEGER,
    failure_reason TEXT,
    quality REAL,
    cost_reduction_pct REAL,
    mean_cost REAL,
    latency_us REAL,
    fitness REAL,
    cell_q INTEGER,
    cell_c INTEGER,
    is_pareto INTEGER DEFAULT 0,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS generations (
    generation INTEGER PRIMARY KEY,
    evaluated INTEGER,
    passed INTEGER,
    rejected INTEGER,
    best_fitness REAL,
    best_quality REAL,
    best_cost_reduction REAL,
    mean_fitness REAL,
    best_hash TEXT,
    ts TEXT
);
"""


@dataclass
class Candidate:
    code: str
    code_hash: str
    thought: str
    origin: str
    operation: str
    island_id: int
    generation: int
    parent_hash: Optional[str]
    quality: float = 0.0
    cost_reduction_pct: float = 0.0
    latency_us: float = 0.0
    fitness: float = -1e9
    cell: Optional[Tuple[int, int]] = None


class SearchEngine:
    def __init__(
        self,
        train_records: List[QueryRecord],
        db_path: str = "evolution_search.db",
        seed: int = 2026,
        num_islands: int = 3,
        population_size: int = 14,
        grid_bins: int = 6,
        offspring_per_island: int = 8,
        migrate_every: int = 2,
        quality_floor: float = 0.40,
        quality_ceiling: float = 0.95,
        use_grid: bool = True,
        grid_parents: bool = False,
        mutator: Optional[SemanticMutator] = None,
    ) -> None:
        self.oracle = EvaluationOracle(train_records)
        self.mutator = mutator if mutator is not None else SemanticMutator(seed=seed + 13)
        self.use_grid = use_grid
        self.grid_parents = grid_parents
        self.db_path = db_path
        self.num_islands = num_islands
        self.population_size = population_size
        self.grid_bins = grid_bins
        self.offspring_per_island = offspring_per_island
        self.migrate_every = migrate_every
        self.q_floor = quality_floor
        self.q_ceiling = quality_ceiling
        self.islands: List[List[Candidate]] = [[] for _ in range(num_islands)]
        self.grid: Dict[Tuple[int, int], Candidate] = {}
        self.archive: Dict[str, Candidate] = {}
        self.rng = random.Random(seed)
        self.generation_logs: List[dict] = []
        self.on_generation: Optional[Callable[[dict], None]] = None
        self._init_db()

    def _init_db(self) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(_SCHEMA)
            con.commit()
        finally:
            con.close()

    def _cell_of(self, quality: float, cost_reduction_pct: float) -> Tuple[int, int]:
        span = max(1e-9, self.q_ceiling - self.q_floor)
        q_frac = max(0.0, min(0.999999, (quality - self.q_floor) / span))
        c_frac = max(0.0, min(0.999999, cost_reduction_pct / 100.0))
        return (int(q_frac * self.grid_bins), int(c_frac * self.grid_bins))

    def _persist(
        self,
        res: EvaluationResult,
        code: str,
        thought: str,
        origin: str,
        operation: str,
        island_id: int,
        generation: int,
        parent_hash: Optional[str],
        cell: Optional[Tuple[int, int]],
    ) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(
                """INSERT OR REPLACE INTO candidates
                (code_hash, generation, island_id, parent_hash, origin, operation, thought, code,
                 gate1, gate2, gate3, passed, failure_reason, quality, cost_reduction_pct,
                 mean_cost, latency_us, fitness, cell_q, cell_c, is_pareto, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)""",
                (
                    res.code_hash,
                    generation,
                    island_id,
                    parent_hash,
                    origin,
                    operation,
                    thought,
                    code,
                    int(res.gate1_passed),
                    int(res.gate2_passed),
                    int(res.gate3_passed),
                    int(res.passed),
                    res.failure_reason,
                    res.quality,
                    res.cost_reduction_pct,
                    res.mean_cost,
                    res.latency_us,
                    res.fitness,
                    cell[0] if cell else None,
                    cell[1] if cell else None,
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
            con.commit()
        finally:
            con.close()

    def evaluate_and_register(
        self,
        code: str,
        thought: str,
        origin: str,
        operation: str,
        island_id: int,
        generation: int,
        parent_hash: Optional[str] = None,
    ) -> Optional[Candidate]:
        res = self.oracle.evaluate(code)
        cell = (
            self._cell_of(res.quality, res.cost_reduction_pct)
            if res.passed
            else None
        )
        if res.code_hash not in self.archive:
            self._persist(res, code, thought, origin, operation, island_id, generation, parent_hash, cell)
        if not res.passed:
            return None
        candidate = Candidate(
            code=code,
            code_hash=res.code_hash,
            thought=thought,
            origin=origin,
            operation=operation,
            island_id=island_id,
            generation=generation,
            parent_hash=parent_hash,
            quality=res.quality,
            cost_reduction_pct=res.cost_reduction_pct,
            latency_us=res.latency_us,
            fitness=res.fitness,
            cell=cell,
        )
        self.archive.setdefault(candidate.code_hash, candidate)
        if self.use_grid:
            slot = self.grid.get(cell)
            if slot is None or candidate.fitness > slot.fitness:
                self.grid[cell] = candidate
        island = self.islands[island_id]
        if all(c.code_hash != candidate.code_hash for c in island):
            island.append(candidate)
        return candidate

    def seed_population(self) -> None:
        for i in range(self.num_islands):
            self.evaluate_and_register(
                INITIAL_BASELINE_CODE,
                "Hand-written heuristic baseline.",
                "baseline",
                "seed",
                i,
                0,
                None,
            )
            thought, code, op = self.mutator.random_policy()
            self.evaluate_and_register(code, thought, "template", op, i, 0, None)
            self._prune(i)

    def _tournament(self, pool: List[Candidate]) -> Optional[Candidate]:
        if self.grid_parents and self.grid:
            elites = list(self.grid.values())
            contenders = self.rng.sample(elites, min(3, len(elites)))
            return max(contenders, key=lambda c: c.fitness)
        if not pool:
            return None
        contenders = self.rng.sample(pool, min(3, len(pool)))
        return max(contenders, key=lambda c: c.fitness)

    def _prune(self, island_id: int) -> None:
        ranked = sorted(
            self.islands[island_id],
            key=lambda c: c.fitness,
            reverse=True,
        )
        kept: List[Candidate] = []
        seen = set()
        for cand in ranked:
            if cand.code_hash in seen:
                continue
            seen.add(cand.code_hash)
            kept.append(cand)
            if len(kept) >= self.population_size:
                break
        self.islands[island_id] = kept

    def _migrate(self) -> None:
        for i in range(self.num_islands):
            src = sorted(self.islands[i], key=lambda c: c.fitness, reverse=True)
            dst = self.islands[(i + 1) % self.num_islands]
            if src and all(c.code_hash != src[0].code_hash for c in dst):
                dst.append(src[0])

    def search(self, generations: int = 10) -> List[dict]:
        for gen in range(1, generations + 1):
            evaluated = 0
            passed = 0
            fits: List[float] = []
            best: Optional[Candidate] = None
            for isl in range(self.num_islands):
                ranked = sorted(self.islands[isl], key=lambda c: c.fitness, reverse=True)
                for _ in range(self.offspring_per_island):
                    if ranked and self.rng.random() < 0.15:
                        thought, code, op = self.mutator.random_policy()
                        parent_hash = None
                        origin = "template"
                    else:
                        parent = self._tournament(ranked)
                        co_parent = (
                            self._tournament(ranked)
                            if parent and self.rng.random() < 0.30
                            else None
                        )
                        thought, code, op = self.mutator.mutate(
                            parent.code if parent else "",
                            co_parent.code if co_parent else None,
                        )
                        parent_hash = parent.code_hash if parent else None
                        origin = "mutation"
                    cand = self.evaluate_and_register(
                        code, thought, origin, op, isl, gen, parent_hash
                    )
                    evaluated += 1
                    if cand is not None:
                        passed += 1
                        fits.append(cand.fitness)
                        if best is None or cand.fitness > best.fitness:
                            best = cand
                self._prune(isl)
            if gen % self.migrate_every == 0:
                self._migrate()
            summary = {
                "generation": gen,
                "evaluated": evaluated,
                "passed": passed,
                "rejected": evaluated - passed,
                "mean_fitness": sum(fits) / len(fits) if fits else 0.0,
                "best_fitness": best.fitness if best else 0.0,
                "best_quality": best.quality if best else 0.0,
                "best_cost_reduction": best.cost_reduction_pct if best else 0.0,
                "best_hash": best.code_hash if best else "-",
                "best_op": best.operation if best else "-",
                "grid_cells": len(self.grid),
                "archive": len(self.archive),
            }
            self._persist_generation(summary)
            self.generation_logs.append(summary)
            if self.on_generation:
                self.on_generation(summary)
        return self.generation_logs

    def _persist_generation(self, s: dict) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(
                """INSERT OR REPLACE INTO generations
                (generation, evaluated, passed, rejected, best_fitness, best_quality,
                 best_cost_reduction, mean_fitness, best_hash, ts)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    s["generation"],
                    s["evaluated"],
                    s["passed"],
                    s["rejected"],
                    s["best_fitness"],
                    s["best_quality"],
                    s["best_cost_reduction"],
                    s["mean_fitness"],
                    s["best_hash"],
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
            con.commit()
        finally:
            con.close()

    def pareto_frontier(self) -> List[Candidate]:
        candidates = list(self.archive.values())
        front: List[Candidate] = []
        for cand in candidates:
            dominated = any(
                other is not cand
                and other.quality >= cand.quality
                and other.cost_reduction_pct >= cand.cost_reduction_pct
                and (
                    other.quality > cand.quality
                    or other.cost_reduction_pct > cand.cost_reduction_pct
                )
                for other in candidates
            )
            if not dominated:
                front.append(cand)
        front.sort(key=lambda c: c.fitness, reverse=True)
        self._mark_pareto([c.code_hash for c in front])
        return front

    def _mark_pareto(self, hashes: List[str]) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("UPDATE candidates SET is_pareto = 0")
            con.executemany(
                "UPDATE candidates SET is_pareto = 1 WHERE code_hash = ?",
                [(h,) for h in hashes],
            )
            con.commit()
        finally:
            con.close()

    def champion(self) -> Optional[Candidate]:
        if not self.archive:
            return None
        return max(self.archive.values(), key=lambda c: c.fitness)

    def db_stats(self) -> dict:
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            total = cur.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            n_passed = cur.execute("SELECT COUNT(*) FROM candidates WHERE passed=1").fetchone()[0]
            rejected = cur.execute("SELECT COUNT(*) FROM candidates WHERE passed=0").fetchone()[0]
            gens = cur.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
            pareto_rows = cur.execute("SELECT COUNT(*) FROM candidates WHERE is_pareto=1").fetchone()[0]
        finally:
            con.close()
        size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        return {
            "path": self.db_path,
            "size_kb": size / 1024.0,
            "candidates_total": total,
            "candidates_passed": n_passed,
            "candidates_rejected": rejected,
            "generations": gens,
            "pareto_marked": pareto_rows,
            "grid_cells": len(self.grid),
        }
