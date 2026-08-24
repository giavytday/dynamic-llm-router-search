"""Safety validation, smoke testing, and benchmark scoring for candidate routing policies."""

import ast
import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from dataset import COST_PER_M_TOKENS, MODEL_TIERS, QueryRecord

MAX_SOURCE_CHARS = 4000
MAX_AST_NODES = 300

FORBIDDEN_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "os",
        "sys",
        "subprocess",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "super",
        "memoryview",
    }
)

SAFE_BUILTINS: Dict[str, object] = {
    "len": len,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "range": range,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
    "sorted": sorted,
    "sum": sum,
    "enumerate": enumerate,
    "zip": zip,
}


def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


class SecurityASTVisitor(ast.NodeVisitor):
    """Rejects imports, exec/global statements, dangerous identifiers, and dunder access."""

    def __init__(self) -> None:
        self.errors: List[str] = []

    def _flag(self, node: ast.AST, why: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {why}")

    def visit_Import(self, node: ast.Import) -> None:
        self._flag(node, "forbidden import statement")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._flag(node, "forbidden import-from statement")

    def visit_Global(self, node: ast.Global) -> None:
        self._flag(node, "forbidden global statement")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._flag(node, "forbidden nonlocal statement")

    def visit_Exec(self, node: ast.AST) -> None:
        self._flag(node, "forbidden exec node")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self._flag(node, f"forbidden identifier '{node.id}'")
        elif node.id.startswith("__"):
            self._flag(node, f"dunder identifier '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self._flag(node, f"dunder attribute access '.{node.attr}'")
        self.generic_visit(node)


def validate_source(code: str) -> Tuple[bool, Optional[str]]:
    if len(code) > MAX_SOURCE_CHARS:
        return False, f"source too large ({len(code)} > {MAX_SOURCE_CHARS} chars)"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc.msg} (line {exc.lineno})"
    if len(list(ast.walk(tree))) > MAX_AST_NODES:
        return False, f"AST too large (> {MAX_AST_NODES} nodes)"
    visitor = SecurityASTVisitor()
    visitor.visit(tree)
    if visitor.errors:
        return False, "; ".join(visitor.errors[:3])
    return True, None


def boundary_feature_sets() -> List[Dict[str, object]]:
    base: Dict[str, object] = {
        "query_id": "boundary",
        "query_length_chars": 0.0,
        "estimated_tokens": 0.0,
        "has_code_syntax": False,
        "has_math_symbols": False,
        "detected_language": "text",
        "readability_score": 50.0,
        "semantic_density": 0.5,
        "budget_remaining_ratio": 1.0,
        "latency_deadline_ms": 800.0,
    }
    cases: List[Dict[str, object]] = [
        dict(base),
        {**base, "estimated_tokens": 200000.0, "query_length_chars": 800000.0},
        {**base, "has_math_symbols": True},
        {**base, "has_code_syntax": True, "detected_language": "python"},
        {
            **base,
            "has_code_syntax": True,
            "has_math_symbols": True,
            "detected_language": "python",
            "estimated_tokens": 64.0,
        },
        {**base, "readability_score": 0.0, "semantic_density": 0.0},
        {
            **base,
            "readability_score": 100.0,
            "semantic_density": 1.0,
            "budget_remaining_ratio": 0.0,
        },
        {
            **base,
            "estimated_tokens": 1.0,
            "latency_deadline_ms": 200.0,
            "budget_remaining_ratio": 0.001,
        },
    ]
    return cases


class BenchmarkError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class EvaluationResult:
    code_hash: str = ""
    passed: bool = False
    failure_reason: Optional[str] = None
    gate1_passed: bool = False
    gate2_passed: bool = False
    gate3_passed: bool = False
    quality: float = 0.0
    cost_reduction_pct: float = 0.0
    mean_cost: float = 0.0
    latency_us: float = 0.0
    fitness: float = -1e9
    n_queries: int = 0


class EvaluationOracle:
    """Three-gate evaluator: (1) AST safety, (2) smoke tests, (3) vectorized benchmark."""

    def __init__(
        self,
        train_records: Sequence[QueryRecord],
        latency_limit_us: float = 100.0,
        latency_hard_penalty: float = 300.0,
    ) -> None:
        self.records: List[QueryRecord] = list(train_records)
        self.features: List[Dict[str, object]] = [r.to_features() for r in self.records]
        self.quality_lut: Dict[str, List[float]] = {
            m: [getattr(r, f"quality_{m}") for r in self.records] for m in MODEL_TIERS
        }
        self.cost_lut: Dict[str, List[float]] = {
            m: [self.query_cost(r, m) for r in self.records] for m in MODEL_TIERS
        }
        n = max(1, len(self.records))
        self.reference_cost = sum(self.cost_lut["frontier"]) / n
        self.latency_limit_us = latency_limit_us
        self.latency_hard_penalty = latency_hard_penalty
        self.boundaries = boundary_feature_sets()

    @staticmethod
    def query_cost(record: QueryRecord, tier: str) -> float:
        tariff = COST_PER_M_TOKENS[tier]
        return (
            record.estimated_tokens * tariff["input"]
            + record.output_tokens * tariff["output"]
        ) / 1e6

    @staticmethod
    def compile_candidate(code: str):
        ok, reason = validate_source(code)
        if not ok:
            return None, reason
        try:
            compiled = compile(code, "<routing-policy>", "exec")
        except (ValueError, SyntaxError, TypeError) as exc:
            return None, f"compile failed: {exc}"
        namespace: Dict[str, object] = {"__builtins__": dict(SAFE_BUILTINS)}
        try:
            exec(compiled, namespace)  # noqa: S102 - sandboxed via Gate 1 + restricted builtins
        except Exception as exc:
            return None, f"exec failed: {type(exc).__name__}: {exc}"
        fn = namespace.get("select_model")
        if not callable(fn):
            return None, "no callable 'select_model' defined"
        return fn, None

    def _gate2_smoke(self, fn) -> Tuple[bool, Optional[str]]:
        for i, case in enumerate(self.boundaries):
            try:
                out = fn(dict(case))
            except Exception as exc:
                return False, f"smoke case {i} raised {type(exc).__name__}: {exc}"
            if not isinstance(out, str) or out not in MODEL_TIERS:
                return False, f"smoke case {i} returned invalid tier {out!r}"
        return True, None

    def _benchmark_vectorized(self, fn, features, quality_lut, cost_lut):
        total_ns = 0
        q_sum = 0.0
        c_sum = 0.0
        n = 0
        for idx, feat in enumerate(features):
            local = dict(feat)
            t0 = time.perf_counter_ns()
            try:
                tier = fn(local)
            except Exception as exc:
                raise BenchmarkError(f"runtime error on query {idx}: {type(exc).__name__}: {exc}")
            t1 = time.perf_counter_ns()
            if not isinstance(tier, str) or tier not in MODEL_TIERS:
                raise BenchmarkError(f"invalid routing decision {tier!r} on query {idx}")
            total_ns += t1 - t0
            q_sum += quality_lut[tier][idx]
            c_sum += cost_lut[tier][idx]
            n += 1
        latency_us = total_ns / max(1, n) / 1000.0
        return q_sum / n, c_sum / n, latency_us, n

    def fitness(self, quality: float, cost_reduction_pct: float, latency_us: float) -> float:
        score = 100.0 * quality + 0.6 * cost_reduction_pct - 0.05 * latency_us
        if latency_us > self.latency_limit_us:
            score -= self.latency_hard_penalty
        return score

    def evaluate_on_records(self, code: str, records: Sequence[QueryRecord]) -> EvaluationResult:
        result = EvaluationResult()
        fn, reason = self.compile_candidate(code)
        if fn is None:
            result.failure_reason = reason
            return result
        result.code_hash = hash_code(code)
        result.gate1_passed = True
        smoke_ok, smoke_reason = self._gate2_smoke(fn)
        result.gate2_passed = smoke_ok
        if not smoke_ok:
            result.failure_reason = smoke_reason
            return result
        recs = list(records)
        features = [r.to_features() for r in recs]
        quality_lut = {m: [getattr(r, f"quality_{m}") for r in recs] for m in MODEL_TIERS}
        cost_lut = {m: [self.query_cost(r, m) for r in recs] for m in MODEL_TIERS}
        try:
            quality, mean_cost, latency_us, n = self._benchmark_vectorized(
                fn, features, quality_lut, cost_lut
            )
        except BenchmarkError as exc:
            result.failure_reason = exc.reason
            return result
        result.gate3_passed = True
        result.passed = True
        result.quality = quality
        result.mean_cost = mean_cost
        result.cost_reduction_pct = (
            100.0 * (self.reference_cost - mean_cost) / self.reference_cost
            if self.reference_cost > 0
            else 0.0
        )
        result.latency_us = latency_us
        result.n_queries = n
        result.fitness = self.fitness(quality, result.cost_reduction_pct, latency_us)
        return result

    def evaluate(self, code: str) -> EvaluationResult:
        return self.evaluate_on_records(code, self.records)

    def evaluate_callable(self, fn, records: Sequence[QueryRecord]) -> dict:
        recs = list(records)
        features = [r.to_features() for r in recs]
        quality_lut = {m: [getattr(r, f"quality_{m}") for r in recs] for m in MODEL_TIERS}
        cost_lut = {m: [self.query_cost(r, m) for r in recs] for m in MODEL_TIERS}
        quality, mean_cost, latency_us, n = self._benchmark_vectorized(
            fn, features, quality_lut, cost_lut
        )
        cost_reduction_pct = (
            100.0 * (self.reference_cost - mean_cost) / self.reference_cost
            if self.reference_cost > 0
            else 0.0
        )
        return {
            "quality": quality,
            "cost_reduction_pct": cost_reduction_pct,
            "mean_cost": mean_cost,
            "latency_us": latency_us,
            "n_queries": n,
            "fitness": self.fitness(quality, cost_reduction_pct, latency_us),
        }
