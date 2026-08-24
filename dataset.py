"""Synthetic query corpus generation for dynamic multi-model LLM routing research."""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

COST_PER_M_TOKENS: Dict[str, Dict[str, float]] = {
    "small": {"input": 0.15, "output": 0.60},
    "medium": {"input": 0.80, "output": 3.20},
    "frontier": {"input": 3.00, "output": 15.00},
}

MODEL_TIERS: Tuple[str, ...] = ("small", "medium", "frontier")

_TASK_WEIGHTS: Tuple[Tuple[str, float], ...] = (
    ("chat", 0.40),
    ("code", 0.25),
    ("math", 0.20),
    ("analysis", 0.15),
)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class QueryRecord:
    query_id: str
    query_length_chars: int
    estimated_tokens: int
    has_code_syntax: bool
    has_math_symbols: bool
    detected_language: str
    readability_score: float
    semantic_density: float
    budget_remaining_ratio: float
    latency_deadline_ms: int
    quality_small: float
    quality_medium: float
    quality_frontier: float
    output_tokens: int

    def to_features(self) -> Dict[str, object]:
        return {
            "query_id": self.query_id,
            "query_length_chars": float(self.query_length_chars),
            "estimated_tokens": float(self.estimated_tokens),
            "has_code_syntax": bool(self.has_code_syntax),
            "has_math_symbols": bool(self.has_math_symbols),
            "detected_language": self.detected_language,
            "readability_score": float(self.readability_score),
            "semantic_density": float(self.semantic_density),
            "budget_remaining_ratio": float(self.budget_remaining_ratio),
            "latency_deadline_ms": float(self.latency_deadline_ms),
        }


def generate_synthetic_dataset(num_samples: int = 10000, seed: int = 42) -> List[QueryRecord]:
    rng = random.Random(seed)
    tasks = [t for t, _ in _TASK_WEIGHTS]
    weights = [p for _, p in _TASK_WEIGHTS]
    records: List[QueryRecord] = []
    for i in range(num_samples):
        task = rng.choices(tasks, weights=weights, k=1)[0]
        if task == "chat":
            chars = int(rng.lognormvariate(5.7, 0.7))
        elif task == "code":
            chars = int(rng.lognormvariate(6.9, 0.55))
        elif task == "math":
            chars = int(rng.lognormvariate(5.9, 0.75))
        else:
            chars = int(rng.lognormvariate(7.0, 0.5))
        chars = max(48, min(15000, chars))
        tokens = max(12, int(chars / rng.uniform(3.4, 4.6)))
        has_code = task == "code" or rng.random() < 0.06
        has_math = task == "math" or (task == "analysis" and rng.random() < 0.35) or rng.random() < 0.03
        if task == "code":
            language = "python" if rng.random() < 0.62 else "javascript"
        elif task == "analysis" and rng.random() < 0.18:
            language = "sql"
        elif has_math:
            language = "latex"
        else:
            language = "text"
        if task == "chat":
            difficulty = rng.uniform(0.42, 0.74)
        elif task == "code":
            difficulty = rng.uniform(0.55, 0.97)
        elif task == "math":
            difficulty = rng.uniform(0.66, 0.99)
        else:
            difficulty = rng.uniform(0.48, 0.86)
        q_small = _clip01(difficulty * rng.uniform(0.52, 0.78) + rng.gauss(0, 0.05))
        q_medium = _clip01(difficulty * rng.uniform(0.76, 0.91) + rng.gauss(0, 0.035) + 0.03)
        q_frontier = _clip01(difficulty * rng.uniform(0.90, 0.985) + rng.gauss(0, 0.02) + 0.07)
        readability = _clip01(rng.gauss(58, 14) - (8 if has_code else 0) + (6 if language == "text" else 0))
        density = _clip01(
            rng.uniform(0.08, 0.50)
            + (0.32 if has_math else 0.0)
            + (0.12 if task == "analysis" else 0.0)
            + rng.gauss(0, 0.05)
        )
        budget = round(rng.random(), 3)
        deadline = rng.choices((200, 400, 800, 1600), weights=(0.24, 0.36, 0.28, 0.12), k=1)[0]
        out_tokens = max(16, int(tokens * rng.uniform(0.18, 0.85)) + (120 if task == "chat" else 0))
        records.append(
            QueryRecord(
                query_id=f"q_{i:06d}",
                query_length_chars=chars,
                estimated_tokens=tokens,
                has_code_syntax=has_code,
                has_math_symbols=has_math,
                detected_language=language,
                readability_score=round(readability, 2),
                semantic_density=round(density, 3),
                budget_remaining_ratio=budget,
                latency_deadline_ms=deadline,
                quality_small=round(q_small, 4),
                quality_medium=round(q_medium, 4),
                quality_frontier=round(q_frontier, 4),
                output_tokens=out_tokens,
            )
        )
    return records


def split_dataset(records: List[QueryRecord], train_ratio: float = 0.6):
    ordered = list(records)
    random.Random(20260824).shuffle(ordered)
    cut = int(len(ordered) * train_ratio)
    return ordered[:cut], ordered[cut:]
