"""Comparative baseline routing policies sharing the exact select_model(q) signature."""

import inspect
import pickle
import random
import re
import sys
from typing import Callable, Dict, List

from dataset import QueryRecord

LANG_VOCAB = ("text", "python", "javascript", "sql", "latex")
_NUM_RE = re.compile(r"(?<![\w.'])(\d+\.\d+|\d+)(?![\w.'])")

_SAMPLE_FEATURES: Dict[str, object] = {
    "query_id": "s",
    "query_length_chars": 0.0,
    "estimated_tokens": 0.0,
    "has_code_syntax": False,
    "has_math_symbols": False,
    "detected_language": "text",
    "readability_score": 0.0,
    "semantic_density": 0.0,
    "budget_remaining_ratio": 0.0,
    "latency_deadline_ms": 0.0,
}


def _vector(q: Dict[str, object]) -> List[float]:
    return [
        float(q["estimated_tokens"]),
        float(q["query_length_chars"]),
        float(q["readability_score"]),
        float(q["semantic_density"]),
        float(q["budget_remaining_ratio"]),
        float(q["latency_deadline_ms"]),
        1.0 if q["has_code_syntax"] else 0.0,
        1.0 if q["has_math_symbols"] else 0.0,
    ] + [1.0 if q["detected_language"] == lang else 0.0 for lang in LANG_VOCAB]


FEATURE_DIM = len(_vector(_SAMPLE_FEATURES))


def always_frontier(q) -> str:
    return "frontier"


def always_medium(q) -> str:
    return "medium"


def always_small(q) -> str:
    return "small"


_RANDOM_RNG = random.Random(7)


def random_router(q) -> str:
    return _RANDOM_RNG.choice(("small", "medium", "frontier"))


def heuristic_threshold_router(q) -> str:
    if q["has_math_symbols"] and q["estimated_tokens"] > 600:
        return "frontier"
    if q["has_code_syntax"] and q["estimated_tokens"] > 1200:
        return "frontier"
    if q["semantic_density"] > 0.85:
        return "frontier"
    if q["estimated_tokens"] < 350 and not q["has_math_symbols"] and not q["has_code_syntax"]:
        return "small"
    if q["readability_score"] > 70 and q["semantic_density"] < 0.45:
        return "small"
    return "medium"


def _optimal_tier(record: QueryRecord, quality_ratio: float = 0.90) -> str:
    frontier_q = record.quality_frontier
    for tier in ("small", "medium", "frontier"):
        if getattr(record, f"quality_{tier}") >= quality_ratio * frontier_q:
            return tier
    return "frontier"


def ml_classifier_router(train_records: List[QueryRecord], test_records: List[QueryRecord]) -> Callable:
    from sklearn.tree import DecisionTreeClassifier

    X_train = [_vector(r.to_features()) for r in train_records]
    y_train = [_optimal_tier(r) for r in train_records]
    clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=0)
    clf.fit(X_train, y_train)

    X_test = [_vector(r.to_features()) for r in test_records]
    y_test = [_optimal_tier(r) for r in test_records]

    def router(q) -> str:
        return clf.predict([_vector(q)])[0]

    router.model = clf  # type: ignore[attr-defined]
    router.n_params = int(clf.tree_.node_count)  # type: ignore[attr-defined]
    router.mem_bytes = len(pickle.dumps(clf))  # type: ignore[attr-defined]
    router.train_accuracy = float(clf.score(X_train, y_train))  # type: ignore[attr-defined]
    router.test_accuracy = float(clf.score(X_test, y_test))  # type: ignore[attr-defined]
    return router


def code_memory_bytes(fn: Callable) -> int:
    code = getattr(fn, "__code__", None)
    if code is None:
        return len(pickle.dumps(fn))
    total = sys.getsizeof(code)
    total += sum(sys.getsizeof(c) for c in code.co_consts if isinstance(c, (str, int, float)))
    total += sum(sys.getsizeof(n) for n in code.co_names)
    return total


def count_numeric_params(fn_or_source) -> int:
    if callable(fn_or_source):
        source = inspect.getsource(fn_or_source)
    else:
        source = fn_or_source
    return len(
        [m for m in _NUM_RE.finditer(source) if m.group(1) not in ("0", "1")]
    )
