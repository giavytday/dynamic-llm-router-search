"""Simulated LLM semantic mutation with deterministic offline fallback generation."""

import random
import re
from typing import List, Optional, Tuple

INITIAL_BASELINE_CODE = (
    "def select_model(q):\n"
    "    if q[\"has_math_symbols\"]:\n"
    "        return \"frontier\"\n"
    "    if q[\"has_code_syntax\"] and q[\"estimated_tokens\"] > 1400:\n"
    "        return \"frontier\"\n"
    "    if q[\"estimated_tokens\"] < 320 and q[\"semantic_density\"] < 0.55:\n"
    "        return \"small\"\n"
    "    if q[\"readability_score\"] > 62.0 and q[\"budget_remaining_ratio\"] < 0.35:\n"
    "        return \"small\"\n"
    "    return \"medium\"\n"
)

THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.S)
CODE_RE = re.compile(r"<code>(.*?)</code>", re.S)
NUM_RE = re.compile(r"(?<![\w.'])(\d+\.\d+|\d+)(?![\w.'])")
CMP_RE = re.compile(r"(<=|>=|==|!=|<|>)")

OPERATIONS: Tuple[str, ...] = (
    "tweak_constants",
    "flip_comparison",
    "rotate_branches",
    "invert_condition",
    "synthesize_policy",
    "crossover",
)
OP_WEIGHTS: Tuple[float, ...] = (0.28, 0.16, 0.14, 0.10, 0.18, 0.14)

_THOUGHTS = {
    "tweak_constants": "Adjusting numeric decision boundaries to shift the cost-quality operating point.",
    "flip_comparison": "Inverting a comparison direction to reroute the marginal traffic segment.",
    "rotate_branches": "Reordering decision branches so earlier rules capture different query mass.",
    "invert_condition": "Negating a guard condition to test the complementary routing hypothesis.",
    "synthesize_policy": "Synthesizing a fresh rule-based policy from the routing grammar.",
    "crossover": "Recombining decision branches from two parent policies into a hybrid offspring.",
}


def _parse_blocks(code: str) -> Optional[Tuple[List[str], List[List[str]], List[str]]]:
    lines = code.rstrip("\n").split("\n")
    try:
        first_if = next(i for i, l in enumerate(lines) if l.startswith("    if "))
    except StopIteration:
        return None
    prologue = lines[:first_if]
    blocks: List[List[str]] = []
    tail: List[str] = []
    current: Optional[List[str]] = None
    closed = True
    for line in lines[first_if:]:
        if line.startswith("    if "):
            current = [line]
            blocks.append(current)
            closed = False
            continue
        if current is None:
            continue
        if not closed:
            current.append(line)
            if line.lstrip().startswith("return "):
                closed = True
        else:
            tail.append(line)
    if not blocks or not tail:
        return None
    return prologue, blocks, tail


def _assemble(prologue: List[str], blocks: List[List[str]], tail: List[str]) -> str:
    lines = list(prologue)
    for block in blocks:
        lines.extend(block)
    lines.extend(tail)
    return "\n".join(lines) + "\n"


class SemanticMutator:
    def __init__(self, seed: int = 99) -> None:
        self.rng = random.Random(seed)

    def extract_thought(self, response: str) -> Optional[str]:
        match = THOUGHT_RE.search(response)
        return match.group(1).strip() if match else None

    def extract_code(self, response: str) -> Optional[str]:
        match = CODE_RE.search(response)
        return match.group(1).strip("\n ") + "\n" if match else None

    def _wrap_response(self, thought: str, code: str) -> str:
        return f"<thought>{thought}</thought>\n<code>\n{code}\n</code>"

    def _tweak_constants(self, code: str) -> str:
        matches = [m for m in NUM_RE.finditer(code) if m.group(1) not in ("0", "1")]
        if not matches:
            return code
        chosen = self.rng.sample(matches, k=min(len(matches), self.rng.randint(1, 3)))
        out: List[str] = []
        last = 0
        for m in sorted(chosen, key=lambda m: m.start()):
            out.append(code[last:m.start()])
            original = m.group(1)
            new_value = float(original) * self.rng.uniform(0.45, 1.8)
            if "." in original:
                out.append(str(round(new_value, 3)))
            else:
                out.append(str(max(1, int(round(new_value)))))
            last = m.end()
        out.append(code[last:])
        return "".join(out)

    def _flip_comparison(self, code: str) -> str:
        matches = list(CMP_RE.finditer(code))
        if not matches:
            return code
        m = self.rng.choice(matches)
        swap = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "!=", "!=": "=="}
        return code[: m.start()] + swap[m.group(0)] + code[m.end():]

    def _rotate_branches(self, code: str) -> str:
        parsed = _parse_blocks(code)
        if parsed is None or len(parsed[1]) < 2:
            return code
        prologue, blocks, tail = parsed
        r = self.rng.randint(1, len(blocks) - 1)
        blocks = blocks[r:] + blocks[:r]
        return _assemble(prologue, blocks, tail)

    def _invert_condition(self, code: str) -> str:
        candidates = [
            i
            for i, l in enumerate(code.split("\n"))
            if l.startswith("    if ") and "not (" not in l
        ]
        if not candidates:
            return code
        lines = code.split("\n")
        idx = self.rng.choice(candidates)
        condition = lines[idx][len("    if "):].rstrip(":")
        lines[idx] = f"    if not ({condition}):"
        return "\n".join(lines)

    def _synthesize_policy(self) -> str:
        rng = self.rng
        languages = ("python", "javascript", "sql", "latex", "text")

        def tok() -> int:
            return rng.choice((96, 128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1600, 2048))

        def dens() -> float:
            return round(rng.uniform(0.15, 0.92), 2)

        def cond_factory(kind: int) -> str:
            table = (
                lambda: f'q["estimated_tokens"] < {tok()}',
                lambda: f'q["estimated_tokens"] > {tok()}',
                lambda: f'q["semantic_density"] < {dens()}',
                lambda: f'q["readability_score"] > {rng.uniform(20, 90):.1f}',
                lambda: 'q["has_code_syntax"]',
                lambda: 'not q["has_math_symbols"]',
                lambda: f'q["has_math_symbols"] and q["estimated_tokens"] > {tok()}',
                lambda: f'q["detected_language"] == "{rng.choice(languages)}"',
                lambda: f'q["budget_remaining_ratio"] < {dens()} and q["estimated_tokens"] < {tok()}',
            )
            return table[kind % len(table)]()

        tier_pool = ("small", "small", "small", "medium", "medium", "frontier", "frontier", "frontier")
        lines = ["def select_model(q):"]
        seen = set()
        for _ in range(rng.randint(2, 4)):
            cond = cond_factory(rng.randrange(9))
            if cond in seen:
                continue
            seen.add(cond)
            lines.append(f"    if {cond}:")
            lines.append(f'        return "{rng.choice(tier_pool)}"')
        if len(lines) == 1:
            lines.extend(['    if q["estimated_tokens"] < 256:', '        return "small"'])
        lines.append(f'    return "{rng.choice(tier_pool)}"')
        return "\n".join(lines) + "\n"

    def _crossover(self, parent_a: str, parent_b: str) -> str:
        pa = _parse_blocks(parent_a)
        pb = _parse_blocks(parent_b)
        if pa is None or pb is None:
            return parent_a
        prologue_a, blocks_a, tail_a = pa
        _, blocks_b, tail_b = pb
        child_blocks: List[List[str]] = []
        for j in range(len(blocks_a)):
            if j < len(blocks_b) and self.rng.random() < 0.5:
                child_blocks.append(list(blocks_b[j]))
            else:
                child_blocks.append(list(blocks_a[j]))
        child_blocks = child_blocks[:6]
        tail = tail_a if self.rng.random() < 0.5 else tail_b
        return _assemble(prologue_a, child_blocks, tail)

    def mutate(self, parent_code: str, co_parent_code: Optional[str] = None) -> Tuple[str, str, str]:
        operation = self.rng.choices(OPERATIONS, weights=list(OP_WEIGHTS), k=1)[0]
        if operation == "crossover" and co_parent_code is None:
            operation = "synthesize_policy"
        if operation == "tweak_constants":
            child = self._tweak_constants(parent_code)
        elif operation == "flip_comparison":
            child = self._flip_comparison(parent_code)
        elif operation == "rotate_branches":
            child = self._rotate_branches(parent_code)
        elif operation == "invert_condition":
            child = self._invert_condition(parent_code)
        elif operation == "crossover":
            child = self._crossover(parent_code, co_parent_code or "")
        else:
            child = self._synthesize_policy()
        if child.strip() == parent_code.strip():
            operation = "synthesize_policy"
            child = self._synthesize_policy()
        thought = _THOUGHTS[operation]
        response = self._wrap_response(thought, child)
        extracted_thought = self.extract_thought(response)
        extracted_code = self.extract_code(response)
        if extracted_code is None or extracted_thought is None:
            return self.fallback_mutate("tag extraction failed")
        return extracted_thought, extracted_code, operation

    def fallback_mutate(self, reason: str) -> Tuple[str, str, str]:
        code = self._synthesize_policy()
        thought = f"[offline-fallback] {reason}; emitted deterministic grammar policy."
        return thought, code, "fallback"

    def random_policy(self) -> Tuple[str, str, str]:
        code = self._synthesize_policy()
        thought = "[grammar-template] Sampled a fresh policy from the routing DSL."
        return thought, code, "synthesize_policy"
