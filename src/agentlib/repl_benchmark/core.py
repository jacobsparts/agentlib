
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


DEFAULT_CATEGORIES = (
    "correctness",
    "instruction_following",
    "repl_proficiency",
    "turn_efficiency",
    "environment_usage",
    "completion_behavior",
)


@dataclass
class BenchmarkViolation:
    code: str
    message: str
    penalty: float = 0.0
    category: str = "instruction_following"


@dataclass
class BenchmarkCategoryScore:
    earned: float
    possible: float
    details: list[str] = field(default_factory=list)


@dataclass
class ScoreWeights:
    correctness: float = 35.0
    instruction_following: float = 20.0
    repl_proficiency: float = 15.0
    turn_efficiency: float = 10.0
    environment_usage: float = 10.0
    completion_behavior: float = 10.0

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in DEFAULT_CATEGORIES}


@dataclass
class BenchmarkTaskContext:
    task: "BenchmarkTask"
    agent: Any
    metrics: dict[str, Any]
    result: Any
    error: Optional[BaseException] = None
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration_seconds(self) -> float:
        if not self.finished_at or not self.started_at:
            return 0.0
        return max(self.finished_at - self.started_at, 0.0)

    @property
    def result_text(self) -> str:
        return "" if self.result is None else str(self.result).strip()


@dataclass
class BenchmarkTask:
    id: str
    prompt: str
    checker: Callable[[BenchmarkTaskContext], tuple[bool, list[BenchmarkViolation], dict[str, BenchmarkCategoryScore]]]
    description: str = ""
    max_turns: int = 10
    max_syntax_retries: int = 3
    setup: Optional[Callable[[Any], None]] = None
    teardown: Optional[Callable[[Any], None]] = None
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    tags: tuple[str, ...] = ()


@dataclass
class BenchmarkTaskResult:
    task_id: str
    passed: bool
    result: Any
    error: Optional[str]
    score_by_category: dict[str, BenchmarkCategoryScore]
    total_score: float
    total_possible: float
    metrics: dict[str, Any]
    violations: list[BenchmarkViolation]


@dataclass
class BenchmarkRunResult:
    model: str
    task_results: list[BenchmarkTaskResult]
    totals_by_category: dict[str, BenchmarkCategoryScore]
    total_score: float
    total_possible: float
    usage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "task_results": [
                {
                    "task_id": item.task_id,
                    "passed": item.passed,
                    "result": item.result,
                    "error": item.error,
                    "score_by_category": {
                        name: {
                            "earned": score.earned,
                            "possible": score.possible,
                            "details": list(score.details),
                        }
                        for name, score in item.score_by_category.items()
                    },
                    "total_score": item.total_score,
                    "total_possible": item.total_possible,
                    "metrics": item.metrics,
                    "violations": [
                        {
                            "code": v.code,
                            "message": v.message,
                            "penalty": v.penalty,
                            "category": v.category,
                        }
                        for v in item.violations
                    ],
                }
                for item in self.task_results
            ],
            "totals_by_category": {
                name: {
                    "earned": score.earned,
                    "possible": score.possible,
                    "details": list(score.details),
                }
                for name, score in self.totals_by_category.items()
            },
            "total_score": self.total_score,
            "total_possible": self.total_possible,
            "usage": self.usage,
        }


def empty_scores(weights: Optional[ScoreWeights] = None) -> dict[str, BenchmarkCategoryScore]:
    values = (weights or ScoreWeights()).as_dict()
    return {
        name: BenchmarkCategoryScore(earned=0.0, possible=values[name], details=[])
        for name in DEFAULT_CATEGORIES
    }


def full_scores(weights: Optional[ScoreWeights] = None) -> dict[str, BenchmarkCategoryScore]:
    values = (weights or ScoreWeights()).as_dict()
    return {
        name: BenchmarkCategoryScore(earned=values[name], possible=values[name], details=[])
        for name in DEFAULT_CATEGORIES
    }


def finalize_scores(scores: dict[str, BenchmarkCategoryScore]) -> tuple[float, float]:
    total = sum(max(0.0, score.earned) for score in scores.values())
    possible = sum(max(0.0, score.possible) for score in scores.values())
    return total, possible


def apply_violations(
    scores: dict[str, BenchmarkCategoryScore],
    violations: list[BenchmarkViolation],
) -> dict[str, BenchmarkCategoryScore]:
    for violation in violations:
        score = scores.setdefault(
            violation.category,
            BenchmarkCategoryScore(earned=0.0, possible=0.0, details=[]),
        )
        score.earned = max(0.0, score.earned - float(violation.penalty))
        score.details.append(f"{violation.code}: {violation.message}")
    return scores


def make_violation(code: str, message: str, penalty: float, category: str) -> BenchmarkViolation:
    return BenchmarkViolation(code=code, message=message, penalty=float(penalty), category=category)


def result_matches(ctx: BenchmarkTaskContext, expected: Any) -> bool:
    return ctx.result_text == str(expected).strip()


def result_int(ctx: BenchmarkTaskContext) -> Optional[int]:
    try:
        return int(ctx.result_text)
    except (TypeError, ValueError):
        return None


def base_scores(ctx: BenchmarkTaskContext) -> tuple[bool, list[BenchmarkViolation], dict[str, BenchmarkCategoryScore]]:
    scores = full_scores(ctx.task.weights)
    violations: list[BenchmarkViolation] = []
    metrics = ctx.metrics

    if ctx.error is not None:
        scores = empty_scores(ctx.task.weights)
        scores["correctness"].details.append(f"task failed with {type(ctx.error).__name__}")
        return False, violations, scores

    syntax_retries = int(metrics.get("syntax_retries", 0) or 0)
    if syntax_retries:
        violations.append(make_violation(
            "syntax_retry",
            f"{syntax_retries} syntax retries",
            min(scores["instruction_following"].possible, 3.0 * syntax_retries),
            "instruction_following",
        ))

    if metrics.get("saw_bash_python"):
        violations.append(make_violation(
            "bash_python",
            "used bash to invoke python inside the REPL",
            min(scores["repl_proficiency"].possible, 8.0),
            "repl_proficiency",
        ))

    if not metrics.get("release_called"):
        violations.append(make_violation(
            "missing_release",
            "did not explicitly call emit(..., release=True)",
            min(scores["completion_behavior"].possible, 10.0),
            "completion_behavior",
        ))

    progress_count = int(metrics.get("progress_count", 0) or 0)
    if progress_count:
        violations.append(make_violation(
            "unnecessary_progress",
            f"emitted progress output {progress_count} times",
            min(scores["completion_behavior"].possible, 1.5 * progress_count),
            "completion_behavior",
        ))

    turns = int(metrics.get("turns", 0) or 0)
    if turns > 1:
        violations.append(make_violation(
            "extra_turns",
            f"completed in {turns} turns",
            min(scores["turn_efficiency"].possible, float(turns - 1) * 2.0),
            "turn_efficiency",
        ))

    runtime_errors = int(metrics.get("runtime_errors", 0) or 0)
    if runtime_errors:
        violations.append(make_violation(
            "runtime_error",
            f"{runtime_errors} runtime error outputs observed",
            min(scores["correctness"].possible, float(runtime_errors) * 6.0),
            "correctness",
        ))

    read_count = int(metrics.get("read_count", 0) or 0)
    if read_count > 3:
        violations.append(make_violation(
            "excessive_reads",
            f"read tools used {read_count} times",
            min(scores["environment_usage"].possible, float(read_count - 3)),
            "environment_usage",
        ))

    scores = apply_violations(scores, violations)
    return True, violations, scores


def checker_expected_text(
    expected: str,
    *,
    penalty: float = 20.0,
    substring: bool = False,
) -> Callable[[BenchmarkTaskContext], tuple[bool, list[BenchmarkViolation], dict[str, BenchmarkCategoryScore]]]:
    def checker(ctx: BenchmarkTaskContext):
        passed, violations, scores = base_scores(ctx)
        if not passed:
            return passed, violations, scores
        actual = ctx.result_text
        ok = expected in actual if substring else actual == expected
        if not ok:
            violations.append(make_violation(
                "wrong_result",
                f"expected {expected!r}, got {actual!r}",
                min(scores["correctness"].possible, penalty),
                "correctness",
            ))
            passed = False
        scores = apply_violations(scores, violations[len(scores["correctness"].details):] if False else [])
        scores = full_scores(ctx.task.weights)
        passed2, base_violations, scores = base_scores(ctx)
        violations = list(base_violations)
        if not ok:
            violations.append(make_violation(
                "wrong_result",
                f"expected {expected!r}, got {actual!r}",
                min(scores["correctness"].possible, penalty),
                "correctness",
            ))
            passed2 = False
        scores = apply_violations(scores, violations)
        return passed2, violations, scores
    return checker


def checker_expected_int(
    expected: int,
    *,
    penalty: float = 20.0,
) -> Callable[[BenchmarkTaskContext], tuple[bool, list[BenchmarkViolation], dict[str, BenchmarkCategoryScore]]]:
    def checker(ctx: BenchmarkTaskContext):
        passed, violations, scores = base_scores(ctx)
        if not passed:
            return passed, violations, scores
        actual = result_int(ctx)
        if actual != expected:
            violations.append(make_violation(
                "wrong_result",
                f"expected {expected}, got {ctx.result_text!r}",
                min(scores["correctness"].possible, penalty),
                "correctness",
            ))
            passed = False
        scores = full_scores(ctx.task.weights)
        passed2, base_violations, scores = base_scores(ctx)
        violations = list(base_violations) + ([
            make_violation(
                "wrong_result",
                f"expected {expected}, got {ctx.result_text!r}",
                min(scores["correctness"].possible, penalty),
                "correctness",
            )
        ] if actual != expected else [])
        scores = apply_violations(scores, violations)
        return passed2 and actual == expected, violations, scores
    return checker


def checker_predicate(
    predicate: Callable[[BenchmarkTaskContext], tuple[bool, Optional[str]]],
    *,
    penalty: float = 20.0,
) -> Callable[[BenchmarkTaskContext], tuple[bool, list[BenchmarkViolation], dict[str, BenchmarkCategoryScore]]]:
    def checker(ctx: BenchmarkTaskContext):
        passed, violations, scores = base_scores(ctx)
        if not passed:
            return passed, violations, scores
        ok, detail = predicate(ctx)
        if not ok:
            violations.append(make_violation(
                "wrong_result",
                detail or "result predicate failed",
                min(scores["correctness"].possible, penalty),
                "correctness",
            ))
        scores = full_scores(ctx.task.weights)
        passed2, base_violations, scores = base_scores(ctx)
        violations = list(base_violations) + ([
            make_violation(
                "wrong_result",
                detail or "result predicate failed",
                min(scores["correctness"].possible, penalty),
                "correctness",
            )
        ] if not ok else [])
        scores = apply_violations(scores, violations)
        return passed2 and ok, violations, scores
    return checker


class InstrumentedREPLBenchmarkMixin:
    _bash_python_pattern = re.compile(r'''bash\((?P<quote>["']).*?\bpython(?:3)?\b.*?(?P=quote)\)''', re.DOTALL)
    _emit_release_pattern = re.compile(r"emit\((?:.|\n)*?release\s*=\s*True", re.DOTALL)
    _name_assignment_pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE)

    def _ensure_setup(self):
        super()._ensure_setup()
        self._suspend_persistence = True
        self._benchmark_reset_metrics()

    def _benchmark_reset_metrics(self):
        self._benchmark_metrics = {
            "turns": 0,
            "syntax_retries": 0,
            "statement_count": 0,
            "tool_calls": {},
            "chunk_counts": {},
            "release_called": False,
            "assistant_messages": [],
            "statement_text": [],
            "runtime_errors": 0,
            "saw_bash_python": False,
            "saw_python_state_reuse": False,
            "empty_output_turns": 0,
            "progress_count": 0,
            "emit_count": 0,
            "print_count": 0,
            "read_count": 0,
            "write_count": 0,
            "bash_calls": 0,
            "assigned_names": [],
            "started_at": time.time(),
            "finished_at": None,
        }

    def on_repl_execute(self, _):
        self._benchmark_metrics["turns"] += 1

    def on_retry(self, kind, attempt):
        if kind == "syntax":
            self._benchmark_metrics["syntax_retries"] += 1

    def on_statement_output(self, statement_chunks):
        self._benchmark_metrics["statement_count"] += 1
        text = "".join(chunk for _, chunk in statement_chunks)
        self._benchmark_metrics["statement_text"].append(text)
        if "Traceback" in text or "Error:" in text:
            self._benchmark_metrics["runtime_errors"] += 1
        if self._bash_python_pattern.search(text):
            self._benchmark_metrics["saw_bash_python"] = True
        assigned = self._name_assignment_pattern.findall(text)
        if assigned:
            self._benchmark_metrics["assigned_names"].extend(assigned)
            if self._benchmark_metrics["turns"] > 1:
                self._benchmark_metrics["saw_python_state_reuse"] = True

    def on_repl_chunk(self, chunk, msg_type):
        counts = self._benchmark_metrics["chunk_counts"]
        counts[msg_type] = counts.get(msg_type, 0) + 1
        if msg_type == "progress":
            self._benchmark_metrics["progress_count"] += 1
        elif msg_type == "emit":
            self._benchmark_metrics["emit_count"] += 1
        elif msg_type == "print":
            self._benchmark_metrics["print_count"] += 1
        elif msg_type in ("read", "read_attach", "read_partial"):
            self._benchmark_metrics["read_count"] += 1
        elif msg_type == "file_written":
            self._benchmark_metrics["write_count"] += 1

    def _on_assistant_message_committed(self, resp):
        content = (resp.get("content") or "")
        self._benchmark_metrics["assistant_messages"].append(content)
        if self._emit_release_pattern.search(content):
            self._benchmark_metrics["release_called"] = True
        if "bash(" in content:
            self._benchmark_metrics["bash_calls"] += content.count("bash(")
        if self._bash_python_pattern.search(content):
            self._benchmark_metrics["saw_bash_python"] = True

    def toolcall(self, toolname, function_args):
        calls = self._benchmark_metrics["tool_calls"]
        calls[toolname] = calls.get(toolname, 0) + 1
        return super().toolcall(toolname, function_args)

    def run_loop(self, *args, **kwargs):
        try:
            return super().run_loop(*args, **kwargs)
        finally:
            self._benchmark_metrics["finished_at"] = time.time()


def default_checker(ctx: BenchmarkTaskContext) -> tuple[bool, list[BenchmarkViolation], dict[str, BenchmarkCategoryScore]]:
    return base_scores(ctx)
