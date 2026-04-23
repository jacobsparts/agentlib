
from .core import (
    BenchmarkCategoryScore,
    BenchmarkRunResult,
    BenchmarkTask,
    BenchmarkTaskContext,
    BenchmarkTaskResult,
    BenchmarkViolation,
    InstrumentedREPLBenchmarkMixin,
    ScoreWeights,
)
from .discovery import discover_task_modules, discover_tasks, load_task_module
from .registry import register_task, task_registry
from .runner import REPLBenchmarkRunner


def format_summary(*args, **kwargs):
    from .cli import format_summary as _format_summary
    return _format_summary(*args, **kwargs)


__all__ = [
    "BenchmarkCategoryScore",
    "BenchmarkRunResult",
    "BenchmarkTask",
    "BenchmarkTaskContext",
    "BenchmarkTaskResult",
    "BenchmarkViolation",
    "InstrumentedREPLBenchmarkMixin",
    "REPLBenchmarkRunner",
    "ScoreWeights",
    "format_summary",
    "discover_task_modules",
    "discover_tasks",
    "load_task_module",
    "register_task",
    "task_registry",
]
