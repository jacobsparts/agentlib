from __future__ import annotations

from agentlib.repl_benchmark.core import BenchmarkTask, default_checker
from agentlib.repl_benchmark.registry import register_task


TASKS = [
    register_task(BenchmarkTask(
        id="smoke/date",
        prompt="What's today's date?",
        description="Basic REPL completion behavior smoke test.",
        checker=default_checker,
        max_turns=4,
    )),
]