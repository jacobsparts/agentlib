
from __future__ import annotations

from agentlib.repl_benchmark.core import BenchmarkTask, checker_expected_int, checker_expected_text
from agentlib.repl_benchmark.registry import register_task


TASKS = [
    register_task(BenchmarkTask(
        id="behavior/no-bash-python",
        prompt="Use direct Python in the REPL, not shelling out. Compute 2 ** 10 and emit only the integer.",
        description="Penalizes bash('python ...') misuse.",
        checker=checker_expected_int(1024),
        max_turns=30,
        tags=("behavior", "repl"),
    )),
    register_task(BenchmarkTask(
        id="behavior/one-turn",
        prompt="Complete this in a single turn if possible: compute len('release discipline') and emit only the integer result.",
        description="Measures over-turning for a trivial task.",
        checker=checker_expected_int(18),
        max_turns=30,
        tags=("behavior", "turns"),
    )),
    register_task(BenchmarkTask(
        id="behavior/no-progress-chat",
        prompt="Do not print progress updates. Compute the last two digits of 12345 squared and emit only the two-digit string.",
        description="Penalizes unnecessary progress chatter.",
        checker=checker_expected_text("25"),
        max_turns=30,
        tags=("behavior", "completion"),
    )),
    register_task(BenchmarkTask(
        id="behavior/release-when-done",
        prompt="Compute the value of sum(range(1, 11)) and finish immediately once you have the answer. Emit only the integer.",
        description="Checks direct completion behavior.",
        checker=checker_expected_int(55),
        max_turns=30,
        tags=("behavior", "release"),
    )),
]
