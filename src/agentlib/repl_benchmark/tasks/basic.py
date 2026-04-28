
from __future__ import annotations

from datetime import date

from agentlib.repl_benchmark.core import BenchmarkTask, checker_expected_int, checker_expected_text
from agentlib.repl_benchmark.registry import register_task


TASKS = [
    register_task(BenchmarkTask(
        id="basic/date-iso",
        prompt="Without any extra text, use Python to determine today's date and return it as YYYY-MM-DD.",
        description="Simple environment access and release behavior.",
        checker=checker_expected_text(date.today().isoformat()),
        max_turns=30,
        tags=("basic", "date"),
    )),
    register_task(BenchmarkTask(
        id="basic/arithmetic",
        prompt="Compute 17 * 19 + 23 in Python and emit only the final integer.",
        description="Simple arithmetic with direct Python use.",
        checker=checker_expected_int(346),
        max_turns=30,
        tags=("basic", "math"),
    )),
    register_task(BenchmarkTask(
        id="basic/string",
        prompt="Create the string 'agentlib benchmark' in Python, convert it to uppercase, and emit only the result.",
        description="Basic Python string manipulation.",
        checker=checker_expected_text("AGENTLIB BENCHMARK"),
        max_turns=30,
        tags=("basic", "string"),
    )),
    register_task(BenchmarkTask(
        id="basic/list-sum",
        prompt="In Python, sum the numbers [5, 8, 13, 21] and emit only the integer result.",
        description="Basic list manipulation.",
        checker=checker_expected_int(47),
        max_turns=30,
        tags=("basic", "list"),
    )),
]
