from __future__ import annotations

from typing import Callable

from .core import BenchmarkTask


task_registry: dict[str, BenchmarkTask] = {}


def register_task(task: BenchmarkTask | None = None, *, replace: bool = False) -> Callable[[BenchmarkTask], BenchmarkTask] | BenchmarkTask:
    def decorator(item: BenchmarkTask) -> BenchmarkTask:
        if not replace and item.id in task_registry:
            raise ValueError(f"Benchmark task already registered: {item.id}")
        task_registry[item.id] = item
        return item

    if task is not None:
        return decorator(task)
    return decorator