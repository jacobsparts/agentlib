
from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

from .core import BenchmarkTask
from .registry import task_registry


def discover_task_modules(paths: Iterable[str | Path] | None = None) -> list[Path]:
    modules: list[Path] = []
    for root in paths or []:
        root_path = Path(root).expanduser()
        if root_path.is_file() and root_path.suffix == ".py":
            modules.append(root_path)
            continue
        if root_path.is_dir():
            modules.extend(sorted(p for p in root_path.rglob("*.py") if p.name != "__init__.py"))
    return modules


def load_task_module(module: str | Path) -> ModuleType:
    if isinstance(module, Path) or (isinstance(module, str) and module.endswith(".py")):
        path = Path(module).expanduser().resolve()
        name = f"agentlib_repl_benchmark_{path.stem}_{abs(hash(str(path)))}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load benchmark module from {path}")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
        return loaded
    return importlib.import_module(str(module))


def load_builtin_task_modules() -> list[ModuleType]:
    import agentlib.repl_benchmark.tasks as builtin_tasks

    loaded = []
    for mod in pkgutil.iter_modules(builtin_tasks.__path__, builtin_tasks.__name__ + "."):
        loaded.append(importlib.import_module(mod.name))
    return loaded


def discover_tasks(
    modules: Iterable[str] | None = None,
    paths: Iterable[str | Path] | None = None,
    include_builtin: bool = True,
) -> list[BenchmarkTask]:
    if include_builtin:
        load_builtin_task_modules()
    for module in modules or ():
        load_task_module(module)
    for path in discover_task_modules(paths):
        load_task_module(path)
    return [task_registry[key] for key in sorted(task_registry)]
