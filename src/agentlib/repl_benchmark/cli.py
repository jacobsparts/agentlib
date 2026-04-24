
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from importlib import import_module

from .code_agent_benchmark import CodeAgentBenchmarkRunner
from .core import BenchmarkCategoryScore, BenchmarkRunResult
from .runner import REPLBenchmarkRunner


def _load_agent_class(path: str):
    if ":" not in path:
        raise ValueError("Agent path must be in module:Class format")
    module_name, class_name = path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the REPL benchmark suite.")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--agent", default="agentlib.agents.code_agent:CodeAgentBase", help="Agent class as module:Class")
    parser.add_argument("--task-module", action="append", default=[], help="Additional benchmark task module")
    parser.add_argument("--task-path", action="append", default=[], help="Directory or .py file containing benchmark tasks")
    parser.add_argument("--no-builtin", action="store_true", help="Disable built-in benchmark tasks")
    parser.add_argument("--no-code-agent-builtin", action="store_true", help="Disable built-in PTY-backed CodeAgent benchmark tasks")
    parser.add_argument("--show-repl-output", action="store_true", help="Stream PTY-backed CodeAgent REPL output while benchmarks run")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable summary")
    return parser


def merge_results(*results: BenchmarkRunResult) -> BenchmarkRunResult:
    items = [item for result in results for item in result.task_results]
    totals: dict[str, BenchmarkCategoryScore] = {}
    usage = {
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cost": 0.0,
        "requests": 0,
    }
    model = results[0].model if results else ""
    for result in results:
        model = result.model or model
        for name, score in result.totals_by_category.items():
            bucket = totals.setdefault(name, BenchmarkCategoryScore(earned=0.0, possible=0.0, details=[]))
            bucket.earned += score.earned
            bucket.possible += score.possible
            bucket.details.extend(score.details)
        for key in usage:
            usage[key] += result.usage.get(key, 0)
    usage["model"] = model
    return BenchmarkRunResult(
        model=model,
        task_results=items,
        totals_by_category=totals,
        total_score=sum(score.earned for score in totals.values()),
        total_possible=sum(score.possible for score in totals.values()),
        usage=usage,
    )


def format_summary(result) -> str:
    lines = []
    pct = 0.0 if not result.total_possible else (100.0 * result.total_score / result.total_possible)
    lines.append("REPL Benchmark Summary")
    lines.append(f"Model: {result.model}")
    lines.append(f"Grand Total: {result.total_score:.1f}/{result.total_possible:.1f} ({pct:.1f}%)")
    lines.append("")
    lines.append("Category Totals:")
    for name, score in sorted(result.totals_by_category.items()):
        part = 0.0 if not score.possible else (100.0 * score.earned / score.possible)
        lines.append(f"  - {name}: {score.earned:.1f}/{score.possible:.1f} ({part:.1f}%)")
    lines.append("")
    lines.append("Usage:")
    usage = result.usage
    lines.append(
        "  - requests={requests} prompt={prompt_tokens} cached={cached_tokens} reasoning={reasoning_tokens} completion={completion_tokens} cost=${cost:.3f}".format(
            **usage
        )
    )
    lines.append("")
    lines.append("Tasks:")
    for item in result.task_results:
        status = "PASS" if item.passed else "FAIL"
        lines.append(
            f"  - {item.task_id}: {status} {item.total_score:.1f}/{item.total_possible:.1f} "
            f"(turns={item.metrics.get('turns', 0)}, retries={item.metrics.get('syntax_retries', 0)})"
        )
        if item.error:
            lines.append(f"      error: {item.error}")
        for violation in item.violations[:6]:
            lines.append(
                f"      {violation.category}: -{violation.penalty:g} {violation.code} — {violation.message}"
            )
    return "\n".join(lines)


def main(argv=None):
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)
    os.environ["AGENTLIB_SUPPRESS_USAGE_ATEXIT"] = "1"
    agent_cls = _load_agent_class(args.agent)
    should_include_generic = (not args.no_builtin) or bool(args.task_module) or bool(args.task_path)
    should_include_code_agent_builtin = (
        not args.no_code_agent_builtin
        and args.agent == "agentlib.agents.code_agent:CodeAgentBase"
    )
    results = []
    stderr_target = sys.stderr if args.show_repl_output else io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr_target):
        if should_include_generic:
            results.append(REPLBenchmarkRunner(
                agent_cls,
                model=args.model,
                task_modules=args.task_module,
                task_paths=args.task_path,
                include_builtin=not args.no_builtin,
            ).run())
        if should_include_code_agent_builtin:
            code_model = args.model or (results[0].model if results else None)
            results.append(CodeAgentBenchmarkRunner(
                model=code_model,
                stream_output=sys.stderr if args.show_repl_output else None,
            ).run())
    if not results:
        raise ValueError("No benchmark tasks found")
    result = results[0] if len(results) == 1 else merge_results(*results)
    if args.json:
        sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
        sys.stdout.flush()
        os._exit(0)
    else:
        print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
