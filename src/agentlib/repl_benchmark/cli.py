
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime, timezone
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
    parser.add_argument("--timeout", type=float, metavar="SECS", help="Override per-task timeout (seconds)")
    parser.add_argument("--db", metavar="PATH", help="Write results and transcripts to a SQLite database file")
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
        total_score=sum(item.effective_total_score for item in items),
        total_possible=sum(score.possible for score in totals.values()),
        usage=usage,
    )


def _score_items(items):
    earned = sum(item.effective_total_score for item in items)
    possible = sum(item.total_possible for item in items)
    pct = 0.0 if not possible else (100.0 * earned / possible)
    return earned, possible, pct


def _use_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def _color(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def _task_status(item) -> str:
    if item.passed:
        return "PASS"
    correctness = item.score_by_category.get("correctness")
    if correctness and correctness.earned >= correctness.possible and not item.error:
        return "ISSUES"
    return "FAIL"


def _format_status(status: str, color: bool) -> str:
    colors = {
        "PASS": "32",
        "ISSUES": "33",
        "FAIL": "31",
    }
    return _color(status, colors.get(status, "0"), color)


def format_summary(result) -> str:
    lines = []
    color = _use_color()
    pct = 0.0 if not result.total_possible else (100.0 * result.total_score / result.total_possible)
    lines.append("REPL Benchmark Summary")
    lines.append(f"Model: {result.model}")
    lines.append(f"Grand Total: {result.total_score:.1f}/{result.total_possible:.1f} ({pct:.1f}%)")
    code_agent_items = [item for item in result.task_results if item.task_id.startswith("code-agent/")]
    warmup_items = [item for item in result.task_results if not item.task_id.startswith("code-agent/")]
    if warmup_items or code_agent_items:
        lines.append("")
        lines.append("Score Groups:")
        if warmup_items:
            earned, possible, group_pct = _score_items(warmup_items)
            lines.append(f"  - warmup/non-code-agent: {earned:.1f}/{possible:.1f} ({group_pct:.1f}%)")
        if code_agent_items:
            earned, possible, group_pct = _score_items(code_agent_items)
            lines.append(f"  - code-agent: {earned:.1f}/{possible:.1f} ({group_pct:.1f}%)")
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
        status = _task_status(item)
        score_note = ""
        if item.adjusted_total_score is not None:
            score_note = f" raw={item.total_score:.1f}"
        lines.append(
            f"  - {item.task_id}: {_format_status(status, color)} {item.effective_total_score:.1f}/{item.total_possible:.1f}{score_note} "
            f"(turns={item.metrics.get('turns', 0)}, retries={item.metrics.get('syntax_retries', 0)})"
        )
        if item.adjusted_total_reason:
            lines.append(f"      scoring: {item.adjusted_total_reason}")
        if item.error:
            lines.append(f"      error: {item.error}")
        for violation in item.violations[:6]:
            lines.append(
                f"      {violation.category}: -{violation.penalty:g} {violation.code} — {violation.message}"
            )
    return "\n".join(lines)


def save_to_db(db_path: str, result: BenchmarkRunResult) -> str:
    run_id = str(uuid.uuid4())
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            created_at TEXT NOT NULL,
            total_score REAL NOT NULL,
            total_possible REAL NOT NULL,
            totals_by_category_json TEXT NOT NULL,
            usage_json TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS task_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            passed INTEGER NOT NULL,
            total_score REAL NOT NULL,
            total_possible REAL NOT NULL,
            result TEXT,
            error TEXT,
            score_by_category_json TEXT NOT NULL,
            violations_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            transcript TEXT,
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        )
    """)
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            result.model,
            datetime.now(timezone.utc).isoformat(),
            result.total_score,
            result.total_possible,
            json.dumps({
                name: {"earned": s.earned, "possible": s.possible}
                for name, s in result.totals_by_category.items()
            }),
            json.dumps(result.usage),
        ),
    )
    for item in result.task_results:
        transcript = str(item.metrics.get("output", ""))
        if not transcript:
            texts = item.metrics.get("statement_text") or item.metrics.get("assistant_messages") or []
            transcript = "\n---\n".join(str(t) for t in texts)
        metrics_clean = {k: v for k, v in item.metrics.items() if k not in ("output", "statement_text", "assistant_messages")}
        conn.execute(
            "INSERT INTO task_results (run_id, task_id, passed, total_score, total_possible, result, error, score_by_category_json, violations_json, metrics_json, transcript) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                item.task_id,
                int(item.passed),
                item.total_score,
                item.total_possible,
                str(item.result) if item.result else None,
                item.error,
                json.dumps({
                    name: {"earned": s.earned, "possible": s.possible}
                    for name, s in item.score_by_category.items()
                } | ({
                    "_adjusted_total": {
                        "earned": item.adjusted_total_score,
                        "possible": item.total_possible,
                        "reason": item.adjusted_total_reason,
                    }
                } if item.adjusted_total_score is not None else {})),
                json.dumps([
                    {"code": v.code, "message": v.message, "penalty": v.penalty, "category": v.category}
                    for v in item.violations
                ]),
                json.dumps(metrics_clean),
                transcript,
            ),
        )
    conn.commit()
    conn.close()
    return run_id


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
    if args.timeout:
        from .code_agent_benchmark import CODE_AGENT_TASKS
        for task in CODE_AGENT_TASKS:
            task.timeout = args.timeout
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
    db_path = args.db
    if db_path is None and not args.json:
        fd, db_path = tempfile.mkstemp(prefix="agentlib-repl-benchmark-", suffix=".sqlite")
        os.close(fd)
    if db_path:
        run_id = save_to_db(db_path, result)
        if not args.json:
            print(f"Results saved to {db_path} (run_id={run_id})")
    if args.json:
        sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
        sys.stdout.flush()
        os._exit(0)
    else:
        print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
