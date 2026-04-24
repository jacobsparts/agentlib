from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .code_agent_harness import PTYRunResult, PTYTimeoutError, run_pty_session, strip_ansi
from .core import (
    BenchmarkCategoryScore,
    BenchmarkRunResult,
    BenchmarkTaskResult,
    BenchmarkViolation,
    ScoreWeights,
    apply_violations,
    finalize_scores,
    full_scores,
    make_violation,
)


Checker = Callable[["CodeAgentBenchmarkContext"], tuple[bool, list[BenchmarkViolation]]]
TaskPrepare = Callable[[Path, Path], dict[str, object]]


@dataclass
class CodeAgentBenchmarkTask:
    id: str
    prompt: str
    checker: Checker
    description: str = ""
    max_turns: int = 8
    timeout: float = 25.0
    wait_for: str = "Session ended. Goodbye!"
    tags: tuple[str, ...] = ()
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    prepare: Optional[TaskPrepare] = None


@dataclass
class CodeAgentBenchmarkContext:
    task: CodeAgentBenchmarkTask
    run: PTYRunResult
    output: str
    cwd: Path
    home: Path
    session_db: Path
    history_db: Path
    task_dir: Path
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def lines(self) -> list[str]:
        return self.output.splitlines()

    @property
    def response_block(self) -> str:
        text = self.output
        marker = "──────────────────────────────────"
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
        if "\n> " in text:
            text = text.split("\n> ", 1)[0]
        lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Session ended. Goodbye!"):
                continue
            if line.startswith("test-code-agent:"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    @property
    def final_line(self) -> str:
        ignored_prefixes = (
            "Code Agent",
            "Enter = submit",
            "Commands:",
            "Loading ",
            "Working... (turn ",
            "────────",
            ">>> ",
            "... (",
            "[Attachment:",
            "test-code-agent:",
            "<frozen runpy>:",
        )
        ignored_exact = {
            "",
            ">",
            "Python REPL-based coding assistant",
            "Session ended. Goodbye!",
        }
        candidates: list[str] = []
        for raw in self.lines:
            line = raw.strip()
            if not line or line in ignored_exact:
                continue
            if line.startswith("> "):
                continue
            if line.endswith("Python REPL-based coding assistant"):
                continue
            if any(line.startswith(prefix) for prefix in ignored_prefixes):
                continue
            if re.fullmatch(r".*· (?:sandbox|no sandbox)", line):
                continue
            candidates.append(line)
        return candidates[-1] if candidates else ""

    @property
    def turn_count(self) -> int:
        turns = {int(num) for num in re.findall(r"Working... \(turn (\d+)\)", self.output)}
        return max(turns) if turns else 0

    def saw_tool(self, tool_name: str) -> bool:
        return f">>> {tool_name}(" in self.output or f"{tool_name}(" in self.output

    def saw_bash_python(self) -> bool:
        return bool(re.search(r'''bash\((?P<quote>["']).*?\bpython(?:3)?\b.*?(?P=quote)\)''', self.output, re.DOTALL))

    def tool_offset(self, tool_name: str) -> int:
        direct = self.output.find(f">>> {tool_name}(")
        if direct != -1:
            return direct
        return self.output.find(f"{tool_name}(")

    def first_tool_offset(self) -> int:
        positions = []
        for tool_name in ("grep", "read", "view_file", "preview", "bash"):
            idx = self.output.find(f">>> {tool_name}(")
            if idx != -1:
                positions.append(idx)
        return min(positions) if positions else -1

    def final_answer_offset(self) -> int:
        line = self.final_line
        return self.output.rfind(line) if line else -1

    @property
    def file_edit_attempts(self) -> int:
        return self.output.count(">>> apply_patch(") + self.output.count(">>> edit(")

    @property
    def failed_edit_attempts(self) -> int:
        patterns = (
            r"not found",
            r"Patch failed",
            r"Failed to apply patch",
            r"Failed to edit",
            r"Edit failed",
        )
        detected = sum(len(re.findall(pattern, self.output, re.IGNORECASE)) for pattern in patterns)
        if detected:
            return detected
        if self.file_edit_attempts > 1:
            return self.file_edit_attempts - 1
        return 0


@dataclass
class CodeAgentBenchmarkResult:
    task_id: str
    passed: bool
    violations: list[BenchmarkViolation] = field(default_factory=list)
    output: str = ""
    returncode: int = 0
    metrics: dict[str, object] = field(default_factory=dict)


class CodeAgentBenchmarkSuite:
    def __init__(
        self,
        tasks: list[CodeAgentBenchmarkTask],
        *,
        model: str = "test-code-agent",
        agent_args: Optional[list[str]] = None,
        stream_output=None,
    ):
        self.tasks = list(tasks)
        self.model = model
        self.agent_args = list(agent_args or [])
        self.stream_output = stream_output

    def run_task(
        self,
        task: CodeAgentBenchmarkTask,
        *,
        env: dict[str, str],
        cwd: str | Path,
        python_executable: str | None = None,
    ) -> CodeAgentBenchmarkResult:
        cwd_path = Path(cwd).resolve()
        with tempfile.TemporaryDirectory(prefix=f"{task.id.replace('/', '-')}-", dir=str(Path(env["HOME"]))) as task_dir_str:
            task_dir = Path(task_dir_str)
            prepared = task.prepare(task_dir, cwd_path) if task.prepare else {}
            prompt = str(prepared.get("prompt", task.prompt))
            metadata = dict(prepared.get("metadata", {}))
            args = [
                python_executable or sys.executable,
                "-m",
                "agentlib.agents.code_agent",
                "--model",
                self.model,
                "--max-turns",
                str(task.max_turns),
                *self.agent_args,
            ]
            timeout_error = None
            try:
                run = run_pty_session(
                    args,
                    inputs=[prompt.rstrip("\n") + "\n"],
                    env=env,
                    cwd=str(cwd_path),
                    timeout=task.timeout,
                    wait_for=task.wait_for,
                    stream_output=self.stream_output,
                )
                output = strip_ansi(run.output).replace("\r", "")
            except PTYTimeoutError as exc:
                timeout_error = str(exc)
                output = strip_ansi(exc.output).replace("\r", "")
                run = PTYRunResult(args=args, returncode=124, output=exc.output)
            ctx = CodeAgentBenchmarkContext(
                task=task,
                run=run,
                output=output,
                cwd=cwd_path,
                home=Path(env["HOME"]),
                session_db=Path(env["AGENTLIB_SESSION_DB"]),
                history_db=Path(env["AGENTLIB_CLI_HISTORY_DB"]),
                task_dir=task_dir,
                metadata=metadata,
            )
            passed, violations = task.checker(ctx)
            metrics = {
                "turns": ctx.turn_count,
                "file_edit_attempts": ctx.file_edit_attempts,
                "failed_edit_attempts": ctx.failed_edit_attempts,
                "task_dir": str(task_dir),
                "target_file": str(metadata.get("target_file", "")),
                "timeout_error": timeout_error,
            }
            return CodeAgentBenchmarkResult(
                task_id=task.id,
                passed=bool(passed),
                violations=violations,
                output=output,
                returncode=run.returncode,
                metrics=metrics,
            )

    def run_all(
        self,
        *,
        env: dict[str, str],
        cwd: str | Path,
        python_executable: str | None = None,
    ) -> list[CodeAgentBenchmarkResult]:
        return [
            self.run_task(task, env=env, cwd=cwd, python_executable=python_executable)
            for task in self.tasks
        ]


class CodeAgentBenchmarkRunner:
    def __init__(
        self,
        tasks: Optional[list[CodeAgentBenchmarkTask]] = None,
        *,
        model: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        cwd: str | Path | None = None,
        python_executable: str | None = None,
        agent_args: Optional[list[str]] = None,
        stream_output=None,
    ):
        self.tasks = list(tasks or CODE_AGENT_TASKS)
        self.model = model or os.getenv("AGENTLIB_CODE_AGENT_MODEL") or "test-code-agent"
        self.env = dict(env or os.environ)
        self.cwd = str(Path(cwd or Path.cwd()).resolve())
        self.python_executable = python_executable
        self.agent_args = list(agent_args or [])
        self.stream_output = stream_output

    def _isolated_env(self, temp_root: Path) -> dict[str, str]:
        env = dict(self.env)
        env.setdefault("AGENTLIB_SESSION_DB", str(temp_root / "sessions.db"))
        env.setdefault("AGENTLIB_CLI_HISTORY_DB", str(temp_root / "cli_history.db"))
        return env

    def run(self) -> BenchmarkRunResult:
        task_results: list[BenchmarkTaskResult] = []
        with tempfile.TemporaryDirectory(prefix="agentlib-code-agent-bench-") as temp_dir:
            env = self._isolated_env(Path(temp_dir))
            suite = CodeAgentBenchmarkSuite(
                self.tasks,
                model=self.model,
                agent_args=self.agent_args,
                stream_output=self.stream_output,
            )
            for task in self.tasks:
                item = suite.run_task(
                    task,
                    env=env,
                    cwd=self.cwd,
                    python_executable=self.python_executable,
                )
                scores = full_scores(task.weights)
                scores = apply_violations(scores, item.violations)
                total_score, total_possible = finalize_scores(scores)
                task_results.append(BenchmarkTaskResult(
                    task_id=item.task_id,
                    passed=item.passed,
                    result="",
                    error=(
                        str(item.metrics.get("timeout_error"))
                        if item.metrics.get("timeout_error")
                        else (None if item.returncode == 0 else f"PTY exited with status {item.returncode}")
                    ),
                    score_by_category=scores,
                    total_score=total_score,
                    total_possible=total_possible,
                    metrics=dict(item.metrics, pty_returncode=item.returncode, output=item.output),
                    violations=item.violations,
                ))

        totals_by_category: dict[str, BenchmarkCategoryScore] = {}
        for result in task_results:
            for name, score in result.score_by_category.items():
                bucket = totals_by_category.setdefault(
                    name,
                    BenchmarkCategoryScore(earned=0.0, possible=0.0, details=[]),
                )
                bucket.earned += score.earned
                bucket.possible += score.possible
        total_score = sum(score.earned for score in totals_by_category.values())
        total_possible = sum(score.possible for score in totals_by_category.values())
        return BenchmarkRunResult(
            model=self.model,
            task_results=task_results,
            totals_by_category=totals_by_category,
            total_score=total_score,
            total_possible=total_possible,
            usage={
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "cost": 0.0,
                "requests": 0,
                "model": self.model,
            },
        )


def build_code_agent_test_env(
    tmp_path: Path,
    *,
    port: int,
    model_alias: str = "test-code-agent",
    provider_name: str = "testlocal",
    extra_env: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    home = tmp_path / "home"
    agentlib_dir = home / ".agentlib"
    agentlib_dir.mkdir(parents=True, exist_ok=True)
    (agentlib_dir / "config.py").write_text(
        "\n".join([
            "register_provider(",
            f"    {provider_name!r},",
            '    host="127.0.0.1",',
            '    path="/v1/chat/completions",',
            f"    port={port},",
            "    timeout=10,",
            "    tpm=1000,",
            "    concurrency=5,",
            "    tools=False,",
            '    api_type="completions",',
            ")",
            "register_model(",
            f"    {provider_name!r},",
            '    "tiny",',
            f"    aliases={model_alias!r},",
            '    model="tiny",',
            "    input_cost=0.0,",
            "    output_cost=0.0,",
            ")",
            f"code_agent_model = {model_alias!r}",
            "",
        ])
    )
    env = dict(os.environ)
    pythonpath_entries = []
    seen = set()
    for entry in [str(Path("src").resolve()), *[p for p in sys.path if p]]:
        if entry not in seen:
            seen.add(entry)
            pythonpath_entries.append(entry)
    env.update({
        "HOME": str(home),
        "TESTLOCAL_API_KEY": "dummy",
        "AGENTLIB_SESSION_DB": str(tmp_path / "sessions.db"),
        "AGENTLIB_CLI_HISTORY_DB": str(tmp_path / "cli_history.db"),
        "PYTHONPATH": os.pathsep.join(pythonpath_entries),
    })
    if extra_env:
        env.update(extra_env)
    return env


def make_code_agent_checker(
    *,
    expected_final: str,
    required_tools: tuple[str, ...] = (),
    min_turns: int | None = None,
    max_turns: int | None = None,
    forbid_bash_python: bool = True,
    require_isolated_dbs: bool = True,
) -> Checker:
    def checker(ctx: CodeAgentBenchmarkContext) -> tuple[bool, list[BenchmarkViolation]]:
        violations: list[BenchmarkViolation] = []
        if ctx.run.returncode != 0:
            violations.append(make_violation(
                "nonzero_exit",
                f"code-agent exited with status {ctx.run.returncode}",
                1.0,
                "correctness",
            ))
        if ctx.final_line != expected_final:
            violations.append(make_violation(
                "wrong_result",
                f"expected final line {expected_final!r}, got {ctx.final_line!r}",
                1.0,
                "correctness",
            ))
        for tool_name in required_tools:
            if not ctx.saw_tool(tool_name):
                violations.append(make_violation(
                    "missing_tool_use",
                    f"expected transcript to use {tool_name}()",
                    1.0,
                    "environment_usage",
                ))
        if min_turns is not None and ctx.turn_count < min_turns:
            violations.append(make_violation(
                "too_few_turns",
                f"expected at least {min_turns} turns, saw {ctx.turn_count}",
                1.0,
                "turn_efficiency",
            ))
        if max_turns is not None and ctx.turn_count > max_turns:
            violations.append(make_violation(
                "too_many_turns",
                f"expected at most {max_turns} turns, saw {ctx.turn_count}",
                1.0,
                "turn_efficiency",
            ))
        if forbid_bash_python and ctx.saw_bash_python():
            violations.append(make_violation(
                "bash_python",
                "used bash('python ...') instead of direct repo tools / Python REPL",
                1.0,
                "repl_proficiency",
            ))
        if require_isolated_dbs:
            if not ctx.session_db.exists():
                violations.append(make_violation(
                    "missing_session_db",
                    "isolated session db was not created",
                    1.0,
                    "environment_usage",
                ))
            if not ctx.history_db.exists():
                violations.append(make_violation(
                    "missing_history_db",
                    "isolated cli history db was not created",
                    1.0,
                    "environment_usage",
                ))
            home_text = str(ctx.home.resolve())
            for db_path in (ctx.session_db, ctx.history_db):
                resolved = str(db_path.resolve())
                if resolved.startswith(str(Path.home())) and not resolved.startswith(home_text):
                    violations.append(make_violation(
                        "sqlite_pollution",
                        f"db path escaped isolated HOME: {resolved}",
                        1.0,
                        "environment_usage",
                    ))
        return (not violations), violations
    return checker


def _contains_all(text: str, parts: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return all(part.lower() in lowered for part in parts)


def checker_sqlite_isolation_explanation(ctx: CodeAgentBenchmarkContext) -> tuple[bool, list[BenchmarkViolation]]:
    passed, violations = make_code_agent_checker(
        expected_final=ctx.final_line,
        required_tools=("grep", "read"),
        min_turns=2,
        max_turns=4,
    )(ctx)
    answer = ctx.response_block or ctx.final_line
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        try:
            payload = json.loads(answer.replace("\n", ""))
        except json.JSONDecodeError:
            violations.append(make_violation(
                "invalid_json",
                "expected final answer to be JSON",
                1.0,
                "correctness",
            ))
            return False, violations

    session_src = str(payload.get("session_db_source", ""))
    history_src = str(payload.get("history_db_source", ""))
    reason = str(payload.get("reason", ""))
    if "AGENTLIB_SESSION_DB" not in session_src:
        violations.append(make_violation(
            "missing_session_source",
            "session_db_source should mention AGENTLIB_SESSION_DB",
            1.0,
            "correctness",
        ))
    if "AGENTLIB_CLI_HISTORY_DB" not in history_src:
        violations.append(make_violation(
            "missing_history_source",
            "history_db_source should mention AGENTLIB_CLI_HISTORY_DB",
            1.0,
            "correctness",
        ))
    reason_ok = (
        ("env" in reason.lower() or "environment" in reason.lower())
        and ("override" in reason.lower() or "overrides" in reason.lower())
        and ("sqlite" in reason.lower() or "db" in reason.lower() or "path" in reason.lower())
    )
    if not reason_ok:
        violations.append(make_violation(
            "weak_reason",
            "reason should explain env-var path override for isolated sqlite files",
            1.0,
            "correctness",
        ))
    if ctx.first_tool_offset() == -1 or (
        ctx.final_answer_offset() != -1 and ctx.final_answer_offset() < ctx.first_tool_offset()
    ):
        violations.append(make_violation(
            "early_release",
            "final answer appeared before repo inspection",
            1.0,
            "instruction_following",
        ))
    return (not violations), violations


def checker_resume_flow_summary(ctx: CodeAgentBenchmarkContext) -> tuple[bool, list[BenchmarkViolation]]:
    passed, violations = make_code_agent_checker(
        expected_final=ctx.final_line,
        required_tools=("read",),
        min_turns=2,
        max_turns=4,
    )(ctx)
    answer = ctx.response_block or ctx.final_line
    normalized = answer.lower()
    required_groups = (
        ("resume_session", "/resume"),
        ("replay_session_into_agent", "replay"),
        ("_replay_display_output", "display output"),
        ("system_reset", "repl session has been reset"),
    )
    for group in required_groups:
        if not any(token.lower() in normalized for token in group):
            violations.append(make_violation(
                "missing_resume_step",
                f"answer missing one of: {group}",
                1.0,
                "correctness",
            ))
    if ";" not in answer and "->" not in answer:
        violations.append(make_violation(
            "unstructured_summary",
            "expected a compact ordered summary using ';' or '->'",
            1.0,
            "instruction_following",
        ))
    return (not violations), violations


def checker_read_vs_view_file(ctx: CodeAgentBenchmarkContext) -> tuple[bool, list[BenchmarkViolation]]:
    passed, violations = make_code_agent_checker(
        expected_final=ctx.final_line,
        required_tools=("read",),
        min_turns=2,
        max_turns=4,
    )(ctx)
    answer = ctx.response_block or ctx.final_line
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        try:
            payload = json.loads(answer.replace("\n", ""))
        except json.JSONDecodeError:
            violations.append(make_violation(
                "invalid_json",
                "expected final answer to be JSON",
                1.0,
                "correctness",
            ))
            return False, violations

    read_desc = str(payload.get("read", ""))
    view_desc = str(payload.get("view_file", ""))
    if not _contains_all(read_desc, ("text",)):
        violations.append(make_violation(
            "missing_read_semantics",
            "read description should mention text/value semantics",
            1.0,
            "correctness",
        ))
    if "attachment" not in view_desc.lower() and "attach" not in view_desc.lower():
        violations.append(make_violation(
            "missing_view_attachment",
            "view_file description should mention attachment behavior",
            1.0,
            "correctness",
        ))
    if "line" not in view_desc.lower() and "number" not in view_desc.lower():
        violations.append(make_violation(
            "missing_view_numbering",
            "view_file description should mention numbered display/lines",
            1.0,
            "correctness",
        ))
    return (not violations), violations


def prepare_file_edit_patch_task(task_dir: Path, cwd: Path) -> dict[str, object]:
    target_file = task_dir / "edit_target.txt"
    original = "\n".join([
        "TITLE: benchmark",
        "STATUS: pending",
        "OWNER: code-agent",
        "",
    ])
    expected = "\n".join([
        "TITLE: benchmark",
        "STATUS: done",
        "OWNER: code-agent",
        "",
    ])
    target_file.write_text(original)
    return {
        "prompt": (
            f"Read {target_file} first, then update only the status line from 'STATUS: pending' "
            f"to 'STATUS: done' using apply_patch. Emit only UPDATED when finished."
        ),
        "metadata": {
            "target_file": str(target_file),
            "expected_content": expected,
        },
    }


def checker_file_edit_patch(ctx: CodeAgentBenchmarkContext) -> tuple[bool, list[BenchmarkViolation]]:
    passed, violations = make_code_agent_checker(
        expected_final="UPDATED",
        required_tools=("read", "apply_patch"),
        min_turns=1,
        max_turns=4,
    )(ctx)
    target_file = Path(str(ctx.metadata.get("target_file", "")))
    expected_content = str(ctx.metadata.get("expected_content", ""))
    if not target_file.exists():
        violations.append(make_violation(
            "missing_target_file",
            "temporary edit target file was not found during validation",
            1.0,
            "correctness",
        ))
        return False, violations
    actual = target_file.read_text()
    if actual != expected_content:
        violations.append(make_violation(
            "wrong_file_content",
            f"edited file content did not match expected result: {actual!r}",
            1.0,
            "correctness",
        ))
    read_offset = ctx.tool_offset("read")
    patch_offset = ctx.tool_offset("apply_patch")
    if read_offset == -1 or patch_offset == -1 or read_offset > patch_offset:
        violations.append(make_violation(
            "read_before_edit_required",
            "expected the file to be read before patching",
            1.0,
            "instruction_following",
        ))
    return (not violations), violations


CODE_AGENT_TASKS = [
    CodeAgentBenchmarkTask(
        id="code-agent/repo-session-db-var",
        prompt="In this repo, scan the codebase and emit only the environment variable that overrides the session sqlite path.",
        description="Requires scanning the repo to answer a codebase question.",
        checker=make_code_agent_checker(
            expected_final="AGENTLIB_SESSION_DB",
            required_tools=("grep",),
            max_turns=2,
        ),
        wait_for="AGENTLIB_SESSION_DB",
        tags=("code-agent", "repo-scan"),
    ),
    CodeAgentBenchmarkTask(
        id="code-agent/repo-cli-history-db-var",
        prompt="Inspect the repo and emit only the environment variable used for the CLI history sqlite path override.",
        description="Requires code search in the real repository.",
        checker=make_code_agent_checker(
            expected_final="AGENTLIB_CLI_HISTORY_DB",
            required_tools=("grep",),
            max_turns=2,
        ),
        wait_for="AGENTLIB_CLI_HISTORY_DB",
        tags=("code-agent", "repo-scan"),
    ),
    CodeAgentBenchmarkTask(
        id="code-agent/multi-turn-synthetic-exchange",
        prompt="Use two turns: first inspect src/agentlib/agents/code_agent.py with repo tools, then on the next turn emit only the exact first progress text from the synthetic exchange.",
        description="Requires multi-step tool use across turns before answering.",
        checker=make_code_agent_checker(
            expected_final="Checking today's date...",
            required_tools=("read",),
            min_turns=2,
            max_turns=3,
        ),
        wait_for="Checking today's date...",
        tags=("code-agent", "multi-turn"),
    ),
    CodeAgentBenchmarkTask(
        id="code-agent/tool-discipline-preview-target-count",
        prompt="Without using bash to invoke python, inspect src/agentlib/agents/code_agent.py and emit only the number of tool names in _preview_targets.",
        description="Penalizes weak tool use like bash('python ...') for local inspection.",
        checker=make_code_agent_checker(
            expected_final="3",
            required_tools=("read",),
            max_turns=2,
            forbid_bash_python=True,
        ),
        wait_for="3",
        tags=("code-agent", "tool-discipline"),
    ),
    CodeAgentBenchmarkTask(
        id="code-agent/sqlite-isolation-explanation",
        prompt="Inspect the repo and explain why CodeAgent benchmark sqlite state can be isolated. Emit JSON with keys session_db_source, history_db_source, and reason.",
        description="Requires multi-file understanding of how session and CLI history sqlite paths are selected.",
        checker=checker_sqlite_isolation_explanation,
        max_turns=4,
        wait_for="reason",
        tags=("code-agent", "architecture", "repo-understanding"),
    ),
    CodeAgentBenchmarkTask(
        id="code-agent/resume-flow-summary",
        prompt="Inspect the repo and summarize what happens when /resume succeeds. Emit one compact ordered summary using ';' or '->'.",
        description="Requires tracing behavior across CodeAgent resume and session replay code.",
        checker=checker_resume_flow_summary,
        max_turns=4,
        wait_for="resume",
        tags=("code-agent", "architecture", "flow"),
    ),
    CodeAgentBenchmarkTask(
        id="code-agent/read-vs-view-file",
        prompt="Inspect the repo and emit JSON with keys read and view_file explaining the functional difference between those tools.",
        description="Requires understanding how CodeAgent treats text reads versus numbered attachment views.",
        checker=checker_read_vs_view_file,
        max_turns=4,
        wait_for="view_file",
        tags=("code-agent", "tool-semantics"),
    ),
    CodeAgentBenchmarkTask(
        id="code-agent/file-edit-patch",
        prompt="placeholder",
        description="Requires reading a temp file, editing it with apply_patch, and producing the correct final content.",
        checker=checker_file_edit_patch,
        max_turns=4,
        wait_for="UPDATED",
        tags=("code-agent", "editing", "patch"),
        prepare=prepare_file_edit_patch_task,
    ),
]


__all__ = [
    "CODE_AGENT_TASKS",
    "CodeAgentBenchmarkContext",
    "CodeAgentBenchmarkResult",
    "CodeAgentBenchmarkRunner",
    "CodeAgentBenchmarkSuite",
    "CodeAgentBenchmarkTask",
    "build_code_agent_test_env",
    "make_code_agent_checker",
]