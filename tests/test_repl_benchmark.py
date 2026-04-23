
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

from agentlib.repl_benchmark import BenchmarkTask, REPLBenchmarkRunner, discover_tasks, format_summary, register_task
from agentlib.repl_benchmark.core import BenchmarkTaskContext, checker_expected_int, default_checker
from agentlib.repl_benchmark.registry import task_registry


class FakeUsageTracker:
    def __init__(self):
        self.history = []

    def _normalize(self, model_name, usage):
        return usage


class FakeLLMClient:
    def __init__(self):
        self.usage_tracker = FakeUsageTracker()


class FakeAgent:
    model = "fake-model"

    def __init__(self):
        self.llm_client = FakeLLMClient()
        self.messages = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def usermsg(self, msg):
        self.messages.append(msg)

    def run_loop(self, max_turns=10, max_syntax_retries=3):
        self.on_repl_execute(None)
        self._on_assistant_message_committed({"content": 'emit("55", release=True)'})
        self.on_statement_output([("echo", "x = 1\n"), ("echo", 'emit("55", release=True)\n')])
        self.llm_client.usage_tracker.history.append((
            self.model,
            {
                "prompt_tokens": 10,
                "cached_tokens": 0,
                "completion_tokens": 5,
                "reasoning_tokens": 0,
                "cost": 0.01,
            },
        ))
        return "55"


class FakeBadAgent(FakeAgent):
    def run_loop(self, max_turns=10, max_syntax_retries=3):
        self.on_repl_execute(None)
        self.on_repl_execute(None)
        self.on_retry("syntax", 1)
        self._on_assistant_message_committed({"content": 'bash("python -c \'print(55)\'")'})
        self.on_repl_chunk("working", "progress")
        self.on_statement_output([("echo", 'bash("python -c \'print(55)\'")\n')])
        self.llm_client.usage_tracker.history.append((
            self.model,
            {
                "prompt_tokens": 12,
                "cached_tokens": 0,
                "completion_tokens": 7,
                "reasoning_tokens": 0,
                "cost": 0.02,
            },
        ))
        return "55"


def test_default_checker_missing_release():
    task = BenchmarkTask(id="t", prompt="p", checker=default_checker)
    ctx = BenchmarkTaskContext(
        task=task,
        agent=None,
        metrics={"turns": 1, "release_called": False, "syntax_retries": 0, "runtime_errors": 0, "saw_bash_python": False},
        result="x",
    )
    passed, violations, scores = default_checker(ctx)
    assert passed is True
    assert any(v.code == "missing_release" for v in violations)
    assert scores["completion_behavior"].earned < scores["completion_behavior"].possible


def test_discover_tasks_from_path(tmp_path):
    task_registry.clear()
    module = tmp_path / "bench_one.py"
    module.write_text(
        "from agentlib.repl_benchmark import BenchmarkTask, register_task\n"
        "from agentlib.repl_benchmark.core import default_checker\n"
        "register_task(BenchmarkTask(id='tmp/task', prompt='hi', checker=default_checker))\n"
    )
    tasks = discover_tasks(paths=[tmp_path], include_builtin=False)
    assert [task.id for task in tasks] == ["tmp/task"]


def test_builtin_task_discovery():
    task_registry.clear()
    tasks = discover_tasks(include_builtin=True)
    ids = {task.id for task in tasks}
    assert "basic/arithmetic" in ids
    assert "behavior/no-bash-python" in ids


def test_runner_returns_summary():
    task_registry.clear()
    register_task(BenchmarkTask(id="fake/task", prompt="hello", checker=checker_expected_int(55)))
    runner = REPLBenchmarkRunner(FakeAgent, include_builtin=False)
    result = runner.run()
    assert result.model == "fake-model"
    assert len(result.task_results) == 1
    assert result.task_results[0].task_id == "fake/task"
    assert result.usage["requests"] == 1
    assert "Grand Total" in format_summary(result)


def test_penalties_keep_room_for_improvement():
    task_registry.clear()
    register_task(BenchmarkTask(id="fake/task", prompt="hello", checker=checker_expected_int(55)))
    runner = REPLBenchmarkRunner(FakeBadAgent, include_builtin=False)
    result = runner.run()
    item = result.task_results[0]
    assert item.total_score < item.total_possible
    codes = {v.code for v in item.violations}
    assert "bash_python" in codes
    assert "missing_release" in codes
    assert "syntax_retry" in codes


def test_cli_json_round_trip():
    task_registry.clear()
    register_task(BenchmarkTask(id="fake/task", prompt="hello", checker=checker_expected_int(55)))
    runner = REPLBenchmarkRunner(FakeAgent, include_builtin=False)
    payload = json.dumps(runner.run().to_dict())
    obj = json.loads(payload)
    assert obj["model"] == "fake-model"


def test_cli_help_without_warning():
    proc = subprocess.run(
        [sys.executable, "-m", "agentlib.repl_benchmark.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Run the REPL benchmark suite." in proc.stdout
    assert "RuntimeWarning" not in proc.stderr


def test_executable_without_args_shows_help():
    proc = subprocess.run(
        [str(Path("repl-benchmark").resolve())],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Run the REPL benchmark suite." in proc.stdout
    assert "usage:" in proc.stdout


def test_cli_human_summary_with_fake_agent(tmp_path):
    task_module = tmp_path / "task_mod.py"
    task_module.write_text(
        "from agentlib.repl_benchmark import BenchmarkTask, register_task\n"
        "from agentlib.repl_benchmark.core import checker_expected_int\n"
        "register_task(BenchmarkTask(id='cli/task', prompt='hello', checker=checker_expected_int(55)))\n"
    )
    agent_module = tmp_path / "fake_agent_mod.py"
    agent_module.write_text(
        dedent('''
        class FakeUsageTracker:
            def __init__(self):
                self.history = []

            def _normalize(self, model_name, usage):
                return usage

        class FakeLLMClient:
            def __init__(self):
                self.usage_tracker = FakeUsageTracker()

        class FakeAgent:
            model = "fake-cli-model"

            def __init__(self):
                self.llm_client = FakeLLMClient()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def usermsg(self, msg):
                self.msg = msg

            def run_loop(self, max_turns=10, max_syntax_retries=3):
                self.on_repl_execute(None)
                self._on_assistant_message_committed({"content": 'emit("55", release=True)'})
                self.on_statement_output([("echo", 'emit("55", release=True)\\n')])
                self.llm_client.usage_tracker.history.append((
                    self.model,
                    {
                        "prompt_tokens": 9,
                        "cached_tokens": 0,
                        "completion_tokens": 4,
                        "reasoning_tokens": 0,
                        "cost": 0.01,
                    },
                ))
                return "55"
        ''').strip() + "\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentlib.repl_benchmark.cli",
            "--agent",
            "fake_agent_mod:FakeAgent",
            "--task-path",
            str(task_module),
            "--no-builtin",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "REPL Benchmark Summary" in proc.stdout
    assert "cli/task: PASS" in proc.stdout
