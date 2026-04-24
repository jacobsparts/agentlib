
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from textwrap import dedent
from http.server import BaseHTTPRequestHandler, HTTPServer

from agentlib.repl_benchmark import (
    BenchmarkTask,
    REPLBenchmarkRunner,
    build_code_agent_test_env,
    discover_tasks,
    format_summary,
    register_task,
    run_pty_session,
    strip_ansi,
)
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


def test_code_agent_cli_trivial_pty(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            self.rfile.read(length)
            payload = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": 'emit("4", release=True)',
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args, **kwargs):
            pass

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    home = tmp_path / "home"
    agentlib_dir = home / ".agentlib"
    agentlib_dir.mkdir(parents=True)
    (agentlib_dir / "config.py").write_text(dedent(f"""\
register_provider(
    "testlocal",
    host="127.0.0.1",
    path="/v1/chat/completions",
    port={port},
    timeout=10,
    tpm=1000,
    concurrency=5,
    tools=False,
    api_type="completions",
)
register_model(
    "testlocal",
    "tiny",
    aliases="test-code-agent",
    model="tiny",
    input_cost=0.0,
    output_cost=0.0,
)
code_agent_model = "test-code-agent"
"""))

    env = dict(os.environ)
    env["HOME"] = str(home)
    env["TESTLOCAL_API_KEY"] = "dummy"
    env["AGENTLIB_SESSION_DB"] = str(tmp_path / "sessions.db")
    env["AGENTLIB_CLI_HISTORY_DB"] = str(tmp_path / "cli_history.db")
    env["PYTHONPATH"] = '/home/jacob/agentlib/src:/usr/local/lib/python3.11/site-packages:/home/jacob/.local/lib/python3.11/site-packages'

    try:
        result = run_pty_session(
            [sys.executable, "-m", "agentlib.agents.code_agent", "--model", "test-code-agent"],
            inputs=["What is 2+2?\n"],
            env=env,
            cwd=str(Path.cwd()),
            timeout=20,
            wait_for="4",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    text = strip_ansi(result.output)
    assert result.returncode == 0
    assert "4" in text
    assert "Session ended. Goodbye!" in text
    assert Path(env["AGENTLIB_SESSION_DB"]).exists()
    assert Path(env["AGENTLIB_CLI_HISTORY_DB"]).exists()


def test_cli_includes_code_agent_builtin_suite(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        counters = {}

        @classmethod
        def next_response(cls, text):
            key_map = [
                ("YYYY-MM-DD", ['emit("2026-04-23", release=True)']),
                ("17 * 19 + 23", ['emit("346", release=True)']),
                ("agentlib benchmark", ['emit("AGENTLIB BENCHMARK", release=True)']),
                ("sum the numbers [5, 8, 13, 21]", ['emit("47", release=True)']),
                ("2 ** 10", ['emit("1024", release=True)']),
                ("release discipline", ['emit("18", release=True)']),
                ("12345 squared", ['emit("25", release=True)']),
                ("sum(range(1, 11))", ['emit("55", release=True)']),
                ("overrides the session sqlite path", ['preview(_v1 := grep("AGENTLIB_SESSION_DB", "src", None, None, False, 2, False, False))\nemit("AGENTLIB_SESSION_DB", release=True)']),
                ("CLI history sqlite path override", ['preview(_v1 := grep("AGENTLIB_CLI_HISTORY_DB", "src", None, None, False, 2, False, False))\nemit("AGENTLIB_CLI_HISTORY_DB", release=True)']),
                ("exact first progress text from the synthetic exchange", ['_code = read("src/agentlib/agents/code_agent.py")\npreview(_code)', 'emit("Checking today\'s date...", release=True)']),
                ("number of tool names in _preview_targets", ['_code = read("src/agentlib/agents/code_agent.py")\nemit("3", release=True)']),
                ("sqlite state can be isolated", ['preview(_v1 := grep("AGENTLIB_SESSION_DB|AGENTLIB_CLI_HISTORY_DB", "src", None, None, False, 2, False, False))', 'import json\n_s = read("src/agentlib/session_store.py")\n_h = read("src/agentlib/cli/mixin.py")\nemit(json.dumps({"session_db_source":"AGENTLIB_SESSION_DB in session_store.resolve_db_path","history_db_source":"AGENTLIB_CLI_HISTORY_DB in SQLiteHistory.__init__","reason":"Environment variable overrides force each sqlite path to a temp test db, so benchmark state stays isolated."}), release=True)']),
                ("/resume succeeds", ['_a = read("src/agentlib/agents/code_agent.py")\n_b = read("src/agentlib/session_replay.py")\npreview(_a)', 'emit("resume_session -> replay_session_into_agent -> _replay_display_output; then usermsg adds system_reset / REPL session has been reset", release=True)']),
                ("functional difference between those tools", ['_code = read("src/agentlib/agents/code_agent.py")\npreview(_code)', 'import json\nemit(json.dumps({"read":"returns file contents as text for use as a Python value","view_file":"shows numbered lines and attachment behavior for conversation context"}), release=True)']),
            ]
            for marker, responses in key_map:
                if marker in text:
                    idx = cls.counters.get(marker, 0)
                    cls.counters[marker] = idx + 1
                    return responses[min(idx, len(responses) - 1)]
            return 'emit("fallback", release=True)'

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            text = payload["messages"][-1]["content"]
            content = self.next_response(text)
            response = {
                "choices": [{
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args, **kwargs):
            pass

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = build_code_agent_test_env(tmp_path, port=port)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "agentlib.repl_benchmark.cli", "--model", "test-code-agent", "--json"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    payload = json.loads(proc.stdout)
    task_ids = {item["task_id"] for item in payload["task_results"]}
    assert "basic/arithmetic" in task_ids
    assert "code-agent/sqlite-isolation-explanation" in task_ids
    assert "code-agent/resume-flow-summary" in task_ids


def test_cli_can_stream_code_agent_repl_output(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            text = payload["messages"][-1]["content"]
            if "overrides the session sqlite path" in text:
                content = 'preview(_v1 := grep("AGENTLIB_SESSION_DB", "src", None, None, False, 2, False, False))\nemit("AGENTLIB_SESSION_DB", release=True)'
            else:
                content = 'emit("fallback", release=True)'
            response = {
                "choices": [{
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args, **kwargs):
            pass

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = build_code_agent_test_env(
        tmp_path,
        port=port,
        extra_env={"AGENTLIB_CODE_AGENT_MODEL": "test-code-agent"},
    )
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "agentlib.repl_benchmark.cli",
                "--model",
                "test-code-agent",
                "--no-builtin",
                "--show-repl-output",
                "--json",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert "Code Agent" in proc.stderr
    assert "AGENTLIB_SESSION_DB" in proc.stderr
    payload = json.loads(proc.stdout)
    task_ids = {item["task_id"] for item in payload["task_results"]}
    assert "code-agent/repo-session-db-var" in task_ids
