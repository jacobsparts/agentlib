import json
import io
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agentlib.repl_benchmark.code_agent_benchmark import (
    CODE_AGENT_TASKS,
    CodeAgentBenchmarkRunner,
    CodeAgentBenchmarkSuite,
    _sqlite_isolation_reason_ok,
    build_code_agent_test_env,
)


class _ScriptedModelServer:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self._server = None
        self._thread = None
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()

    def __enter__(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                payload = json.loads(self.rfile.read(length))
                outer.requests.append(payload)
                next_item = outer._responses.pop(0) if outer._responses else 'emit("fallback", release=True)'
                content = next_item(payload) if callable(next_item) else next_item
                response = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
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

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1)
        return False


def _run_task(tmp_path, task, responses):
    with _ScriptedModelServer(responses) as server:
        env = build_code_agent_test_env(tmp_path, port=server.port)
        suite = CodeAgentBenchmarkSuite([task])
        result = suite.run_task(task, env=env, cwd=Path.cwd())
        return result, env, server


def _codes(result):
    return {violation.code for violation in result.violations}


def test_code_agent_benchmark_tasks_allow_repo_discovery_turn_budget():
    assert all(task.max_turns >= 30 for task in CODE_AGENT_TASKS)


def test_sqlite_isolation_reason_accepts_semantic_redirect_wording():
    assert _sqlite_isolation_reason_ok(
        "The benchmark harness sets AGENTLIB_SESSION_DB and AGENTLIB_CLI_HISTORY_DB "
        "to temp-directory sqlite files, so both persisted session events and CLI "
        "input history are redirected away from the user's real HOME state."
    )
    assert not _sqlite_isolation_reason_ok("It uses sqlite somewhere.")


def test_code_agent_benchmark_repo_scan_session_db_var(tmp_path):
    task = next(task for task in CODE_AGENT_TASKS if task.id == "code-agent/repo-session-db-var")
    result, env, server = _run_task(
        tmp_path,
        task,
        [
            'preview(grep("AGENTLIB_SESSION_DB", "src", None, None, False, 2, False, False))\nemit("AGENTLIB_SESSION_DB", release=True)',
        ],
    )
    assert result.passed, (result.output, _codes(result), [(v.code, v.message) for v in result.violations])
    assert result.returncode == 0
    assert Path(env["AGENTLIB_SESSION_DB"]).exists()
    assert Path(env["AGENTLIB_CLI_HISTORY_DB"]).exists()
    assert len(server.requests) >= 1


def test_code_agent_benchmark_counts_syntax_retry(tmp_path):
    task = next(task for task in CODE_AGENT_TASKS if task.id == "code-agent/repo-session-db-var")
    result, env, server = _run_task(
        tmp_path,
        task,
        [
            "not valid python !!!",
            'preview(grep("AGENTLIB_SESSION_DB", "src", None, None, False, 2, False, False))\nemit("AGENTLIB_SESSION_DB", release=True)',
        ],
    )
    assert result.returncode == 0
    assert result.metrics["syntax_retries"] == 1
    assert "syntax_retry" in _codes(result)
    assert any(v.category == "syntax_errors" for v in result.violations)
    assert len(server.requests) >= 2


def test_code_agent_benchmark_accepts_quoted_final_line(tmp_path):
    task = next(task for task in CODE_AGENT_TASKS if task.id == "code-agent/repo-session-db-var")
    result, env, server = _run_task(
        tmp_path,
        task,
        [
            'preview(grep("AGENTLIB_SESSION_DB", "src", None, None, False, 2, False, False))\nemit("`AGENTLIB_SESSION_DB`", release=True)',
        ],
    )
    assert result.passed, (result.output, _codes(result), [(v.code, v.message) for v in result.violations])
    assert result.returncode == 0
    assert "wrong_result" not in _codes(result)


def test_code_agent_benchmark_repo_scan_cli_history_var(tmp_path):
    task = next(task for task in CODE_AGENT_TASKS if task.id == "code-agent/repo-cli-history-db-var")
    result, env, server = _run_task(
        tmp_path,
        task,
        [
            'preview(grep("AGENTLIB_CLI_HISTORY_DB", "src", None, None, False, 2, False, False))\nemit("AGENTLIB_CLI_HISTORY_DB", release=True)',
        ],
    )
    assert result.passed, (result.output, _codes(result), [(v.code, v.message) for v in result.violations])
    assert result.returncode == 0
    assert Path(env["AGENTLIB_SESSION_DB"]).exists()
    assert Path(env["AGENTLIB_CLI_HISTORY_DB"]).exists()
    assert len(server.requests) >= 1


def test_code_agent_benchmark_multi_turn_repo_task(tmp_path):
    task = next(task for task in CODE_AGENT_TASKS if task.id == "code-agent/multi-turn-synthetic-exchange")
    result, env, server = _run_task(
        tmp_path,
        task,
        [
            '_code = read("src/agentlib/agents/code_agent.py")\npreview(_code)',
            'emit("Checking today\'s date...", release=True)',
        ],
    )
    assert result.passed, (result.output, _codes(result), [(v.code, v.message) for v in result.violations])
    assert result.returncode == 0
    assert len(server.requests) >= 2
    assert Path(env["AGENTLIB_SESSION_DB"]).exists()
    assert Path(env["AGENTLIB_CLI_HISTORY_DB"]).exists()


def test_code_agent_benchmark_flags_bash_python_misuse(tmp_path):
    task = next(task for task in CODE_AGENT_TASKS if task.id == "code-agent/tool-discipline-preview-target-count")
    result, env, server = _run_task(
        tmp_path,
        task,
        [
            'bash("python -c \"print(3)\"")\nemit("3", release=True)',
        ],
    )
    codes = {violation.code for violation in result.violations}
    assert result.passed is False
    assert "bash_python" in codes
    assert result.returncode == 0
    assert len(server.requests) >= 1
    assert Path(env["AGENTLIB_SESSION_DB"]).exists()
    assert Path(env["AGENTLIB_CLI_HISTORY_DB"]).exists()


def test_code_agent_benchmark_sqlite_isolation_explanation(tmp_path):
    task = next(task for task in CODE_AGENT_TASKS if task.id == "code-agent/sqlite-isolation-explanation")
    result, env, server = _run_task(
        tmp_path,
        task,
        [
            'preview(grep("AGENTLIB_SESSION_DB|AGENTLIB_CLI_HISTORY_DB", "src", None, None, False, 2, False, False))',
            'import json\n_s = read("src/agentlib/session_store.py")\n_h = read("src/agentlib/cli/mixin.py")\nemit(json.dumps({"session_db_source":"AGENTLIB_SESSION_DB in session_store.resolve_db_path","history_db_source":"AGENTLIB_CLI_HISTORY_DB in SQLiteHistory.__init__","reason":"Environment variable overrides force each sqlite path to a temp test db, so benchmark state stays isolated."}), release=True)',
        ],
    )
    assert result.passed, (result.output, _codes(result), [(v.code, v.message) for v in result.violations])
    assert result.returncode == 0
    assert len(server.requests) >= 2
    assert Path(env["AGENTLIB_SESSION_DB"]).exists()
    assert Path(env["AGENTLIB_CLI_HISTORY_DB"]).exists()


def test_code_agent_benchmark_resume_flow_summary(tmp_path):
    task = next(task for task in CODE_AGENT_TASKS if task.id == "code-agent/resume-flow-summary")
    result, env, server = _run_task(
        tmp_path,
        task,
        [
            '_a = read("src/agentlib/agents/code_agent.py")\n_b = read("src/agentlib/session_replay.py")\npreview(_a)',
            'emit("resume_session -> replay_session_into_agent -> _replay_display_output; then usermsg adds system_reset / REPL session has been reset", release=True)',
        ],
    )
    assert result.passed, (result.output, _codes(result), [(v.code, v.message) for v in result.violations])
    assert result.returncode == 0
    assert len(server.requests) >= 2
    assert Path(env["AGENTLIB_SESSION_DB"]).exists()
    assert Path(env["AGENTLIB_CLI_HISTORY_DB"]).exists()


def test_code_agent_benchmark_read_vs_view_semantics(tmp_path):
    task = next(task for task in CODE_AGENT_TASKS if task.id == "code-agent/read-vs-view-file")
    result, env, server = _run_task(
        tmp_path,
        task,
        [
            '_code = read("src/agentlib/agents/code_agent.py")\npreview(_code)',
            'import json\nemit(json.dumps({"read":"returns file contents as text for use as a Python value","view":"shows numbered lines and attachment behavior for conversation context"}), release=True)',
        ],
    )
    assert result.passed, (result.output, _codes(result), [(v.code, v.message) for v in result.violations])
    assert result.returncode == 0
    assert len(server.requests) >= 2
    assert Path(env["AGENTLIB_SESSION_DB"]).exists()
    assert Path(env["AGENTLIB_CLI_HISTORY_DB"]).exists()


def test_code_agent_benchmark_runner_can_stream_output(tmp_path):
    task = next(task for task in CODE_AGENT_TASKS if task.id == "code-agent/repo-session-db-var")
    stream = io.StringIO()
    with _ScriptedModelServer([
        'preview(grep("AGENTLIB_SESSION_DB", "src", None, None, False, 2, False, False))\nemit("AGENTLIB_SESSION_DB", release=True)',
    ]) as server:
        env = build_code_agent_test_env(tmp_path, port=server.port)
        result = CodeAgentBenchmarkRunner(
            tasks=[task],
            model="test-code-agent",
            env=env,
            cwd=Path.cwd(),
            stream_output=stream,
        ).run()
    assert len(result.task_results) == 1
    assert result.task_results[0].task_id == "code-agent/repo-session-db-var"
    assert "Code Agent" in stream.getvalue()
    assert "AGENTLIB_SESSION_DB" in stream.getvalue()
def test_code_agent_benchmark_file_edit_patch_tracks_attempts_and_cleans_up(tmp_path):
    task = next(task for task in CODE_AGENT_TASKS if task.id == "code-agent/file-edit-patch")
    state = {"target": None}

    def first_response(payload):
        text = payload["messages"][-1]["content"]
        match = re.search(r"Read (.+?) first, then update only the status line", text)
        assert match, text
        state["target"] = match.group(1)
        return f"read({state['target']!r})"

    def second_response(payload):
        return (
            'apply_patch("""*** Begin Patch\n'
            f'*** Update File: {state["target"]}\n'
            '@@\n'
            '-STATUS: waiting\n'
            '+STATUS: done\n'
            '*** End Patch""")'
        )

    def third_response(payload):
        return (
            'apply_patch("""*** Begin Patch\n'
            f'*** Update File: {state["target"]}\n'
            '@@\n'
            '-STATUS: pending\n'
            '+STATUS: done\n'
            '*** End Patch""")\n'
            'emit("UPDATED", release=True)'
        )

    with _ScriptedModelServer([first_response, second_response, third_response]) as server:
        env = build_code_agent_test_env(tmp_path, port=server.port)
        suite = CodeAgentBenchmarkSuite([task])
        result = suite.run_task(task, env=env, cwd=Path.cwd())

    assert result.passed, (result.output, _codes(result), [(v.code, v.message) for v in result.violations])
    assert result.metrics["file_edit_attempts"] >= 2
    assert result.metrics["failed_edit_attempts"] >= 1
    assert result.metrics["turns"] >= 2
    target_file = Path(result.metrics["target_file"])
    task_dir = Path(result.metrics["task_dir"])
    assert not target_file.exists()
    assert not task_dir.exists()
