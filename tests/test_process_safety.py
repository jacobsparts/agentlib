import subprocess
import sys

import agentlib.tools.subshell as subshell
import agentlib.tools.transports as transports


class _FakeQueue:
    def put(self, _item):
        pass


class _FakeProcess:
    pid = None

    def start(self):
        pass

    def is_alive(self):
        return False


class _FakeContext:
    def Queue(self, **_kwargs):
        return _FakeQueue()

    def Process(self, **_kwargs):
        return _FakeProcess()


def test_multiprocessing_transport_explicitly_uses_spawn(monkeypatch):
    methods = []
    monkeypatch.setattr(
        transports.mp,
        "get_context",
        lambda method: methods.append(method) or _FakeContext(),
    )

    transport = transports.MultiprocessingTransport(lambda: None)
    transport.start()

    assert methods == ["spawn"]


def test_subshell_explicitly_uses_spawn(monkeypatch):
    methods = []
    monkeypatch.setattr(
        subshell.mp,
        "get_context",
        lambda method: methods.append(method) or _FakeContext(),
    )

    shell = subshell.SubShell()
    shell._ensure_session()

    assert methods == ["spawn"]


def test_external_fork_fails_fast():
    script = r"""
import os
import threading
from agentlib.process_safety import assert_safe_process_context

ready = threading.Event()
stop = threading.Event()

def hold_thread():
    ready.set()
    stop.wait()

thread = threading.Thread(target=hold_thread)
thread.start()
ready.wait()
read_fd, write_fd = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(read_fd)
    try:
        assert_safe_process_context()
    except RuntimeError as exc:
        os.write(write_fd, str(exc).encode())
        os._exit(0)
    os._exit(2)

os.close(write_fd)
message = os.read(read_fd, 4096).decode()
_, status = os.waitpid(pid, 0)
stop.set()
thread.join()
assert os.waitstatus_to_exitcode(status) == 0
assert "created with fork" in message
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


def test_provider_admission_rejects_external_fork():
    script = r"""
import os
import tempfile
from agentlib.provider_admission import ProviderAdmission

pid = os.fork()
if pid == 0:
    try:
        root = tempfile.mkdtemp()
        ProviderAdmission(
            "unsafe-fork",
            1,
            request_timeout=1,
            db_path=os.path.join(root, "admission.sqlite3"),
            notify_path=os.path.join(root, "admission.notify"),
        )
    except RuntimeError as exc:
        os._exit(0 if "created with fork" in str(exc) else 3)
    os._exit(2)

_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)
