import contextlib
import sqlite3
import threading

from agentlib.provider_admission import ProviderAdmission, quota_pool_key


def test_acquire_respects_current_concurrency_and_release(tmp_path):
    db = tmp_path / "admission.sqlite3"
    admission = ProviderAdmission(
        "pool", 1, request_timeout=1, db_path=db
    )
    first = admission.acquire()
    acquired = threading.Event()

    def acquire_second():
        second = admission.acquire()
        acquired.set()
        admission.release(second)

    worker = threading.Thread(target=acquire_second)
    worker.start()
    assert not acquired.wait(0.15)
    assert admission.release(first)
    worker.join(2)
    assert not worker.is_alive()
    assert acquired.is_set()




def test_lease_timeout_has_no_grace_and_none_does_not_expire(tmp_path):
    db = tmp_path / "admission.sqlite3"
    timed = ProviderAdmission(
        "timed", 1, request_timeout=1, db_path=db
    )
    lease = timed.acquire()
    assert 0.9 <= lease.expires_at - lease.acquired_at <= 1.1
    assert timed.release(lease)

    unlimited = ProviderAdmission(
        "unlimited", 1, request_timeout=None, db_path=db
    )
    lease = unlimited.acquire()
    assert lease.expires_at is None
    with sqlite3.connect(db) as conn:
        expires_at = conn.execute(
            "SELECT expires_at FROM provider_waiters_v2 "
            "WHERE pool_key='unlimited' AND state='active'"
        ).fetchone()[0]
    assert expires_at is None
    assert unlimited.release(lease)


def test_context_manager_releases_on_error(tmp_path):
    db = tmp_path / "admission.sqlite3"
    admission = ProviderAdmission(
        "pool", 1, request_timeout=1, db_path=db
    )
    try:
        with admission.admitted():
            raise RuntimeError("failed")
    except RuntimeError:
        pass

    lease = admission.acquire()
    assert admission.release(lease)


def test_from_model_config_and_pool_identity(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "AGENTLIB_ADMISSION_DB",
        str(tmp_path / "admission.sqlite3"),
    )
    assert ProviderAdmission.from_model_config(
        "provider/model", {"timeout": 1}
    ) is None

    config = {
        "host": "https://example.test",
        "api_key": "secret",
        "timeout": 1,
        "concurrency": 3,
        "rpm": 120,
    }
    admission = ProviderAdmission.from_model_config(
        "provider/model", config
    )

    same = quota_pool_key("provider/other", config)
    different = quota_pool_key(
        "provider/model", {**config, "api_key": "other"}
    )
    assert same == admission.pool_key
    assert different != same


def test_client_wraps_each_retry_with_admission(monkeypatch):
    from agentlib import client as client_module

    events = []
    calls = 0

    class FakeAdmission:
        @contextlib.contextmanager
        def admitted(self):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

    monkeypatch.setattr(
        client_module,
        "get_model_config",
        lambda _name: {
            "timeout": 1,
            "concurrency": 1,
            "tools": False,
            "api_type": "completions",
        },
    )
    monkeypatch.setattr(
        client_module.ProviderAdmission,
        "from_model_config",
        lambda *_args: FakeAdmission(),
    )
    client = client_module.LLMClient("test/model")
    monkeypatch.setattr(
        client, "_sleep_backoff",
        lambda *_args: events.append("backoff"),
    )

    def call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        events.append(f"call-{calls}")
        if calls == 1:
            raise RuntimeError("temporary")
        return {"content": "ok"}

    monkeypatch.setattr(client, "_call", call)
    assert client.text_call([], retry=1) == {"content": "ok"}
    assert events == [
        "enter", "call-1", "exit", "backoff",
        "enter", "call-2", "exit",
    ]
