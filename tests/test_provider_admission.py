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


def test_configuration_is_not_persisted(tmp_path):
    db = tmp_path / "admission.sqlite3"
    one = ProviderAdmission(
        "pool", 1, request_timeout=1, rate_per_minute=10, db_path=db
    )
    first = one.acquire()

    two = ProviderAdmission(
        "pool", 2, request_timeout=1, rate_per_minute=20, db_path=db
    )
    second = two.acquire()

    with sqlite3.connect(db) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "admission_pools" not in tables
        assert "admission_slots" not in tables
        columns = {
            row[1]
            for table in ("provider_leases", "provider_admissions")
            for row in conn.execute(f"PRAGMA table_info({table})")
        }
    assert "capacity" not in columns
    assert "rate_per_minute" not in columns

    assert one.release(first)
    assert two.release(second)


def test_existing_legacy_configuration_is_ignored(tmp_path):
    db = tmp_path / "admission.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE admission_pools "
            "(pool_key TEXT PRIMARY KEY, capacity INTEGER, "
            "rate_per_minute REAL)"
        )
        conn.execute(
            "INSERT INTO admission_pools VALUES ('pool', 5, 20)"
        )

    admission = ProviderAdmission(
        "pool", 100, request_timeout=1, db_path=db
    )
    lease = admission.acquire()
    assert admission.release(lease)


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
    assert admission.capacity == 3
    assert admission.rate_per_minute == 120

    same = quota_pool_key("provider/other", config)
    different = quota_pool_key(
        "provider/model", {**config, "api_key": "other"}
    )
    assert same == admission.pool_key
    assert different != same
    assert "secret" not in same


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
