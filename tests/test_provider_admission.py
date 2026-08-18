import contextlib
import multiprocessing
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from agentlib import provider_admission as admission_module
from agentlib.provider_admission import (
    AdmissionConfigurationError,
    ProviderAdmission,
    quota_pool_key,
)


def paths(root):
    root = Path(root)
    return root / "admission.sqlite3", root / "admission.notify"


def worker_acquire(root, hold, output):
    db, notify = paths(root)
    admission = ProviderAdmission(
        "pool", 1, request_timeout=2,
        db_path=db, notify_path=notify,
    )
    lease = admission.acquire()
    output.put((lease.ticket, lease.slot_no))
    time.sleep(hold)
    admission.release(lease)


def worker_die_active(root, output):
    admission_module._CLAIM_WINDOW_SECONDS = 0.05
    admission_module._LEASE_GRACE_SECONDS = 0.0
    db, notify = paths(root)
    admission = ProviderAdmission(
        "pool", 1, request_timeout=0.15, db_path=db, notify_path=notify,
    )
    output.put(admission.acquire().ticket)
    output.close()
    output.join_thread()
    os._exit(0)


def wait_for_waiters(db, count, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with sqlite3.connect(db) as conn:
            found = conn.execute(
                "SELECT COUNT(*) FROM admission_waiters WHERE state='waiting'"
            ).fetchone()[0]
        if found >= count:
            return
        time.sleep(0.01)
    raise AssertionError("waiters did not arrive")


def test_immediate_slots_and_conditional_release(tmp_path):
    db, notify = paths(tmp_path)
    admission = ProviderAdmission(
        "pool", 2, request_timeout=1, db_path=db, notify_path=notify
    )
    first = admission.acquire()
    second = admission.acquire()
    assert (first.ticket, first.slot_no) == (1, 0)
    assert (second.ticket, second.slot_no) == (2, 1)
    assert admission.release(first)
    assert not admission.release(first)
    assert admission.release(second)


def test_exception_releases_and_pools_are_independent(tmp_path):
    db, notify = paths(tmp_path)
    first = ProviderAdmission(
        "first", 1, request_timeout=1, db_path=db, notify_path=notify
    )
    second = ProviderAdmission(
        "second", 1, request_timeout=1, db_path=db, notify_path=notify
    )
    with pytest.raises(RuntimeError):
        with first.admitted(), second.admitted():
            raise RuntimeError("failed")
    assert first.release(first.acquire())
    assert second.release(second.acquire())


def test_capacity_and_rate_updates_adapt_existing_pool(tmp_path):
    db, notify = paths(tmp_path)
    ProviderAdmission("pool", 1, request_timeout=1, rate_per_minute=60, db_path=db, notify_path=notify)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT capacity, rate_per_minute FROM admission_pools WHERE pool_key='pool'").fetchone() == (1, 60.0)
        assert conn.execute("SELECT COUNT(*) FROM admission_slots WHERE pool_key='pool'").fetchone()[0] == 1

    # Increase capacity and change rate
    expanded = ProviderAdmission("pool", 3, request_timeout=1, rate_per_minute=120, db_path=db, notify_path=notify)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT capacity, rate_per_minute FROM admission_pools WHERE pool_key='pool'").fetchone() == (3, 120.0)
        assert conn.execute("SELECT COUNT(*) FROM admission_slots WHERE pool_key='pool'").fetchone()[0] == 3

    # Shrink capacity
    shrunk = ProviderAdmission("pool", 2, request_timeout=1, rate_per_minute=120, db_path=db, notify_path=notify)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT capacity, rate_per_minute FROM admission_pools WHERE pool_key='pool'").fetchone() == (2, 120.0)
        assert conn.execute("SELECT COUNT(*) FROM admission_slots WHERE pool_key='pool'").fetchone()[0] == 2


def test_initialization_prunes_terminal_waiters_only(tmp_path):
    db, notify = paths(tmp_path)
    admission = ProviderAdmission(
        "pool", 2, request_timeout=2, db_path=db, notify_path=notify
    )
    finished = admission.acquire()
    active = admission.acquire()
    admission.release(finished)

    with sqlite3.connect(db) as conn:
        next_ticket = conn.execute(
            "SELECT next_ticket FROM admission_pools WHERE pool_key='pool'"
        ).fetchone()[0]
        conn.executemany(
            "INSERT INTO admission_waiters"
            "(pool_key,ticket,waiter_id,state,queued_at,finished_at) "
            "VALUES ('pool',?,?,?,?,?)",
            [
                (next_ticket, "expired-waiter", "expired", time.time(), time.time()),
                (
                    next_ticket + 1,
                    "cancelled-waiter",
                    "cancelled",
                    time.time(),
                    time.time(),
                ),
            ],
        )
        conn.execute(
            "UPDATE admission_pools SET next_ticket=? WHERE pool_key='pool'",
            (next_ticket + 2,),
        )

    ProviderAdmission(
        "other", 1, request_timeout=2, db_path=db, notify_path=notify
    )

    with sqlite3.connect(db) as conn:
        states = conn.execute(
            "SELECT state FROM admission_waiters ORDER BY ticket"
        ).fetchall()
    assert states == [("active",)]
    admission.release(active)


def test_pruned_expired_waiter_requeues_instead_of_failing(tmp_path, monkeypatch):
    monkeypatch.setattr(admission_module, "_CLAIM_WINDOW_SECONDS", 0.05)
    db, notify = paths(tmp_path)
    first = ProviderAdmission(
        "pool", 1, request_timeout=2,
        db_path=db, notify_path=notify,
    )
    other = ProviderAdmission(
        "pool", 1, request_timeout=2,
        db_path=db, notify_path=notify,
    )

    queued_at = time.time()
    ticket, waiter_id = first._enqueue()
    other_ticket, other_waiter_id = other._enqueue()
    assert other._try_claim(other_ticket, other_waiter_id, time.time()).state == "waiting"
    time.sleep(0.06)
    other_lease = other._try_claim(
        other_ticket, other_waiter_id, time.time()
    ).lease
    assert other_lease is not None

    ProviderAdmission(
        "new-client", 1, request_timeout=2,
        db_path=db, notify_path=notify,
    )
    original_enqueue = first._enqueue
    queued = [(ticket, waiter_id)]

    def enqueue():
        return queued.pop(0) if queued else original_enqueue()

    monkeypatch.setattr(first, "_enqueue", enqueue)
    other.release(other_lease)

    lease = first.acquire()
    assert lease.ticket > other_ticket
    assert first.release(lease)


@pytest.mark.skipif(os.name != "posix", reason="Linux inotify required")
def test_cross_process_fifo(tmp_path, monkeypatch):
    monkeypatch.setattr(admission_module, "_CLAIM_WINDOW_SECONDS", 1.0)
    ctx = multiprocessing.get_context("fork")
    db, notify = paths(tmp_path)
    owner = ProviderAdmission(
        "pool", 1, request_timeout=2,
        db_path=db, notify_path=notify,
    )
    owner_lease = owner.acquire()
    output = ctx.Queue()
    workers = []
    for expected in range(1, 4):
        proc = ctx.Process(target=worker_acquire, args=(str(tmp_path), 0.05, output))
        proc.start()
        workers.append(proc)
        wait_for_waiters(db, expected)
    owner.release(owner_lease)
    results = [output.get(timeout=5) for _ in workers]
    for proc in workers:
        proc.join(5)
        assert proc.exitcode == 0
    assert [ticket for ticket, _slot in results] == [2, 3, 4]


@pytest.mark.skipif(os.name != "posix", reason="Linux inotify required")
def test_dead_waiting_head_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(admission_module, "_CLAIM_WINDOW_SECONDS", 0.2)
    ctx = multiprocessing.get_context("fork")
    db, notify = paths(tmp_path)
    owner = ProviderAdmission(
        "pool", 1, request_timeout=2,
        db_path=db, notify_path=notify,
    )
    owner_lease = owner.acquire()
    output = ctx.Queue()
    dead = ctx.Process(target=worker_acquire, args=(str(tmp_path), 1, output))
    dead.start()
    wait_for_waiters(db, 1)
    dead.terminate()
    dead.join(2)
    live = ctx.Process(target=worker_acquire, args=(str(tmp_path), 0, output))
    live.start()
    wait_for_waiters(db, 2)
    owner.release(owner_lease)
    ticket, slot = output.get(timeout=5)
    live.join(5)
    assert live.exitcode == 0
    assert (ticket, slot) == (3, 0)
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT state FROM admission_waiters WHERE ticket=2"
        ).fetchone()[0] == "expired"


@pytest.mark.skipif(os.name != "posix", reason="Linux inotify required")
def test_dead_active_owner_reclaimed_at_expiration(tmp_path, monkeypatch):
    monkeypatch.setattr(admission_module, "_CLAIM_WINDOW_SECONDS", 0.05)
    monkeypatch.setattr(admission_module, "_LEASE_GRACE_SECONDS", 0.0)
    ctx = multiprocessing.get_context("fork")
    db, notify = paths(tmp_path)
    output = ctx.Queue()
    dead = ctx.Process(target=worker_die_active, args=(str(tmp_path), output))
    dead.start()
    assert output.get(timeout=3) == 1
    dead.join(3)
    admission = ProviderAdmission(
        "pool", 1, request_timeout=0.15, db_path=db, notify_path=notify,
    )
    started = time.monotonic()
    lease = admission.acquire()
    assert time.monotonic() - started >= 0.08
    assert lease.ticket == 2
    admission.release(lease)


def test_rate_limit_waits_without_holding_a_slot(tmp_path):
    db, notify = paths(tmp_path)
    admission = ProviderAdmission(
        "pool", 1, request_timeout=1, rate_per_minute=600, db_path=db, notify_path=notify,
    )
    first = admission.acquire()
    admission.release(first)

    started = time.monotonic()
    second = admission.acquire()
    elapsed = time.monotonic() - started
    assert elapsed >= 0.08
    assert second.slot_no == 0
    admission.release(second)


def test_rate_and_slot_are_claimed_atomically_across_processes(tmp_path):
    ctx = multiprocessing.get_context("fork")
    db, notify = paths(tmp_path)
    output = ctx.Queue()

    def rate_worker(root, result):
        worker_db, worker_notify = paths(root)
        admission = ProviderAdmission(
            "pool", 1, request_timeout=1, rate_per_minute=600, db_path=worker_db, notify_path=worker_notify,
        )
        lease = admission.acquire()
        result.put((lease.ticket, time.time()))
        admission.release(lease)

    workers = [
        ctx.Process(target=rate_worker, args=(str(tmp_path), output))
        for _ in range(3)
    ]
    for worker in workers:
        worker.start()
    results = [output.get(timeout=5) for _ in workers]
    for worker in workers:
        worker.join(5)
        assert worker.exitcode == 0
    results.sort()
    assert [ticket for ticket, _when in results] == [1, 2, 3]
    times = [when for _ticket, when in results]
    assert times[1] - times[0] >= 0.07
    assert times[2] - times[1] >= 0.07

    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM admission_slots WHERE lease_id IS NOT NULL"
        ).fetchone()[0] == 0


def test_from_model_config_is_enabled_only_by_concurrency(monkeypatch, tmp_path):
    db, notify = paths(tmp_path)
    monkeypatch.setenv("AGENTLIB_ADMISSION_DB", str(db))
    monkeypatch.setenv("AGENTLIB_ADMISSION_NOTIFY", str(notify))

    assert ProviderAdmission.from_model_config(
        "camelai/model", {"timeout": 1}
    ) is None
    assert ProviderAdmission.from_model_config(
        "camelai/model", {"timeout": 1, "concurrency": None}
    ) is None

    admission = ProviderAdmission.from_model_config(
        "camelai/model", {
            "timeout": 1, "concurrency": 3, "rpm": 120,
        }
    )
    assert admission is not None
    assert admission.capacity == 3
    assert admission.rate_per_minute == 120


def test_pool_key_hides_secret_and_shares_models():
    one = quota_pool_key("camelai/a", {"host": "x", "api_key": "secret-a"})
    two = quota_pool_key("camelai/b", {"host": "x", "api_key": "secret-a"})
    other = quota_pool_key("camelai/a", {"host": "x", "api_key": "secret-b"})
    assert one == two
    assert one != other
    assert "secret" not in one


def test_client_wraps_call_and_retry_reacquires(monkeypatch):
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
        client_module, "get_model_config",
        lambda _name: {
            "timeout": 1, "concurrency": 1, "tools": False,
            "api_type": "completions",
        },
    )
    monkeypatch.setattr(
        client_module.ProviderAdmission, "from_model_config",
        lambda *_args: FakeAdmission(),
    )
    client = client_module.LLMClient("test/model")
    monkeypatch.setattr(client, "_sleep_backoff", lambda *_args: events.append("backoff"))

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


def test_http_call_starts_only_after_admission(tmp_path, monkeypatch):
    from agentlib import client as client_module

    db, notify = paths(tmp_path)
    admission = ProviderAdmission(
        "pool", 1, request_timeout=2,
        db_path=db, notify_path=notify,
    )
    owner = admission.acquire()
    monkeypatch.setattr(
        client_module, "get_model_config",
        lambda _name: {
            "timeout": 2, "concurrency": 1, "tools": False,
            "api_type": "completions",
        },
    )
    monkeypatch.setattr(
        client_module.ProviderAdmission, "from_model_config",
        lambda *_args: admission,
    )
    client = client_module.LLMClient("test/model")
    started = threading.Event()
    result = []
    monkeypatch.setattr(
        client, "_call",
        lambda *_args, **_kwargs: started.set() or {"content": "ok"},
    )
    worker = threading.Thread(target=lambda: result.append(client.text_call([], retry=0)))
    worker.start()
    wait_for_waiters(db, 1)
    assert not started.wait(0.15)
    admission.release(owner)
    worker.join(3)
    assert not worker.is_alive()
    assert started.is_set()
    assert result == [{"content": "ok"}]
