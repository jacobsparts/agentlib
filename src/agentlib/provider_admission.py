"""Host-wide FIFO provider admission using runtime configuration."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import select
import sqlite3
import stat
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional, Union

from .process_safety import assert_safe_process_context

_DEFAULT_DB = Path("~/.agentlib/.provider_admission.sqlite3").expanduser()
_DEFAULT_NOTIFY = Path("~/.agentlib/.provider_admission.notify").expanduser()
_IN_MODIFY = 0x00000002
_IN_ATTRIB = 0x00000004
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_IGNORED = 0x00008000
_IN_Q_OVERFLOW = 0x00004000
_IN_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_IN_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_WATCH_MASK = (
    _IN_MODIFY
    | _IN_ATTRIB
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_IGNORED
    | _IN_Q_OVERFLOW
)
_EVENT_HEADER = struct.Struct("iIII")
_LEASE_GRACE_SECONDS = 30.0


class AdmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdmissionLease:
    pool_key: str
    ticket: int
    waiter_id: str
    lease_id: str
    queued_at: float
    acquired_at: float
    expires_at: float

    @property
    def queue_wait_seconds(self) -> float:
        return self.acquired_at - self.queued_at


@dataclass(frozen=True)
class _ClaimResult:
    state: str
    lease: Optional[AdmissionLease]
    next_wall_deadline: Optional[float]
    changed: bool


def admission_paths() -> tuple[Path, Path]:
    return (
        Path(
            os.environ.get(
                "AGENTLIB_ADMISSION_DB",
                str(_DEFAULT_DB),
            )
        ).expanduser(),
        Path(
            os.environ.get(
                "AGENTLIB_ADMISSION_NOTIFY",
                str(_DEFAULT_NOTIFY),
            )
        ).expanduser(),
    )


def admission_path() -> Path:
    return admission_paths()[0]


def quota_pool_key(model_name: str, config: Mapping[str, object]) -> str:
    provider = model_name.split("/", 1)[0]
    credentials = []
    for key in sorted(config):
        lowered = key.lower()
        if lowered in {"api_key", "token", "access_token"} or lowered.endswith(
            ("_api_key", "_token")
        ):
            value = config.get(key)
            if value:
                credentials.append(f"{key}={value}")
    material = "\0".join([str(config.get("host") or ""), *credentials])
    fingerprint = hashlib.sha256(material.encode()).hexdigest()[:16]
    return f"{provider}:{fingerprint}"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _InotifyWatch:
    def __init__(self, path: Path):
        if os.name != "posix" or not hasattr(select, "poll"):
            raise AdmissionError("provider admission requires Linux inotify")
        self.path = path
        self._libc = ctypes.CDLL(None, use_errno=True)
        if not hasattr(self._libc, "inotify_init1"):
            raise AdmissionError("provider admission requires Linux inotify")
        self._libc.inotify_init1.argtypes = [ctypes.c_int]
        self._libc.inotify_init1.restype = ctypes.c_int
        self._libc.inotify_add_watch.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self._libc.inotify_add_watch.restype = ctypes.c_int
        self.fd = self._libc.inotify_init1(_IN_CLOEXEC | _IN_NONBLOCK)
        if self.fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        self._add_watch()

    def _add_watch(self) -> None:
        wd = self._libc.inotify_add_watch(
            self.fd,
            os.fsencode(self.path),
            _WATCH_MASK,
        )
        if wd < 0:
            err = ctypes.get_errno()
            self.close()
            raise OSError(err, os.strerror(err), self.path)
        self.wd = wd

    def drain(self) -> bool:
        invalidated = False
        while True:
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                break
            except InterruptedError:
                continue
            if not data:
                break
            offset = 0
            while offset + _EVENT_HEADER.size <= len(data):
                _wd, mask, _cookie, name_len = _EVENT_HEADER.unpack_from(
                    data,
                    offset,
                )
                offset += _EVENT_HEADER.size + name_len
                if mask & (
                    _IN_DELETE_SELF
                    | _IN_MOVE_SELF
                    | _IN_IGNORED
                    | _IN_Q_OVERFLOW
                ):
                    invalidated = True
        return invalidated

    def wait(self, timeout: Optional[float]) -> None:
        poller = select.poll()
        poller.register(self.fd, select.POLLIN)
        milliseconds = -1 if timeout is None else max(0, int(timeout * 1000))
        while True:
            try:
                poller.poll(milliseconds)
                return
            except InterruptedError:
                continue

    def close(self) -> None:
        fd = getattr(self, "fd", -1)
        if fd >= 0:
            self.fd = -1
            os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class ProviderAdmission:
    def __init__(
        self,
        pool_key: str,
        capacity: int,
        *,
        request_timeout: Optional[float],
        rate_per_minute: Optional[float] = None,
        db_path: Optional[Union[Path, str]] = None,
        notify_path: Optional[Union[Path, str]] = None,
    ):
        assert_safe_process_context()
        if not pool_key:
            raise ValueError("pool_key must not be empty")
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if rate_per_minute is not None and (
            isinstance(rate_per_minute, bool)
            or float(rate_per_minute) <= 0
        ):
            raise ValueError("rate_per_minute must be positive")

        default_db, default_notify = admission_paths()
        self.pool_key = pool_key
        self.capacity = int(capacity)
        self.request_timeout = request_timeout
        self.rate_per_minute = (
            None
            if rate_per_minute is None
            else float(rate_per_minute)
        )
        self.db_path = Path(db_path or default_db).expanduser()
        self.notify_path = Path(notify_path or default_notify).expanduser()
        self._initialize_lock = threading.Lock()
        self._initialized = False
        self._ensure_storage()

    @classmethod
    def from_model_config(cls, model_name: str, config: Mapping[str, object]):
        concurrency = config.get("concurrency")
        if concurrency is None:
            return None
        return cls(
            quota_pool_key(model_name, config),
            int(concurrency),
            request_timeout=config.get("timeout", 300),
            rate_per_minute=config.get("rpm"),
        )

    def _secure_file(self, path: Path) -> None:
        flags = os.O_CREAT | os.O_RDWR | _IN_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise AdmissionError(f"not a regular file: {path}")
            if hasattr(os, "getuid") and info.st_uid != os.getuid():
                raise AdmissionError(f"file is owned by another user: {path}")
            os.fchmod(fd, 0o600)
            if path == self.notify_path and info.st_size == 0:
                os.write(fd, b"\0")
        finally:
            os.close(fd)

    def _ensure_storage(self) -> None:
        if self._initialized:
            return
        with self._initialize_lock:
            if self._initialized:
                return
            for directory in {self.db_path.parent, self.notify_path.parent}:
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._secure_file(self.db_path)
            self._secure_file(self.notify_path)

            last_error = None
            for _ in range(20):
                try:
                    conn = sqlite3.connect(
                        self.db_path,
                        timeout=5,
                        isolation_level=None,
                    )
                    try:
                        conn.execute("PRAGMA busy_timeout=5000")
                        if (
                            conn.execute("PRAGMA journal_mode")
                            .fetchone()[0]
                            .lower()
                            != "wal"
                        ):
                            conn.execute("PRAGMA journal_mode=WAL")
                        conn.execute("PRAGMA synchronous=NORMAL")
                        conn.executescript(_SCHEMA)
                    finally:
                        conn.close()
                    os.chmod(self.db_path, 0o600)
                    self._initialized = True
                    return
                except sqlite3.OperationalError as exc:
                    last_error = exc
                    time.sleep(0.025)
            raise AdmissionError(
                f"could not initialize admission database: {last_error}"
            )

    def _connect(self):
        conn = sqlite3.connect(
            self.db_path,
            timeout=5,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _broadcast(self) -> bool:
        flags = os.O_WRONLY | _IN_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.notify_path, flags)
            try:
                os.pwrite(fd, b"\1", 0)
            finally:
                os.close(fd)
            return True
        except OSError:
            return False

    def _lease_duration(self) -> float:
        timeout = (
            600.0
            if self.request_timeout is None
            else float(self.request_timeout)
        )
        return timeout + _LEASE_GRACE_SECONDS

    def _enqueue(self) -> tuple[int, str, float]:
        queued_at = time.time()
        waiter_id = uuid.uuid4().hex
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR IGNORE INTO provider_admission_state"
                "(pool_key,next_ticket) VALUES (?,1)",
                (self.pool_key,),
            )
            ticket = conn.execute(
                "SELECT next_ticket FROM provider_admission_state "
                "WHERE pool_key=?",
                (self.pool_key,),
            ).fetchone()["next_ticket"]
            conn.execute(
                "UPDATE provider_admission_state SET next_ticket=? "
                "WHERE pool_key=?",
                (ticket + 1, self.pool_key),
            )
            conn.execute(
                "INSERT INTO provider_waiters_v2"
                "(pool_key,ticket,waiter_id,state,queued_at,owner_pid) "
                "VALUES (?,?,?,'waiting',?,?)",
                (
                    self.pool_key,
                    ticket,
                    waiter_id,
                    queued_at,
                    os.getpid(),
                ),
            )
            conn.commit()
            return ticket, waiter_id, queued_at
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _cleanup_runtime(self, conn, now: float) -> bool:
        changed = bool(
            conn.execute(
                "UPDATE provider_waiters_v2 "
                "SET state='expired',finished_at=? "
                "WHERE pool_key=? AND state='active' AND expires_at<=?",
                (now, self.pool_key, now),
            ).rowcount
        )

        waiting = conn.execute(
            "SELECT ticket,owner_pid FROM provider_waiters_v2 "
            "WHERE pool_key=? AND state='waiting'",
            (self.pool_key,),
        ).fetchall()
        dead_tickets = [
            row["ticket"]
            for row in waiting
            if not _pid_is_alive(row["owner_pid"])
        ]
        if dead_tickets:
            conn.executemany(
                "UPDATE provider_waiters_v2 "
                "SET state='expired',finished_at=? "
                "WHERE pool_key=? AND ticket=? AND state='waiting'",
                (
                    (now, self.pool_key, ticket)
                    for ticket in dead_tickets
                ),
            )
            changed = True

        conn.execute(
            "DELETE FROM provider_admissions WHERE pool_key=? "
            "AND admitted_at<=?",
            (self.pool_key, now - 60.0),
        )
        return changed

    def _try_claim(
        self,
        ticket: int,
        waiter_id: str,
        queued_at: float,
    ) -> _ClaimResult:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            changed = self._cleanup_runtime(conn, now)

            own = conn.execute(
                "SELECT state FROM provider_waiters_v2 "
                "WHERE pool_key=? AND ticket=? AND waiter_id=?",
                (self.pool_key, ticket, waiter_id),
            ).fetchone()
            state = own["state"] if own else "missing"

            head = conn.execute(
                "SELECT ticket,waiter_id FROM provider_waiters_v2 "
                "WHERE pool_key=? AND state='waiting' "
                "ORDER BY ticket LIMIT 1",
                (self.pool_key,),
            ).fetchone()
            active_count = conn.execute(
                "SELECT COUNT(*) FROM provider_waiters_v2 "
                "WHERE pool_key=? AND state='active' AND expires_at>?",
                (self.pool_key, now),
            ).fetchone()[0]

            rate_ready = True
            rate_deadline = None
            if self.rate_per_minute is not None:
                recent = conn.execute(
                    "SELECT admitted_at FROM provider_admissions "
                    "WHERE pool_key=? ORDER BY admitted_at",
                    (self.pool_key,),
                ).fetchall()
                rate_ready = len(recent) < self.rate_per_minute
                if not rate_ready:
                    rate_deadline = recent[0]["admitted_at"] + 60.0

            lease = None
            if (
                state == "waiting"
                and head is not None
                and head["ticket"] == ticket
                and head["waiter_id"] == waiter_id
                and active_count < self.capacity
                and rate_ready
            ):
                lease_id = uuid.uuid4().hex
                expires_at = now + self._lease_duration()
                updated = conn.execute(
                    "UPDATE provider_waiters_v2 "
                    "SET state='active',lease_id=?,acquired_at=?,expires_at=? "
                    "WHERE pool_key=? AND ticket=? AND waiter_id=? "
                    "AND state='waiting'",
                    (
                        lease_id,
                        now,
                        expires_at,
                        self.pool_key,
                        ticket,
                        waiter_id,
                    ),
                ).rowcount
                if updated != 1:
                    raise AdmissionError("waiting request changed unexpectedly")
                if self.rate_per_minute is not None:
                    conn.execute(
                        "INSERT INTO provider_admissions"
                        "(pool_key,admitted_at) VALUES (?,?)",
                        (self.pool_key, now),
                    )
                lease = AdmissionLease(
                    self.pool_key,
                    ticket,
                    waiter_id,
                    lease_id,
                    queued_at,
                    now,
                    expires_at,
                )
                state = "active"
                changed = True

            lease_deadline = conn.execute(
                "SELECT MIN(expires_at) FROM provider_waiters_v2 "
                "WHERE pool_key=? AND state='active'",
                (self.pool_key,),
            ).fetchone()[0]
            deadlines = [
                value
                for value in (lease_deadline, rate_deadline)
                if value is not None
            ]
            conn.commit()
            return _ClaimResult(
                state,
                lease,
                min(deadlines) if deadlines else None,
                changed,
            )
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _cancel(self, ticket: int, waiter_id: str) -> bool:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            changed = bool(
                conn.execute(
                    "UPDATE provider_waiters_v2 "
                    "SET state='cancelled',finished_at=? "
                    "WHERE pool_key=? AND ticket=? AND waiter_id=? "
                    "AND state='waiting'",
                    (
                        time.time(),
                        self.pool_key,
                        ticket,
                        waiter_id,
                    ),
                ).rowcount
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        if changed:
            self._broadcast()
        return changed

    def acquire(self) -> AdmissionLease:
        watch = _InotifyWatch(self.notify_path)
        ticket = -1
        waiter_id = ""
        queued_at = 0.0
        try:
            ticket, waiter_id, queued_at = self._enqueue()
            self._broadcast()
            while True:
                if watch.drain():
                    watch.close()
                    self._secure_file(self.notify_path)
                    watch = _InotifyWatch(self.notify_path)

                result = self._try_claim(
                    ticket,
                    waiter_id,
                    queued_at,
                )
                if result.changed:
                    self._broadcast()
                if result.lease is not None:
                    return result.lease
                if result.state != "waiting":
                    raise AdmissionError(
                        f"unexpected waiter state {result.state!r}"
                    )

                timeout = None
                if result.next_wall_deadline is not None:
                    timeout = max(
                        0.0,
                        result.next_wall_deadline - time.time(),
                    )
                watch.wait(timeout)
        except BaseException:
            if ticket >= 0 and waiter_id:
                self._cancel(ticket, waiter_id)
            raise
        finally:
            watch.close()

    def release(self, lease: AdmissionLease) -> bool:
        if lease.pool_key != self.pool_key:
            raise ValueError("lease belongs to another pool")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                "UPDATE provider_waiters_v2 "
                "SET state='finished',lease_id=NULL,finished_at=? "
                "WHERE pool_key=? AND ticket=? AND waiter_id=? "
                "AND lease_id=? AND state='active'",
                (
                    time.time(),
                    self.pool_key,
                    lease.ticket,
                    lease.waiter_id,
                    lease.lease_id,
                ),
            ).rowcount
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        if updated:
            self._broadcast()
        return bool(updated)

    @contextlib.contextmanager
    def admitted(self) -> Iterator[AdmissionLease]:
        lease = self.acquire()
        try:
            yield lease
        finally:
            self.release(lease)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_admission_state (
    pool_key TEXT PRIMARY KEY,
    next_ticket INTEGER NOT NULL CHECK(next_ticket>0)
);

CREATE TABLE IF NOT EXISTS provider_waiters_v2 (
    pool_key TEXT NOT NULL,
    ticket INTEGER NOT NULL,
    waiter_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK(
        state IN ('waiting','active','finished','expired','cancelled')
    ),
    queued_at REAL NOT NULL,
    owner_pid INTEGER NOT NULL,
    lease_id TEXT,
    acquired_at REAL,
    expires_at REAL,
    finished_at REAL,
    PRIMARY KEY(pool_key,ticket)
);
CREATE INDEX IF NOT EXISTS provider_waiters_v2_order
ON provider_waiters_v2(pool_key,state,ticket);
CREATE INDEX IF NOT EXISTS provider_waiters_v2_expiration
ON provider_waiters_v2(pool_key,state,expires_at);

CREATE TABLE IF NOT EXISTS provider_admissions (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pool_key TEXT NOT NULL,
    admitted_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS provider_admissions_pool_time
ON provider_admissions(pool_key,admitted_at);
"""
