"""Host-wide FIFO admission control for provider requests."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import math
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
    _IN_MODIFY | _IN_ATTRIB | _IN_DELETE_SELF | _IN_MOVE_SELF
    | _IN_IGNORED | _IN_Q_OVERFLOW
)
_EVENT_HEADER = struct.Struct("iIII")


class AdmissionError(RuntimeError):
    pass


class AdmissionConfigurationError(AdmissionError):
    pass


class AdmissionQueueTimeout(AdmissionError, TimeoutError):
    pass


@dataclass(frozen=True)
class AdmissionLease:
    pool_key: str
    ticket: int
    waiter_id: str
    slot_no: int
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
        Path(os.environ.get("AGENTLIB_ADMISSION_DB", str(_DEFAULT_DB))).expanduser(),
        Path(
            os.environ.get("AGENTLIB_ADMISSION_NOTIFY", str(_DEFAULT_NOTIFY))
        ).expanduser(),
    )


def quota_pool_key(model_name: str, config: Mapping[str, object]) -> str:
    explicit = config.get("admission_pool")
    if explicit:
        return str(explicit)
    provider = model_name.split("/", 1)[0]
    credential_parts = []
    for key in sorted(config):
        lowered = key.lower()
        if lowered in {"api_key", "token", "access_token"} or lowered.endswith(
            ("_api_key", "_token")
        ):
            value = config.get(key)
            if value:
                credential_parts.append(f"{key}={value}")
    material = "\0".join([str(config.get("host") or ""), *credential_parts])
    fingerprint = hashlib.sha256(material.encode()).hexdigest()[:16]
    return f"{provider}:{fingerprint}"


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
            ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32
        ]
        self._libc.inotify_add_watch.restype = ctypes.c_int
        self.fd = self._libc.inotify_init1(_IN_CLOEXEC | _IN_NONBLOCK)
        if self.fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        self._add_watch()

    def _add_watch(self) -> None:
        wd = self._libc.inotify_add_watch(
            self.fd, os.fsencode(self.path), _WATCH_MASK
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
                _wd, mask, _cookie, name_len = _EVENT_HEADER.unpack_from(data, offset)
                offset += _EVENT_HEADER.size + name_len
                if mask & (_IN_DELETE_SELF | _IN_MOVE_SELF | _IN_IGNORED):
                    invalidated = True
        return invalidated

    def wait(self, timeout: Optional[float]) -> None:
        timeout_ms = -1 if timeout is None else max(0, math.ceil(timeout * 1000))
        poller = select.poll()
        poller.register(self.fd, select.POLLIN | select.POLLERR | select.POLLHUP)
        while True:
            try:
                poller.poll(timeout_ms)
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
        burst_capacity: Optional[float] = None,
        claim_window: float = 0.1,
        lease_grace: float = 30.0,
        queue_timeout: Optional[float] = None,
        db_path: Optional[Union[Path, str]] = None,
        notify_path: Optional[Union[Path, str]] = None,
    ):
        if not pool_key:
            raise ValueError("pool_key must not be empty")
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if rate_per_minute is not None and (
            isinstance(rate_per_minute, bool) or float(rate_per_minute) <= 0
        ):
            raise ValueError("rate_per_minute must be positive")
        if burst_capacity is None:
            burst_capacity = capacity
        if (
            isinstance(burst_capacity, bool)
            or float(burst_capacity) < 1
        ):
            raise ValueError("burst_capacity must be at least one")
        if rate_per_minute is None and burst_capacity != capacity:
            raise ValueError("burst_capacity requires rate_per_minute")
        if claim_window <= 0 or lease_grace < 0:
            raise ValueError("invalid claim window or lease grace")
        if queue_timeout is not None and queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive")
        default_db, default_notify = admission_paths()
        self.pool_key = pool_key
        self.capacity = int(capacity)
        self.request_timeout = request_timeout
        self.rate_per_minute = (
            None if rate_per_minute is None else float(rate_per_minute)
        )
        self.burst_capacity = float(burst_capacity)
        self.claim_window = float(claim_window)
        self.lease_grace = float(lease_grace)
        self.queue_timeout = queue_timeout
        self.db_path = Path(db_path or default_db).expanduser()
        self.notify_path = Path(notify_path or default_notify).expanduser()
        self._initialize_lock = threading.Lock()
        self._initialized = False
        self._ensure_storage()
        self._register_pool()

    @classmethod
    def from_model_config(cls, model_name: str, config: Mapping[str, object]):
        if "concurrency" not in config or config["concurrency"] is None:
            return None
        return cls(
            quota_pool_key(model_name, config),
            int(config["concurrency"]),
            request_timeout=config.get("timeout", 300),
            rate_per_minute=config.get("rpm"),
            burst_capacity=config.get("admission_burst", config["concurrency"]),
            claim_window=float(config.get("admission_claim_window", 0.1)),
            lease_grace=float(config.get("admission_lease_grace", 30.0)),
            queue_timeout=config.get("admission_queue_timeout"),
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
                    conn = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
                    try:
                        conn.execute("PRAGMA busy_timeout=5000")
                        if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                            conn.execute("PRAGMA journal_mode=WAL")
                        conn.execute("PRAGMA synchronous=NORMAL")
                        conn.executescript(_SCHEMA)
                        columns = {
                            row[1] for row in conn.execute(
                                "PRAGMA table_info(admission_waiters)"
                            )
                        }
                        if "acquired_at" not in columns:
                            conn.execute(
                                "ALTER TABLE admission_waiters "
                                "ADD COLUMN acquired_at REAL"
                            )
                        pool_columns = {
                            row[1] for row in conn.execute(
                                "PRAGMA table_info(admission_pools)"
                            )
                        }
                        for name, declaration in (
                            ("rate_per_minute", "REAL"),
                            ("burst_capacity", "REAL"),
                            ("available_tokens", "REAL"),
                            ("tokens_updated_at", "REAL"),
                        ):
                            if name not in pool_columns:
                                conn.execute(
                                    f"ALTER TABLE admission_pools "
                                    f"ADD COLUMN {name} {declaration}"
                                )
                    finally:
                        conn.close()
                    os.chmod(self.db_path, 0o600)
                    self._initialized = True
                    return
                except sqlite3.OperationalError as exc:
                    last_error = exc
                    time.sleep(0.025)
            raise AdmissionError(f"could not initialize admission database: {last_error}")

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _register_pool(self) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_pool(conn)
            conn.execute(
                "DELETE FROM admission_waiters "
                "WHERE state IN ('finished','expired','cancelled')"
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

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
        base = 600.0 if self.request_timeout is None else float(self.request_timeout)
        return max(base + self.lease_grace, self.claim_window * 2)

    def _ensure_pool(self, conn) -> None:
        now = time.time()
        conn.execute(
            "INSERT OR IGNORE INTO admission_pools"
            "(pool_key,capacity,next_ticket,rate_per_minute,burst_capacity,"
            "available_tokens,tokens_updated_at) VALUES (?,?,1,?,?,?,?)",
            (
                self.pool_key,
                self.capacity,
                self.rate_per_minute,
                self.burst_capacity if self.rate_per_minute is not None else None,
                self.burst_capacity if self.rate_per_minute is not None else None,
                now if self.rate_per_minute is not None else None,
            ),
        )
        actual = conn.execute(
            "SELECT capacity,rate_per_minute,burst_capacity "
            "FROM admission_pools WHERE pool_key=?",
            (self.pool_key,),
        ).fetchone()
        if actual["capacity"] != self.capacity:
            raise AdmissionConfigurationError(
                f"pool {self.pool_key!r} has capacity {actual['capacity']}, "
                f"not {self.capacity}"
            )
        if actual["rate_per_minute"] is None and self.rate_per_minute is not None:
            conn.execute(
                "UPDATE admission_pools SET rate_per_minute=?,burst_capacity=?,"
                "available_tokens=?,tokens_updated_at=? WHERE pool_key=? "
                "AND rate_per_minute IS NULL",
                (
                    self.rate_per_minute, self.burst_capacity,
                    self.burst_capacity, now, self.pool_key,
                ),
            )
            actual = conn.execute(
                "SELECT capacity,rate_per_minute,burst_capacity "
                "FROM admission_pools WHERE pool_key=?",
                (self.pool_key,),
            ).fetchone()
        actual_rate = actual["rate_per_minute"]
        actual_burst = actual["burst_capacity"]
        rate_matches = (
            actual_rate is None and self.rate_per_minute is None
        ) or (
            actual_rate is not None and self.rate_per_minute is not None
            and math.isclose(actual_rate, self.rate_per_minute)
        )
        burst_matches = (
            actual_burst is None and self.rate_per_minute is None
        ) or (
            actual_burst is not None and self.rate_per_minute is not None
            and math.isclose(actual_burst, self.burst_capacity)
        )
        if not rate_matches or not burst_matches:
            raise AdmissionConfigurationError(
                f"pool {self.pool_key!r} has rate/burst "
                f"{actual_rate}/{actual_burst}, not "
                f"{self.rate_per_minute}/{self.burst_capacity}"
            )
        conn.executemany(
            "INSERT OR IGNORE INTO admission_slots(pool_key, slot_no) VALUES (?, ?)",
            ((self.pool_key, slot) for slot in range(self.capacity)),
        )

    def _enqueue(self):
        now = time.time()
        waiter_id = uuid.uuid4().hex
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_pool(conn)
            ticket = conn.execute(
                "SELECT next_ticket FROM admission_pools WHERE pool_key=?",
                (self.pool_key,),
            ).fetchone()["next_ticket"]
            conn.execute(
                "UPDATE admission_pools SET next_ticket=? WHERE pool_key=?",
                (ticket + 1, self.pool_key),
            )
            conn.execute(
                "INSERT INTO admission_waiters"
                "(pool_key,ticket,waiter_id,state,queued_at) "
                "VALUES (?,?,?,'waiting',?)",
                (self.pool_key, ticket, waiter_id, now),
            )
            conn.commit()
            return ticket, waiter_id
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _token_status(self, conn, now):
        if self.rate_per_minute is None:
            return True, None, None
        row = conn.execute(
            "SELECT available_tokens,tokens_updated_at FROM admission_pools "
            "WHERE pool_key=?",
            (self.pool_key,),
        ).fetchone()
        available = min(
            self.burst_capacity,
            float(row["available_tokens"])
            + max(0.0, now - float(row["tokens_updated_at"]))
            * self.rate_per_minute / 60.0,
        )
        if available >= 1.0:
            return True, available, None
        deadline = now + (1.0 - available) * 60.0 / self.rate_per_minute
        return False, available, deadline

    def _consume_token(self, conn, now) -> bool:
        available, tokens, _deadline = self._token_status(conn, now)
        if not available:
            return False
        if self.rate_per_minute is not None:
            conn.execute(
                "UPDATE admission_pools SET available_tokens=?,"
                "tokens_updated_at=? WHERE pool_key=?",
                (tokens - 1.0, now, self.pool_key),
            )
        return True

    def _reap_and_arm(self, conn, now):
        changed = False
        expired = conn.execute(
            "SELECT slot_no,waiter_id,lease_id FROM admission_slots "
            "WHERE pool_key=? AND lease_id IS NOT NULL AND expires_at<=?",
            (self.pool_key, now),
        ).fetchall()
        for slot in expired:
            conn.execute(
                "UPDATE admission_waiters SET state='expired',slot_no=NULL,"
                "lease_id=NULL,finished_at=? WHERE waiter_id=? AND lease_id=? "
                "AND state='active'",
                (now, slot["waiter_id"], slot["lease_id"]),
            )
            conn.execute(
                "UPDATE admission_slots SET lease_id=NULL,waiter_id=NULL,"
                "ticket=NULL,owner_pid=NULL,acquired_at=NULL,expires_at=NULL "
                "WHERE pool_key=? AND slot_no=? AND lease_id=?",
                (self.pool_key, slot["slot_no"], slot["lease_id"]),
            )
            changed = True
        free = conn.execute(
            "SELECT slot_no FROM admission_slots WHERE pool_key=? "
            "AND lease_id IS NULL ORDER BY slot_no LIMIT 1",
            (self.pool_key,),
        ).fetchone()
        token_ready, _tokens, _token_deadline = self._token_status(conn, now)
        if not token_ready:
            return free, None, changed
        while free is not None:
            head = conn.execute(
                "SELECT ticket,waiter_id,head_claim_expires_at "
                "FROM admission_waiters WHERE pool_key=? AND state='waiting' "
                "ORDER BY ticket LIMIT 1",
                (self.pool_key,),
            ).fetchone()
            if head is None:
                return free, None, changed
            deadline = head["head_claim_expires_at"]
            if deadline is not None and deadline <= now:
                conn.execute(
                    "UPDATE admission_waiters SET state='expired',finished_at=? "
                    "WHERE pool_key=? AND ticket=? AND state='waiting'",
                    (now, self.pool_key, head["ticket"]),
                )
                changed = True
                continue
            if deadline is None:
                deadline = now + self.claim_window
                conn.execute(
                    "UPDATE admission_waiters SET head_claim_expires_at=? "
                    "WHERE pool_key=? AND ticket=? AND state='waiting'",
                    (deadline, self.pool_key, head["ticket"]),
                )
                changed = True
                head = dict(head)
                head["head_claim_expires_at"] = deadline
            return free, head, changed
        return None, None, changed

    def _try_claim(self, ticket, waiter_id, queued_at):
        conn = self._connect()
        lease = None
        changed = False
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            self._ensure_pool(conn)
            free, head, changed = self._reap_and_arm(conn, now)
            own = conn.execute(
                "SELECT state FROM admission_waiters WHERE pool_key=? "
                "AND ticket=? AND waiter_id=?",
                (self.pool_key, ticket, waiter_id),
            ).fetchone()
            state = own["state"] if own else "cancelled"
            token_ready, _tokens, _token_deadline = self._token_status(conn, now)
            if (
                state == "waiting" and free is not None and head is not None
                and token_ready
                and head["ticket"] == ticket and head["waiter_id"] == waiter_id
                and head["head_claim_expires_at"] > now
            ):
                if not self._consume_token(conn, now):
                    raise AdmissionError("rate token changed unexpectedly")
                lease_id = uuid.uuid4().hex
                acquired = now
                expires = acquired + self._lease_duration()
                slot_no = free["slot_no"]
                if conn.execute(
                    "UPDATE admission_slots SET lease_id=?,waiter_id=?,ticket=?,"
                    "owner_pid=?,acquired_at=?,expires_at=? WHERE pool_key=? "
                    "AND slot_no=? AND lease_id IS NULL",
                    (lease_id, waiter_id, ticket, os.getpid(), acquired, expires,
                     self.pool_key, slot_no),
                ).rowcount != 1:
                    raise AdmissionError("free slot changed unexpectedly")
                conn.execute(
                    "UPDATE admission_waiters SET state='active',slot_no=?,"
                    "lease_id=?,acquired_at=?,head_claim_expires_at=NULL "
                    "WHERE pool_key=? AND ticket=? AND waiter_id=? "
                    "AND state='waiting'",
                    (slot_no, lease_id, acquired, self.pool_key, ticket, waiter_id),
                )
                lease = AdmissionLease(
                    self.pool_key, ticket, waiter_id, slot_no, lease_id,
                    queued_at, acquired, expires,
                )
                state = "active"
                changed = True
                self._reap_and_arm(conn, acquired)
            deadlines = conn.execute(
                "SELECT "
                "(SELECT MIN(head_claim_expires_at) FROM admission_waiters "
                " WHERE pool_key=? AND state='waiting') AS head_deadline,"
                "(SELECT MIN(expires_at) FROM admission_slots "
                " WHERE pool_key=? AND lease_id IS NOT NULL) AS lease_deadline",
                (self.pool_key, self.pool_key),
            ).fetchone()
            _ready, _tokens, token_deadline = self._token_status(conn, now)
            values = [
                value for value in (
                    deadlines["head_deadline"],
                    deadlines["lease_deadline"],
                    token_deadline,
                )
                if value is not None
            ]
            next_deadline = min(values) if values else None
            conn.commit()
            return _ClaimResult(state, lease, next_deadline, changed)
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _cancel(self, ticket, waiter_id):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            changed = conn.execute(
                "UPDATE admission_waiters SET state='cancelled',finished_at=? "
                "WHERE pool_key=? AND ticket=? AND waiter_id=? AND state='waiting'",
                (now, self.pool_key, ticket, waiter_id),
            ).rowcount == 1
            if changed:
                self._reap_and_arm(conn, now)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        if changed:
            self._broadcast()

    def acquire(self):
        started_mono = time.monotonic()
        queued_at = time.time()
        queue_deadline = (
            None if self.queue_timeout is None
            else started_mono + float(self.queue_timeout)
        )
        watch = _InotifyWatch(self.notify_path)
        ticket = -1
        waiter_id = ""
        try:
            while True:
                if ticket < 0:
                    ticket, waiter_id = self._enqueue()
                if watch.drain():
                    watch.close()
                    self._secure_file(self.notify_path)
                    watch = _InotifyWatch(self.notify_path)
                result = self._try_claim(ticket, waiter_id, queued_at)
                if result.lease is not None:
                    try:
                        if result.changed:
                            self._broadcast()
                    except BaseException:
                        self.release(result.lease)
                        raise
                    return result.lease
                if result.changed:
                    self._broadcast()
                if result.state == "expired":
                    ticket, waiter_id = -1, ""
                    continue
                if result.state != "waiting":
                    raise AdmissionError(f"unexpected waiter state {result.state!r}")
                now_mono = time.monotonic()
                if queue_deadline is not None and now_mono >= queue_deadline:
                    raise AdmissionQueueTimeout(
                        f"timed out waiting for {self.pool_key!r}"
                    )
                waits = []
                if result.next_wall_deadline is not None:
                    waits.append(max(0, result.next_wall_deadline - time.time()))
                if queue_deadline is not None:
                    waits.append(max(0, queue_deadline - now_mono))
                watch.wait(min(waits) if waits else None)
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
            now = time.time()
            updated = conn.execute(
                "UPDATE admission_slots SET lease_id=NULL,waiter_id=NULL,"
                "ticket=NULL,owner_pid=NULL,acquired_at=NULL,expires_at=NULL "
                "WHERE pool_key=? AND slot_no=? AND lease_id=? AND waiter_id=?",
                (self.pool_key, lease.slot_no, lease.lease_id, lease.waiter_id),
            ).rowcount
            if updated:
                conn.execute(
                    "UPDATE admission_waiters SET state='finished',slot_no=NULL,"
                    "lease_id=NULL,finished_at=? WHERE pool_key=? AND ticket=? "
                    "AND waiter_id=? AND lease_id=? AND state='active'",
                    (now, self.pool_key, lease.ticket, lease.waiter_id,
                     lease.lease_id),
                )
                self._reap_and_arm(conn, now)
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
CREATE TABLE IF NOT EXISTS admission_pools (
    pool_key TEXT PRIMARY KEY,
    capacity INTEGER NOT NULL CHECK(capacity>0),
    next_ticket INTEGER NOT NULL CHECK(next_ticket>0),
    rate_per_minute REAL CHECK(rate_per_minute>0),
    burst_capacity REAL CHECK(burst_capacity>=1),
    available_tokens REAL CHECK(available_tokens>=0),
    tokens_updated_at REAL
);
CREATE TABLE IF NOT EXISTS admission_slots (
    pool_key TEXT NOT NULL, slot_no INTEGER NOT NULL, lease_id TEXT,
    waiter_id TEXT, ticket INTEGER, owner_pid INTEGER, acquired_at REAL,
    expires_at REAL, PRIMARY KEY(pool_key,slot_no),
    FOREIGN KEY(pool_key) REFERENCES admission_pools(pool_key)
);
CREATE TABLE IF NOT EXISTS admission_waiters (
    pool_key TEXT NOT NULL, ticket INTEGER NOT NULL,
    waiter_id TEXT NOT NULL UNIQUE, state TEXT NOT NULL CHECK(
      state IN ('waiting','active','finished','expired','cancelled')),
    queued_at REAL NOT NULL, acquired_at REAL, head_claim_expires_at REAL,
    slot_no INTEGER, lease_id TEXT, finished_at REAL,
    PRIMARY KEY(pool_key,ticket),
    FOREIGN KEY(pool_key) REFERENCES admission_pools(pool_key)
);
CREATE INDEX IF NOT EXISTS admission_waiting_order
ON admission_waiters(pool_key,state,ticket);
CREATE INDEX IF NOT EXISTS admission_active_expiration
ON admission_slots(pool_key,expires_at) WHERE lease_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS admission_active_slot
ON admission_waiters(pool_key,slot_no) WHERE state='active';
"""
