"""Simple host-wide provider admission using current model configuration."""

from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional, Union

_DEFAULT_DB = Path("~/.agentlib/.provider_admission.sqlite3").expanduser()
_POLL_SECONDS = 0.05
_LEASE_GRACE_SECONDS = 30.0


class AdmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdmissionLease:
    pool_key: str
    lease_id: str
    acquired_at: float
    expires_at: float


def admission_path() -> Path:
    return Path(
        os.environ.get("AGENTLIB_ADMISSION_DB", str(_DEFAULT_DB))
    ).expanduser()


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


class ProviderAdmission:
    def __init__(
        self,
        pool_key: str,
        capacity: int,
        *,
        request_timeout: Optional[float],
        rate_per_minute: Optional[float] = None,
        db_path: Optional[Union[Path, str]] = None,
    ):
        if not pool_key:
            raise ValueError("pool_key must not be empty")
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if rate_per_minute is not None and (
            isinstance(rate_per_minute, bool) or float(rate_per_minute) <= 0
        ):
            raise ValueError("rate_per_minute must be positive")

        self.pool_key = pool_key
        self.capacity = int(capacity)
        self.request_timeout = request_timeout
        self.rate_per_minute = (
            None if rate_per_minute is None else float(rate_per_minute)
        )
        self.db_path = Path(db_path or admission_path()).expanduser()
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

    def _ensure_storage(self) -> None:
        if self._initialized:
            return
        with self._initialize_lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(self.db_path, flags, 0o600)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise AdmissionError(f"not a regular file: {self.db_path}")
                if hasattr(os, "getuid") and info.st_uid != os.getuid():
                    raise AdmissionError(
                        f"file is owned by another user: {self.db_path}"
                    )
                os.fchmod(fd, 0o600)
            finally:
                os.close(fd)

            conn = self._connect()
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.executescript(_SCHEMA)
            finally:
                conn.close()
            os.chmod(self.db_path, 0o600)
            self._initialized = True

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _lease_duration(self) -> float:
        timeout = 600.0 if self.request_timeout is None else float(
            self.request_timeout
        )
        return timeout + _LEASE_GRACE_SECONDS

    def _remove_stale_leases(self, conn, now: float) -> None:
        rows = conn.execute(
            "SELECT lease_id,owner_pid,expires_at FROM provider_leases "
            "WHERE pool_key=?",
            (self.pool_key,),
        ).fetchall()
        stale = [
            row["lease_id"]
            for row in rows
            if row["expires_at"] <= now or not _pid_is_alive(row["owner_pid"])
        ]
        if stale:
            conn.executemany(
                "DELETE FROM provider_leases WHERE lease_id=?",
                ((lease_id,) for lease_id in stale),
            )

    def _try_acquire(self) -> tuple[Optional[AdmissionLease], Optional[float]]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            self._remove_stale_leases(conn, now)
            conn.execute(
                "DELETE FROM provider_admissions WHERE admitted_at<=?",
                (now - 60.0,),
            )

            active = conn.execute(
                "SELECT COUNT(*) FROM provider_leases WHERE pool_key=?",
                (self.pool_key,),
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
            if active < self.capacity and rate_ready:
                lease_id = uuid.uuid4().hex
                expires_at = now + self._lease_duration()
                conn.execute(
                    "INSERT INTO provider_leases"
                    "(pool_key,lease_id,owner_pid,acquired_at,expires_at) "
                    "VALUES (?,?,?,?,?)",
                    (
                        self.pool_key,
                        lease_id,
                        os.getpid(),
                        now,
                        expires_at,
                    ),
                )
                if self.rate_per_minute is not None:
                    conn.execute(
                        "INSERT INTO provider_admissions(pool_key,admitted_at) "
                        "VALUES (?,?)",
                        (self.pool_key, now),
                    )
                lease = AdmissionLease(
                    self.pool_key, lease_id, now, expires_at
                )

            lease_deadline = conn.execute(
                "SELECT MIN(expires_at) FROM provider_leases WHERE pool_key=?",
                (self.pool_key,),
            ).fetchone()[0]
            deadlines = [
                deadline
                for deadline in (lease_deadline, rate_deadline)
                if deadline is not None
            ]
            conn.commit()
            return lease, min(deadlines) if deadlines else None
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def acquire(self) -> AdmissionLease:
        while True:
            lease, deadline = self._try_acquire()
            if lease is not None:
                return lease
            delay = _POLL_SECONDS
            if deadline is not None:
                delay = min(delay, max(0.001, deadline - time.time()))
            time.sleep(delay)

    def release(self, lease: AdmissionLease) -> bool:
        if lease.pool_key != self.pool_key:
            raise ValueError("lease belongs to another pool")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            removed = conn.execute(
                "DELETE FROM provider_leases "
                "WHERE pool_key=? AND lease_id=?",
                (self.pool_key, lease.lease_id),
            ).rowcount
            conn.commit()
            return bool(removed)
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextlib.contextmanager
    def admitted(self) -> Iterator[AdmissionLease]:
        lease = self.acquire()
        try:
            yield lease
        finally:
            self.release(lease)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_leases (
    pool_key TEXT NOT NULL,
    lease_id TEXT PRIMARY KEY,
    owner_pid INTEGER NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS provider_leases_pool
ON provider_leases(pool_key);

CREATE TABLE IF NOT EXISTS provider_admissions (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pool_key TEXT NOT NULL,
    admitted_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS provider_admissions_pool_time
ON provider_admissions(pool_key,admitted_at);
"""
