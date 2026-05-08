import json
import os
import socket
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_db_path() -> Path:
    if override := os.getenv("AGENTLIB_SESSION_DB"):
        path = Path(override).expanduser()
    else:
        path = Path.home() / ".agentlib" / "sessions.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class SessionStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = str(Path(db_path).expanduser()) if db_path else str(resolve_db_path())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    cwd TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    model TEXT,
                    title TEXT,
                    last_user_text TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE(session_id, seq),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preview_blobs (
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    content TEXT NOT NULL,
                    PRIMARY KEY(session_id, key),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_locks (
                    session_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    hostname TEXT,
                    acquired_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_cwd ON sessions(cwd)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_events_session_seq ON session_events(session_id, seq)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_locks_expires_at ON session_locks(expires_at)")
            conn.commit()

    def create_session(self, cwd: str, model: str | None = None) -> str:
        session_id = str(uuid.uuid4())
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, cwd, created_at, updated_at, model)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, cwd, now, now, model),
            )
            conn.commit()
        return session_id

    def fork_session(self, source_session_id: str, *, cwd: str | None = None, model: str | None = None) -> str:
        new_session_id = str(uuid.uuid4())
        now = utc_now_iso()
        with self._connect() as conn:
            source = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (source_session_id,),
            ).fetchone()
            if source is None:
                raise ValueError(f"Session not found: {source_session_id}")

            conn.execute(
                """
                INSERT INTO sessions(session_id, cwd, created_at, updated_at, model, title, last_user_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_session_id,
                    cwd if cwd is not None else source["cwd"],
                    now,
                    now,
                    model if model is not None else source["model"],
                    source["title"],
                    source["last_user_text"],
                ),
            )
            conn.execute(
                """
                INSERT INTO session_events(session_id, seq, created_at, event_type, payload_json)
                SELECT ?, seq, created_at, event_type, payload_json
                FROM session_events
                WHERE session_id = ?
                ORDER BY seq ASC
                """,
                (new_session_id, source_session_id),
            )
            conn.execute(
                """
                INSERT INTO preview_blobs(session_id, key, created_at, content)
                SELECT ?, key, created_at, content
                FROM preview_blobs
                WHERE session_id = ?
                """,
                (new_session_id, source_session_id),
            )
            conn.commit()
        return new_session_id

    def _lock_expiry(self, ttl_seconds: int) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()

    @staticmethod
    def default_lock_owner() -> str:
        return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}"

    def acquire_session_lock(self, session_id: str, owner: str, ttl_seconds: int = 3600) -> tuple[bool, dict | None]:
        now = utc_now_iso()
        expires_at = self._lock_expiry(ttl_seconds)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT * FROM session_locks WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if existing is not None and existing["owner"] != owner:
                    parsed_expiry = datetime.fromisoformat(existing["expires_at"])
                    if parsed_expiry.tzinfo is None:
                        parsed_expiry = parsed_expiry.replace(tzinfo=timezone.utc)
                    if parsed_expiry > datetime.now(timezone.utc):
                        conn.commit()
                        return False, dict(existing)

                conn.execute(
                    """
                    INSERT INTO session_locks(session_id, owner, pid, hostname, acquired_at, heartbeat_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        owner = excluded.owner,
                        pid = excluded.pid,
                        hostname = excluded.hostname,
                        acquired_at = excluded.acquired_at,
                        heartbeat_at = excluded.heartbeat_at,
                        expires_at = excluded.expires_at
                    """,
                    (session_id, owner, os.getpid(), socket.gethostname(), now, now, expires_at),
                )
                conn.commit()
                return True, None
            except Exception:
                conn.rollback()
                raise

    def heartbeat_session_lock(self, session_id: str, owner: str, ttl_seconds: int = 3600) -> bool:
        now = utc_now_iso()
        expires_at = self._lock_expiry(ttl_seconds)
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE session_locks
                SET heartbeat_at = ?, expires_at = ?
                WHERE session_id = ? AND owner = ?
                """,
                (now, expires_at, session_id, owner),
            )
            conn.commit()
            return cur.rowcount > 0

    def release_session_lock(self, session_id: str, owner: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM session_locks WHERE session_id = ? AND owner = ?",
                (session_id, owner),
            )
            conn.commit()
            return cur.rowcount > 0

    def get_session_lock(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_locks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        lock = dict(row)
        parsed_expiry = datetime.fromisoformat(lock["expires_at"])
        if parsed_expiry.tzinfo is None:
            parsed_expiry = parsed_expiry.replace(tzinfo=timezone.utc)
        if parsed_expiry <= datetime.now(timezone.utc):
            return None
        return lock

    def append_event(self, session_id: str, seq: int, event_type: str, payload: dict) -> int:
        now = utc_now_iso()
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_events(session_id, seq, created_at, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, seq, now, event_type, payload_json),
            )
            last_user_text = None
            if event_type == "message_added":
                msg = payload.get("message", {})
                if msg.get("role") == "user" and msg.get("_user_content"):
                    last_user_text = msg["_user_content"]
            conn.execute(
                """
                UPDATE sessions
                SET updated_at = ?, last_user_text = COALESCE(?, last_user_text)
                WHERE session_id = ?
                """,
                (now, last_user_text, session_id),
            )
            conn.commit()
        return seq

    def get_events(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT seq, created_at, event_type, payload_json
                FROM session_events
                WHERE session_id = ?
                ORDER BY seq ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "seq": row["seq"],
                "created_at": row["created_at"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def get_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_next_seq(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM session_events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["max_seq"]) + 1

    def list_sessions(self, cwd: str | None = None, limit: int = 100) -> list[dict]:
        query = "SELECT * FROM sessions"
        args = []
        if cwd is not None:
            query += " WHERE cwd = ?"
            args.append(cwd)
        query += " ORDER BY updated_at DESC LIMIT ?"
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()
        return [dict(row) for row in rows]

    def save_preview_blob(self, session_id: str, key: str, content: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO preview_blobs(session_id, key, created_at, content)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, key, now, content),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()

    def get_preview_blob(self, session_id: str, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM preview_blobs WHERE session_id = ? AND key = ?",
                (session_id, key),
            ).fetchone()
        return row["content"] if row else None
