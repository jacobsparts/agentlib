import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_db_path() -> Path:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_cwd ON sessions(cwd)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_events_session_seq ON session_events(session_id, seq)")
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
                if msg.get("role") == "user":
                    last_user_text = msg.get("_user_content") or msg.get("content", "")
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