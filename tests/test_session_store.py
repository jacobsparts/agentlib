from datetime import datetime, timedelta, timezone

from agentlib.session_store import SessionStore


def test_fork_session_copies_events_and_preview_blobs_without_lock(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    source = store.create_session("/repo", "model-a")
    store.append_event(source, 1, "message_added", {"message": {"role": "user", "_user_content": "hello"}})
    store.append_event(source, 2, "display", {"kind": "status", "text": "ok\n"})
    store.save_preview_blob(source, "abc", "blob content")
    ok, _ = store.acquire_session_lock(source, "owner-a")
    assert ok

    forked = store.fork_session(source, cwd="/other", model="model-b")

    forked_session = store.get_session(forked)
    assert forked_session["cwd"] == "/other"
    assert forked_session["model"] == "model-b"
    assert forked_session["last_user_text"] == "hello"
    assert store.get_events(forked) == store.get_events(source)
    assert store.get_preview_blob(forked, "abc") == "blob content"
    assert store.get_session_lock(forked) is None
    assert store.get_session_lock(source)["owner"] == "owner-a"


def test_session_lock_blocks_other_owner_until_released(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")

    ok, lock = store.acquire_session_lock(session_id, "owner-a")
    assert ok
    assert lock is None

    ok, lock = store.acquire_session_lock(session_id, "owner-b")
    assert not ok
    assert lock["owner"] == "owner-a"

    assert store.heartbeat_session_lock(session_id, "owner-a")
    assert store.release_session_lock(session_id, "owner-a")

    ok, lock = store.acquire_session_lock(session_id, "owner-b")
    assert ok
    assert lock is None


def test_session_lock_can_be_stolen_after_expiry(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")

    ok, _ = store.acquire_session_lock(session_id, "owner-a", ttl_seconds=-1)
    assert ok
    assert store.get_session_lock(session_id) is None

    ok, lock = store.acquire_session_lock(session_id, "owner-b")
    assert ok
    assert lock is None
    assert store.get_session_lock(session_id)["owner"] == "owner-b"


def test_session_lock_same_owner_refreshes(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    session_id = store.create_session("/repo", "model")

    ok, _ = store.acquire_session_lock(session_id, "owner-a", ttl_seconds=1)
    assert ok
    first = store.get_session_lock(session_id)

    ok, _ = store.acquire_session_lock(session_id, "owner-a", ttl_seconds=120)
    assert ok
    second = store.get_session_lock(session_id)

    assert second["owner"] == "owner-a"
    assert datetime.fromisoformat(second["expires_at"]) >= datetime.fromisoformat(first["expires_at"])



def test_create_session_from_messages_appends_events_and_copies_preview_blobs(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    source = store.create_session("/repo", "model-a")
    store.append_event(source, 1, "message_added", {"message": {"role": "user", "_user_content": "source"}})
    store.save_preview_blob(source, "abc", "blob content")
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": "condensed",
            "_attachments": {"file.txt": "content"},
            "_attachment_refs": {"blob": "session://preview/abc"},
            "_render_segments": [{"type": "stdout", "content": "condensed"}],
        },
        {"role": "assistant", "content": "emit('ok')"},
    ]

    created = store.create_session_from_messages(
        messages,
        cwd="/new",
        model="model-b",
        preview_blobs_from=source,
    )

    assert created != source
    assert store.get_session(created)["cwd"] == "/new"
    assert store.get_session(created)["model"] == "model-b"
    assert store.get_preview_blob(created, "abc") == "blob content"
    events = store.get_events(created)
    assert [event["seq"] for event in events] == [1, 2]
    assert [event["event_type"] for event in events] == ["message_added", "message_added"]
    first_message = events[0]["payload"]["message"]
    assert first_message["role"] == "user"
    assert first_message["_attachment_refs"] == {"blob": "session://preview/abc"}
    assert "_attachments" not in first_message
    assert store.get_events(source)[0]["payload"]["message"]["_user_content"] == "source"