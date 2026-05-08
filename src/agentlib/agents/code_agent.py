#!/usr/bin/env python3
"""Code assistant with Python REPL execution and native tools.

Combines REPLAgent with CLIMixin to create an interactive coding assistant
that executes Python code directly. Uses native implementations for file
operations, ripgrep for search, and Jina AI for web access.

Dependencies:
    - ripgrep (rg) must be installed: apt install ripgrep
    - JINA_API_KEY env var for higher rate limits (optional, get free key at https://jina.ai/?sui=apikey)
"""

from agentlib.code_agent_preprocess import preprocess_code_agent
import base64
import json
import os
import shutil
import sys
import copy
import re
from typing import Optional
from pathlib import Path
from agentlib import REPLAgent, SandboxMixin, REPLAttachmentMixin, MCPMixin
from agentlib.cli import CLIMixin
from agentlib.jina_mixin import JinaMixin
from agentlib.llm_registry import ModelNotFoundError
from agentlib.cli.terminal import DIM, RESET, Panel, strip_ansi
from agentlib.session_store import SessionStore
from agentlib.session_replay import replay_session_into_agent, replay_display_text
from dotenv import load_dotenv

load_dotenv()

if not shutil.which('rg'):
    sys.exit("Error: ripgrep (rg) is required but not found. Install with: apt install ripgrep")

#import logging; logging.getLogger('agentlib').setLevel(logging.DEBUG)


def _get_config_value(attr_name, default):
    """Lazy load user config value."""
    from agentlib.config import get_user_config
    config = get_user_config()
    return getattr(config, attr_name, default) if config else default


def gather_auto_attach_files():
    '''Find CLAUDE.md or AGENTS.md files and their @ imports.

    Searches current directory and parent directories for CLAUDE.md or AGENTS.md.
    Recursively processes @ imports (lines starting with @ followed by filename).
    Returns list of file paths relative to current directory, with no duplicates.
    '''
    current = Path.cwd()
    found_files = []
    seen_paths = set()

    def add_file_and_imports(file_path: Path, base_dir: Path):
        'Recursively add file and its imports.'
        # Normalize to absolute path for deduplication
        abs_path = file_path.resolve()
        if abs_path in seen_paths:
            return
        seen_paths.add(abs_path)

        # Add to results (relative to cwd)
        rel_path = os.path.relpath(abs_path, current)
        found_files.append(rel_path)

        # Scan for @ imports
        try:
            content = file_path.read_text()
            for line in content.split('\n'):
                if line.startswith('@'):
                    import_name = line[1:].strip()
                    if import_name:
                        # Resolve relative to the directory containing this file
                        import_path = (file_path.parent / import_name).resolve()
                        if import_path.exists() and import_path.is_file():
                            add_file_and_imports(import_path, file_path.parent)
        except Exception:
            pass  # Ignore read errors

    # Search for CLAUDE.md or AGENTS.md in current and parent directories
    md_files = []
    search_dir = current
    while True:
        for name in ['CLAUDE.md', 'AGENTS.md']:
            candidate = search_dir / name
            if candidate.exists():
                md_files.append(candidate)

        parent = search_dir.parent
        if parent == search_dir:  # Reached root
            break
        search_dir = parent

    # Sort so parent directories come first (reverse order of discovery)
    md_files.reverse()

    # Process each markdown file and its imports
    for md_file in md_files:
        add_file_and_imports(md_file, md_file.parent)

    return found_files


def _skill_description(path: Path) -> str:
    try:
        for line in path.read_text().splitlines():
            text = line.strip()
            if not text:
                continue
            if text.startswith('#'):
                text = text.lstrip('#').strip()
            return text
    except Exception:
        pass
    return ""


class CodeAgentBase(REPLAttachmentMixin, CLIMixin, REPLAgent):
    """Code assistant with Python REPL execution."""

    model = _get_config_value("code_agent_model", "sonnet")

    def _ensure_setup(self):
        super()._ensure_setup()
        if not hasattr(self, '_session_store'):
            self._session_store = SessionStore()
            self._session_id = None
            self._next_event_seq = 1
            self._suspend_persistence = False
            self._explicit_attachment_refs = {}
            self._pending_explicit_attachment_refs = {}
            self._pending_session_events = []
            self._display_capture = []
            self._pending_unviewed_files = set()
            self._auto_context_attachment_names = set()
        self.llm_client.on_retry = self.on_retry

    def _ensure_live_session(self):
        if self._session_id is None:
            self._session_id = self._session_store.create_session(str(Path.cwd()), getattr(self, 'model', None))
            self._next_event_seq = 1

    def _flush_pending_session_events(self):
        if not self._session_id:
            return
        pending = self._pending_session_events
        self._pending_session_events = []
        for event_type, payload in pending:
            seq = self._next_event_seq
            self._session_store.append_event(self._session_id, seq, event_type, payload)
            self._next_event_seq += 1

    def _bootstrap_persisted_conversation(self):
        if getattr(self, '_suspend_persistence', False):
            return
        if self._session_id is None:
            self._ensure_live_session()
            self._flush_pending_session_events()
        for msg in self.conversation.messages[1:]:
            if msg.get('_synthetic'):
                continue
            if msg.get('_event_seq') is None:
                self._persist_message(msg)

    def _append_session_event(self, event_type: str, payload: dict, create_session: bool = True) -> int | None:
        if getattr(self, '_suspend_persistence', False):
            return None
        if self._session_id is None and not create_session:
            self._pending_session_events.append((event_type, copy.deepcopy(payload)))
            return None
        self._ensure_live_session()
        self._flush_pending_session_events()
        seq = self._next_event_seq
        self._session_store.append_event(self._session_id, seq, event_type, payload)
        self._next_event_seq += 1
        return seq

    def _record_display_event(self, kind: str, text: str, create_session: bool = False):
        if not text:
            return
        self._append_session_event("display", {"kind": kind, "text": text}, create_session=create_session)

    def _display_text(self, text: str, kind: str = "status", end: str = "\n", create_session: bool = False):
        print(text, end=end, flush=True)
        self._record_display_event(kind, strip_ansi(text) + end, create_session=create_session)

    def _display_input_block(self, text: str):
        lines = text.rstrip("\n").split("\n") if text else [""]
        rendered = [f"{self.cli_prompt}{lines[0]}"]
        rendered.extend(lines[1:])
        self._record_display_event("input", "\n".join(rendered) + "\n\n")

    def _replay_display_output(self):
        if not self._session_id:
            return
        display_text = replay_display_text(self._session_id, self._session_store, format_response=self.format_response)
        sys.stdout.write(display_text)
        if display_text and not display_text.endswith("\n"):
            print()

    def _reset_display_capture(self):
        self._display_capture = []

    def _capture_display_line(self, text: str = ""):
        self._display_capture.append(strip_ansi(text) + "\n")

    def _show_python_header_if_pending(self):
        if getattr(self, '_header_pending', False):
            header = "\x1b[34m"+("─"*13)+" Python "+("─"*13)+"\x1b[0m"
            print(f"\x1b[1G\x1b[K{header}")
            self._capture_display_line("───────────── Python ─────────────")
            self._header_pending = False
            self._repl_printed_header = True

    def _flush_edit_echo_buffer(self):
        buffer = getattr(self, '_edit_echo_buffer', [])
        if not buffer:
            return
        self._show_python_header_if_pending()
        for line in buffer:
            print(line, flush=True)
            self._capture_display_line(line)
        self._edit_echo_buffer = []
        self._in_edit_echo = False

    def _compact_edit_echo(self, diff: str) -> str:
        buffer = getattr(self, '_edit_echo_buffer', [])
        first = buffer[0] if buffer else ""
        func = "line_patch" if first.startswith(">>> line_patch(") else "edit"

        filename = None
        for prefix in ("+++ ", "--- "):
            for line in diff.splitlines():
                if line.startswith(prefix):
                    candidate = line[len(prefix):].strip()
                    if candidate != "/dev/null":
                        filename = candidate
                        break
            if filename is not None:
                break

        if filename is None:
            return f">>> {func}(...)"
        return f">>> {func}({filename!r}, ...)"

    def _flush_display_capture(self):
        if not self._display_capture:
            return
        self._record_display_event("python", "".join(self._display_capture))
        self._display_capture = []

    def _sanitize_message_for_persistence(self, message: dict) -> dict:
        def encode_media(value):
            if isinstance(value, bytes):
                return {"__b64__": base64.b64encode(value).decode("ascii")}
            if isinstance(value, list):
                return [encode_media(item) for item in value]
            return copy.deepcopy(value)

        msg = {}
        for key, value in message.items():
            if key == '_attachments':
                continue
            if key in {'images', 'audio'}:
                msg[key] = encode_media(value)
            elif key in {'role', 'content', '_stdout', '_user_content', 'name', 'tool_call_id', '_synthetic', '_render_segments', '_final_result', '_emit_value'}:
                msg[key] = copy.deepcopy(value)
        refs = message.get('_attachment_refs')
        if refs:
            msg['_attachment_refs'] = copy.deepcopy(refs)
        return msg

    def _tag_latest_segment_seq(self, message: dict, seq: int):
        for seg in reversed(message.get("_render_segments") or []):
            if "_event_seq" not in seg:
                seg["_event_seq"] = seq
                break

    def _persist_message(self, message: dict):
        if message.get('_synthetic'):
            return
        if self._session_id is None:
            self._bootstrap_persisted_conversation()
            if message.get("_event_seq") is not None:
                return
        seq = self._append_session_event("message_added", {"message": self._sanitize_message_for_persistence(message)})
        if seq is not None:
            message["_event_seq"] = seq
            self._tag_latest_segment_seq(message, seq)

    def _persist_append_to_last_user_message(self, target_message: dict, content: str, kwargs: dict):
        user_content = target_message.get('_user_content', content)
        latest_segment = (target_message.get("_render_segments") or [None])[-1]
        new_msg = {"role": "user", "content": content, "_user_content": user_content}
        if kwargs.get('_stdout') is not None:
            new_msg["_stdout"] = kwargs['_stdout']
        if kwargs.get('_attachment_refs'):
            new_msg["_attachment_refs"] = copy.deepcopy(kwargs['_attachment_refs'])
        for key in ('images', 'audio'):
            if kwargs.get(key):
                new_msg[key] = kwargs[key]
        if latest_segment is not None:
            new_msg["_render_segments"] = [{k: v for k, v in latest_segment.items() if k != "_event_seq"}]
        seq = self._append_session_event("message_added", {"message": self._sanitize_message_for_persistence(new_msg)})
        if seq is not None:
            target_message["_event_seq"] = seq
            self._tag_latest_segment_seq(target_message, seq)

    def _on_append_last_user_message(self, target_message: dict, content, kwargs):
        self._persist_append_to_last_user_message(target_message, content, kwargs)

    def toolmsg(self, content, **kwargs):
        result = super().toolmsg(content, **kwargs)
        self._persist_message(self.conversation.messages[-1])
        return result

    def attach_file_ref(self, filepath: str, name: str | None = None):
        path = str(Path(filepath).expanduser())
        attach_name = name or filepath
        content = Path(path).read_text()
        self.attach(attach_name, content)
        self._explicit_attachment_refs[attach_name] = path
        self._pending_explicit_attachment_refs[attach_name] = path
        self._append_session_event("attachment_added", {"name": attach_name, "path": path}, create_session=False)

    def detach_file_ref(self, name: str):
        self.detach(name)
        self._explicit_attachment_refs.pop(name, None)
        self._pending_explicit_attachment_refs.pop(name, None)
        self._append_session_event("attachment_removed", {"name": name}, create_session=False)

    def _builtin_skills_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "skills"

    def _user_skills_dir(self) -> Path:
        return Path.home() / ".agentlib" / "skills"

    @staticmethod
    def _is_session_uri(name: str) -> bool:
        return isinstance(name, str) and name.startswith("session://")

    def _is_auto_context_file(self, name: str) -> bool:
        return (
            isinstance(name, str)
            and (
                name in getattr(self, '_auto_context_attachment_names', set())
                or Path(name).name in {"CLAUDE.md", "AGENTS.md"}
            )
        )

    def list_attachments(self, include_session_blobs: bool = False, include_auto_context: bool = True) -> dict[str, str]:
        attachments = super().list_attachments()
        if not include_session_blobs:
            attachments = {
                name: content
                for name, content in attachments.items()
                if not self._is_session_uri(name)
            }
        if not include_auto_context:
            attachments = {
                name: content
                for name, content in attachments.items()
                if not self._is_auto_context_file(name)
            }
        return attachments


    def list_skills(self) -> list[dict]:
        skills = {}
        for source, directory in (("built-in", self._builtin_skills_dir()), ("user", self._user_skills_dir())):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.md")):
                name = path.stem
                skills[name] = {
                    "name": name,
                    "file_name": path.name,
                    "path": path,
                    "source": source,
                    "description": _skill_description(path),
                }
        active = set(self.list_attachments())
        items = []
        for name in sorted(skills):
            item = dict(skills[name])
            item["attached"] = item["file_name"] in active
            items.append(item)
        return items

    def resolve_skill(self, name: str) -> dict | None:
        skill_name = name.strip()
        if skill_name.endswith(".md"):
            skill_name = skill_name[:-3]
        for item in self.list_skills():
            if item["name"] == skill_name:
                return item
        return None

    def attach_skill(self, name: str) -> tuple[bool, str]:
        skill = self.resolve_skill(name)
        if not skill:
            return False, f"Skill not found: {name}"
        self.attach_file_ref(str(skill["path"]), skill["file_name"])
        return True, f"Attached skill: {skill['name']} [{skill['source']}]"

    def apply_skill_selection(self, selected_skills: list[dict]) -> list[str]:
        current = {item["file_name"]: item for item in self.list_skills() if item["attached"]}
        desired = {item["file_name"]: item for item in selected_skills if item["attached"]}
        messages = []
        for file_name, item in current.items():
            if file_name not in desired:
                self.detach_file_ref(file_name)
                messages.append(f"Detached skill: {item['name']}")
        for file_name, item in desired.items():
            if file_name not in current:
                self.attach_file_ref(str(item["path"]), file_name)
                messages.append(f"Attached skill: {item['name']} [{item['source']}]")
        return messages

    def _invalidate_attachment(self, name: str):
        had_attachment = any(name in msg.get('_attachments', {}) for msg in self.conversation.messages)
        had_ref = any(name in (msg.get('_attachment_refs') or {}) for msg in self.conversation.messages)
        had_pending_ref = name in self._explicit_attachment_refs or name in self._pending_explicit_attachment_refs
        super()._invalidate_attachment(name)
        for msg in self.conversation.messages:
            refs = msg.get('_attachment_refs')
            if refs and name in refs:
                del refs[name]
                if not refs:
                    del msg['_attachment_refs']
        self._explicit_attachment_refs.pop(name, None)
        self._pending_explicit_attachment_refs.pop(name, None)
        if getattr(self, '_suspend_persistence', False):
            return
        if not (had_attachment or had_ref or had_pending_ref):
            return
        self._append_session_event("attachment_invalidated", {"name": name})

    def _on_assistant_message_committed(self, message: dict):
        self._persist_message(message)

    def build_output_for_llm(self, output_chunks):
        """Build LLM output, converting complete reads to attachments."""
        self._read_attachments = {}
        result = []
        attach_path = None
        partial_read_path = None
        written_files = []
        unviewed_files = set(getattr(self, '_pending_unviewed_files', set()))
        self._pending_unviewed_files = set()
        for msg_type, chunk in output_chunks:
            if msg_type == "emit":
                continue
            if msg_type == "file_unviewed":
                unviewed_files.add(chunk.strip())
                continue
            if msg_type == "read_attach":
                attach_path = chunk.strip()
                continue
            if msg_type == "read_partial":
                partial_read_path = chunk.strip()
                continue
            if msg_type == "read" and attach_path:
                path = attach_path
                attach_path = None
                if path in unviewed_files:
                    result.append(chunk)
                    continue
                self._invalidate_attachment(path)
                self._read_attachments[path] = chunk.rstrip('\n')
                result.append(f"[Attachment: {path}]\n")
                continue
            if msg_type == "read" and partial_read_path:
                path = partial_read_path
                partial_read_path = None
                if self._is_attached(path):
                    result.append(f"ValueError: Partial view denied for file already in context. Call view() without offset or limit to reload the file, or call unview({path!r}) to remove it from future context and enable partial views.\n")
                    continue
                result.append(chunk)
                continue
            if msg_type == "file_written":
                written_files.append(chunk.strip())
                continue
            if msg_type == "file_diff":
                continue

            attach_path = None
            partial_read_path = None
            result.append(chunk)

        # Auto-refresh: re-read attached files that were written but not re-read by agent
        for path in written_files:
            if path in self._read_attachments:
                continue  # Agent already re-read this file
            attached_name = self._attached_file_name(path)
            if attached_name is None:
                continue  # File wasn't attached
            try:
                content = Path(path).read_text()
                lines = content.split('\n')
                formatted = '\n'.join(f"{i+1:>5}→{line}" for i, line in enumerate(lines))
                self._invalidate_attachment(attached_name)
                self._read_attachments[attached_name] = formatted
                result.append(f">>> view({attached_name!r})\n[Attachment: {attached_name}]\n")
            except Exception:
                pass

        return "".join(result)

    def _same_file(self, left: str, right: str) -> bool:
        try:
            return os.path.samefile(left, right)
        except (FileNotFoundError, OSError, TypeError):
            try:
                return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
            except (OSError, RuntimeError):
                return left == right

    def _attached_file_name(self, path: str) -> str | None:
        """Return the active attachment name for a filesystem path."""
        for msg in self.conversation.messages:
            for name in msg.get('_attachments', {}):
                if self._same_file(name, path):
                    return name
        return None

    def _is_attached(self, name: str) -> bool:
        """Check if a file is currently attached in any message."""
        return self._attached_file_name(name) is not None

    @REPLAgent.tool
    def view_images(self,
            files: list[str | bytes] = "List of image filepaths or binary data",
            notes: str = "Observations, objectives, what to look for"
        ):
        '''Load images into context for visual analysis on next turn.'''
        images = []
        total_bytes = 0

        if not isinstance(files, list):
            files = [files]

        for data in files:
            # Stub reads files in REPL, so we should only get bytes here
            if not isinstance(data, bytes):
                raise TypeError(f"Expected bytes, got {type(data).__name__}")

            # Validate JPEG or PNG
            if len(data) < 4:
                raise ValueError("Invalid image data (too short)")

            is_jpeg = data.startswith(b'\xff\xd8\xff')
            is_png = data.startswith(b'\x89PNG')

            if not (is_jpeg or is_png):
                raise ValueError(f"Unsupported image format ({len(data)} bytes) - only JPEG and PNG supported")

            images.append(data)
            total_bytes += len(data)

        self._pending_images = getattr(self, '_pending_images', []) + images
        return f"{len(images)} image(s) queued ({total_bytes // 1000}KB) - {notes}"

    view_images._tool_files_param = "files"

    def _file_context_ephemeral(self, names: list[str]) -> str:
        if not names:
            return ""
        lines = ["Files currently in context:"]
        lines.extend(f"- {name}" for name in names)
        lines.extend(["", "Remove files that are irrelevant to recent conversation state with unview(path)."])
        return "\n".join(lines)

    def _current_file_context_names(self, extra=None) -> list[str]:
        names = {}
        for name in self.list_attachments(include_auto_context=False):
            names[name] = None
        for name in (extra or {}):
            if not self._is_session_uri(name) and not self._is_auto_context_file(name):
                names[name] = None
        return list(names)

    def usermsg(self, content, **kwargs):
        """Override to attach pending images and read-attachments."""
        if getattr(self, '_pending_explicit_attachment_refs', None):
            refs = kwargs.get('_attachment_refs', {})
            refs.update(self._pending_explicit_attachment_refs)
            kwargs['_attachment_refs'] = refs
            self._pending_explicit_attachment_refs = {}
        if pending := getattr(self, '_read_attachments', None):
            existing = kwargs.get('_attachments', {})
            existing.update(pending)
            kwargs['_attachments'] = existing
            refs = kwargs.get('_attachment_refs', {})
            for name in pending:
                refs.setdefault(name, name)
            kwargs['_attachment_refs'] = refs
            self._read_attachments = {}
        if pending := getattr(self, '_pending_images', None):
            kwargs['images'] = kwargs.get('images', []) + pending
            self._pending_images = []
        self.ephemeral = self._file_context_ephemeral(
            self._current_file_context_names(kwargs.get('_attachments'))
        )
        before_len = len(self.conversation.messages)
        result = super().usermsg(content, **kwargs)
        if len(self.conversation.messages) > before_len:
            self._persist_message(self.conversation.messages[-1])
        return result

    welcome_message = "[bold]Code Agent[/bold]\nPython REPL-based coding assistant"
    thinking_message = "Working..."
    interactive = True  # Enables multi-turn autonomous workflow
    max_turns = _get_config_value("code_agent_max_turns", 100)
    system = """>>> help(assistant)

You are an interactive coding assistant operating within a Python REPL.
Your responses ARE Python code—no markdown blocks, no prose preamble.
The code you write is executed directly in a persistent environment.

Every assistant turn must be valid Python source code.
- If you want to communicate with the user, call emit(...)
- Never reply with plain English outside Python code
- If the task is complete, use emit(..., release=True)
- If you are still working, do not release control
- If a prior attempt would have been invalid as Python, immediately correct it
  by sending a new turn containing only valid Python code

The user may see REPL echoes, tool output, and prior emitted text mixed into
the conversation. Treat that transcript as execution context. Continue from the
latest user instruction rather than explaining the transcript unless asked.

The Python environment persists across turns. Variables, imports, connections,
and tool state may already exist from earlier execution. Reuse existing state
when appropriate, but verify assumptions before relying on it.

If the user asks for an opinion or summary and no computation is needed,
respond with emit(...) directly rather than writing unnecessary setup code.

>>> how_this_works()

1. You write Python code as your response (no markdown fences)
2. The code executes in a persistent REPL environment
3. Output from print() and expression results appear IN YOUR NEXT TURN
4. Use emit(value) to output results
5. Use emit(value, release=True) to release control to the user

CRITICAL: You see REPL output in your next turn. The user does NOT control
the conversation until you explicitly release with emit(..., release=True).

>>> emit(value, release=False)

The ONLY way to return results:

    emit("I found 3 issues in the code")              # Output emitted, you KEEP WORKING
    emit("Here's the result: ...", release=True)      # Release control to user

- emit() with release=False (default): Value is emitted but YOU continue
  working. Use this for progress updates when doing long tasks.
- emit() with release=True: Releases control to user. Use when:
  * You need user input: a question, approval, or guidance on next steps
  * You're stuck and need help
  * Requirements are unclear and you need clarification
  * Task is complete AND you have verified the results yourself

Both print() and emit() output are visible. The difference:
- print(): For YOUR inspection in the next turn. Use freely to debug/explore.
- emit(): Deliberate output for the user. Results, questions, or status updates.

>>> autonomous_workflow()

You control execution. The user cannot respond until you call
emit(..., release=True). Work through as many turns as needed.

Do real work on every turn — read files, run commands, write code.
Never emit placeholder turns like print("ready") or print("thinking").

If your final emit includes computed results (test output, command output),
run the computation first, then verify the output on your next turn before
releasing. Do not claim success based on output you haven't reviewed.

Housekeeping: viewed files persist across turns automatically. Use
unview(path) to clean up files that turned out to be irrelevant, were viewed
by mistake, or are no longer needed for the current task.

NEVER:
- Ask permission for read-only operations (reading files, exploring code)
- Ask the user to copy/paste output - you can access it yourself
- Release just to show intermediate results (use print() instead)
- Re-establish database connections each turn (they persist)
- Explain what you're "about to do" - just do it
- Call emit() without release=True unless you're providing a progress update
  on a long-running task
- Release just because a task "should be done" - verify it IS done first

The user CAN interrupt you (Ctrl+C) and drop into the REPL themselves.
But unless they do, YOU are in control until you call emit(..., release=True).

>>> database_connections()

Database connections persist across turns. Set up once:

    import mysql.connector
    conn = mysql.connector.connect(host='localhost', user='...', password='...', database='...')
    cursor = conn.cursor(dictionary=True)
    def q(sql): cursor.execute(sql); return cursor.fetchall()

Then reuse in subsequent turns:

    q("SELECT * FROM users LIMIT 5")
    q("UPDATE users SET active=1 WHERE id=42")

Don't reconnect every turn - the connection object persists.

>>> context_management()

read() returns file contents as text. Use it when you need to assign, search,
split, parse, or otherwise process file contents in Python.

view("file.py") is for inspecting code with line numbers:
- Prefer one full view(file_path) when inspecting a normal-sized source
  file you may need to reason about or edit across turns.
- Do not use view() just to get a string value; use read() for that.
- Do not repeatedly call partial view(..., offset=..., limit=...) on the
  same normal-sized file.

Use partial view(..., offset=..., limit=...) only when:
- the file is genuinely huge, minified, generated, or vendored, or
- you already have the full file in context and need a very narrow line-number
  check, or
- the user explicitly asks for a small line range.

If you accidentally view an irrelevant file, call unview(file_path) to
remove it from future context.

When files change, previous full views stay up to date automatically.
Re-view a file only when you need to inspect it again.

preview(value) prints a potentially long value. Non-strings are previewed via
repr(value). Short values print in full; long values show a head/tail summary
and save the full content to a session://preview/... URI. Pass that URI to
read() to get the full text or view() to inspect it with line numbers.

>>> tone_and_style()

- Prioritize technical accuracy over validation. Disagree when necessary.
- Provide direct, objective technical info without superlatives or praise.
- Investigate uncertainty rather than confirming assumptions.
- NEVER create files unless absolutely necessary. Prefer editing existing files.

>>> doing_tasks()

Before modifying code, view it first. Never propose changes to code you
haven't seen. Use grep() to locate files or anchors, then prefer one full
view() of the target file rather than several narrow view(...,
offset=..., limit=...) slices, unless the file is genuinely huge. Use read()
when you need file contents as a Python text value for processing.

Avoid over-engineering:
- Only make changes directly requested or clearly necessary
- Don't add features, refactoring, or "improvements" beyond what was asked
- Don't add docstrings, comments, or type annotations to unchanged code
- Don't add error handling for scenarios that can't happen
- Don't create abstractions for one-time operations

Security: Be careful not to introduce vulnerabilities (command injection,
XSS, SQL injection, OWASP top 10). Fix insecure code immediately.

>>> file_editing()

Two methods for editing files:

edit(file_path, old_string, new_string, replace_all=False)
    Replace exact string matches.
    - old_string must match exactly (whitespace, indentation)
    - Fails if not found or multiple matches (unless replace_all=True)

line_patch(file_path, patch)
    Edit an existing file by line number.
    - Prefer a full view(file_path) first; if absent, line_patch uses current on-disk contents
    - Line numbers refer to the current viewed file contents, or current file contents if not viewed
    - Operation headers must start at column 1 in the patch string
    - Body lines are literal file content; indent only if the file should contain that indentation
    - Multiple operations in one call are atomic
    - You may call line_patch() repeatedly after one full view(file_path)
    - For create/move/delete, use Python file APIs such as Path.write_text(), Path.rename(), or Path.unlink()

    line_patch("src/app.py", '''replace 10:12
def name():
    return "new"

insert after 25
print("done")

delete 40:44''')

    Operations:
      replace START:END
      delete START:END
      insert before LINE
      insert after LINE

    `insert after 0` prepends to the file.
    `insert before LINE_COUNT + 1` appends to the file.

>>> anti_patterns()

# BAD: Releasing immediately to show what you found
files = list(Path('.').glob("**/*.py"))
emit(f"Found {len(files)} Python files", release=True)  # WRONG - keep working!

# GOOD: Keep working, release when done
files = list(Path('.').glob("**/*.py"))
print(f"Found {len(files)} files")  # You see this, keep going
for f in files[:5]:
    read(str(f))  # Contents appear directly, don't assign
# ... analyze in next turn ...
emit("Analysis complete. Here's what I found: ...", release=True)

# BAD: Asking permission for read-only work
emit("Should I read the config file?", release=True)  # WRONG - just read it

# GOOD: Just do it
read("config.json")  # Contents appear in your next turn

# BAD: Using view() as a value
content = view("file.py")  # WRONG - view() is display-only
print(view("file.py"))     # WRONG - use view("file.py") directly

# GOOD: read() returns text, view() displays/attaches
content = read("file.py")
preview(read("file.py"))
view("file.py")

# BAD: Reading in small chunks unnecessarily
read("file.py", offset=1, limit=50)   # WRONG - just read the whole file

# GOOD: Just call read() directly
read("file.py")

# BAD: Recreating a partial view manually for source inspection
content = read("file.py")
print("\n".join(content.splitlines()[100:140]))

# GOOD: Use view() for source inspection with line numbers/context tracking
view("file.py")

# BAD: Repeated narrow view() calls on the same normal-sized file
view("app.js", offset=2200, limit=40)
view("app.js", offset=2400, limit=30)
view("app.js", offset=3300, limit=20)

# GOOD: Use grep() to locate anchors, then inspect the file once
grep("triggerFindPrompt|focusTerminalFromTouch", "app.js", None, None, False, 0, False, False)
view("app.js")

# GOOD: If you viewed the wrong file, remove it from future context
view("wrong_file.py")
unview("wrong_file.py")

# BAD: Re-establishing connections
conn = mysql.connector.connect(...)  # Every turn? No!

# GOOD: Check if connection exists
if 'conn' not in dir():
    conn = mysql.connector.connect(...)

>>> when_uncertain()

If you don't know how to proceed:
1. Use print() to inspect state and gather information
2. Use think() to reason through the problem
3. Only release with emit(..., release=True) if you truly need user input
"""

    max_output_kb = _get_config_value("code_agent_max_output_kb", 50)  # Large output protection

    def process_repl_output(self, output: str) -> str:
        """Truncate output if too large - used for both display and model."""
        # Truncate if too large
        max_bytes = int(self.max_output_kb * 1000)
        if len(output) > max_bytes:
            import tempfile
            size_kb = len(output) / 1000

            with tempfile.NamedTemporaryFile(
                mode='w', prefix='code_agent-', suffix='.txt', delete=False
            ) as f:
                f.write(output)  # Write full original output
                temp_path = f.name

            # Track for cleanup on exit
            if not hasattr(self, '_temp_files'):
                self._temp_files = []
            self._temp_files.append(temp_path)

            truncated = output[:max_bytes // 2]
            msg = f"[ {size_kb:.1f}KB output truncated - written to {temp_path} ]"
            return f"{truncated}\n\n{msg}"

        return output

    # REPL output hooks
    def on_repl_execute(self, code) -> None:
        """Called at start of each turn."""
        pass  # No-op, display happens in on_repl_output

    def on_repl_chunk(self, chunk: str, msg_type: str = "echo") -> None:
        """Called for each output chunk. Display echo and progress immediately."""
        # Suppress display during direct user REPL mode
        if getattr(self, '_in_user_repl', False):
            return
        if msg_type == "echo":
            # Mark that we've started processing output. Do not clear the
            # carriage-return status line until we have visible output to draw;
            # clearing too early can leave a single non-dim character under the
            # cursor while a long read_attach is processed.
            if not getattr(self, '_turn_output_started', False):
                self._turn_output_started = True
                self.console.clear_line()  # Clear status text
                # Only set header pending if we haven't already printed it this interaction
                if not getattr(self, '_repl_printed_header', False):
                    self._header_pending = True
            # Echo lines already have >>> or ... prefix
            # Skip emit() calls - user sees progress/result via green text
            # Buffer print() calls - decide display based on output truncation
            # Buffer edit()/line_patch() calls - replace with compact echo if a diff arrives
            for line in chunk.rstrip('\n').split('\n'):

                if line.startswith('>>> '):
                    # New statement starting - flush any pending print/edit buffers first
                    # (handles print() with no output or empty output, or edit errors before diff)
                    if getattr(self, '_print_echo_buffer', []):
                        for echo_line in self._print_echo_buffer:
                            self._show_python_header_if_pending()
                            print(echo_line, flush=True)
                            self._capture_display_line(echo_line)
                        self._print_echo_buffer = []
                        self._in_print_echo = False
                    self._flush_edit_echo_buffer()

                    # New statement - check if it's emit(), print(), edit(), or line_patch()
                    self._in_emit_echo = line.startswith('>>> emit(')
                    if line.startswith('>>> print('):
                        self._in_print_echo = True
                        self._print_echo_buffer = [line]
                        # Check if argument is a variable (not a string literal)
                        # String literals start with quotes after print(
                        # Bare print() is treated like a string literal
                        arg_start = line[10:].lstrip()
                        self._print_uses_variable = not (
                            arg_start.startswith(')') or  # print()
                            arg_start.startswith('"') or
                            arg_start.startswith("'") or
                            arg_start.startswith('f"') or
                            arg_start.startswith("f'") or
                            arg_start.startswith('r"') or
                            arg_start.startswith("r'")
                        )
                        continue
                    if line.startswith('>>> edit(') or line.startswith('>>> line_patch('):
                        self._in_edit_echo = True
                        self._edit_echo_buffer = [line]
                        continue
                    self._in_print_echo = False
                    self._in_edit_echo = False

                # Buffer continuation lines for print()
                if getattr(self, '_in_print_echo', False):
                    self._print_echo_buffer.append(line)
                    continue
                # Buffer continuation lines for edit()/line_patch()
                if getattr(self, '_in_edit_echo', False):
                    self._edit_echo_buffer.append(line)
                    continue

                # Skip emit() echo
                if not getattr(self, '_in_emit_echo', False):
                    self._show_python_header_if_pending()
                    print(line, flush=True)
                    self._capture_display_line(line)

        elif msg_type == "progress":
            text = chunk.rstrip('\n')
            for line in text.split('\n'):
                print(f"\x1b[92m{line}\x1b[0m", flush=True)  # Bright green
                self._capture_display_line(line)
        elif msg_type == "emit":
            return
        elif msg_type in ("output", "print"):
            if msg_type == "output" and getattr(self, '_suppress_next_edit_result', False):
                stripped = chunk.strip()
                if (
                    stripped in {"'Edit applied.'", "'Line patch applied.'"}
                    or re.match(r"^'All \d+ occurrences replaced\.'$", stripped)
                ):
                    self._suppress_next_edit_result = False
                    return

            # Print header if pending (in case output comes before visible echo)
            self._show_python_header_if_pending()

            text = chunk.rstrip('\n')
            # For string literals (return values), show value instead of repr
            if msg_type == "output":
                try:
                    import ast
                    value = ast.literal_eval(text)
                    if isinstance(value, str):
                        text = value.rstrip('\n')
                except (ValueError, SyntaxError):
                    pass
            # Truncate to 3 lines or 240 chars, unless this already looks like
            # preview() output with its own summary header.
            lines = text.split('\n')
            total_lines = len(lines)
            truncated_at_lines = False
            truncated_at_chars = False
            disable_truncation = (
                total_lines > 0
                and bool(__import__('re').match(r'^\(\d+ lines, \d+ chars\)$', lines[0]))
            )
            if not disable_truncation and len(lines) > 5:
                lines = lines[:5]
                truncated_at_lines = True
            display = '\n'.join(lines)
            if not disable_truncation and len(display) > 240:
                display = display[:240]
                truncated_at_chars = True
            is_truncated = truncated_at_lines or truncated_at_chars
            # For print(): show echo only if truncated or uses a variable
            is_print_output = msg_type == "print"
            print_echo_buffer = getattr(self, '_print_echo_buffer', [])
            print_uses_variable = getattr(self, '_print_uses_variable', False)
            if is_print_output and print_echo_buffer:
                if is_truncated or print_uses_variable:
                    # Show the buffered echo
                    for echo_line in print_echo_buffer:
                        print(echo_line, flush=True)
                        self._capture_display_line(echo_line)
                # Clear the buffer
                self._print_echo_buffer = []
                self._in_print_echo = False
            # Print with appropriate continuation
            if truncated_at_chars and not truncated_at_lines:
                # Cut mid-line: ellipsis on same line
                print(f"{DIM}{display}...{RESET}", flush=True)
                print(f"{DIM}({total_lines} lines total){RESET}", flush=True)
                self._capture_display_line(f"{display}...")
                self._capture_display_line(f"({total_lines} lines total)")
            elif is_truncated:
                # Cut at line boundary: ellipsis on own line
                for line in display.split('\n'):
                    print(f"{DIM}{line}{RESET}", flush=True)
                    self._capture_display_line(line)
                print(f"{DIM}... ({total_lines} lines total){RESET}", flush=True)
                self._capture_display_line(f"... ({total_lines} lines total)")
            elif is_print_output:
                # No truncation for print(): show in yellow, echo already handled
                for line in display.split('\n'):
                    print(f"\x1b[33m{line}\x1b[0m", flush=True)
                    self._capture_display_line(line)
            else:
                # No truncation for expression output: show in dim
                for line in display.split('\n'):
                    print(f"{DIM}{line}{RESET}", flush=True)
                    self._capture_display_line(line)
        elif msg_type == "file_diff":
            self._show_python_header_if_pending()
            if getattr(self, '_edit_echo_buffer', []):
                echo_line = self._compact_edit_echo(chunk)
                print(echo_line, flush=True)
                self._capture_display_line(echo_line)
                self._edit_echo_buffer = []
                self._in_edit_echo = False
            for line in chunk.rstrip('\n').split('\n'):
                if line.startswith('--- ') or line.startswith('+++ '):
                    continue
                if line.startswith('+'):
                    color = "\x1b[32m"
                elif line.startswith('-'):
                    color = "\x1b[31m"
                elif line.startswith('@@'):
                    color = "\x1b[36m"
                else:
                    color = DIM
                print(f"{color}{line}{RESET}", flush=True)
                self._capture_display_line(line)
            self._suppress_next_edit_result = True

    max_display_chars = _get_config_value("code_agent_max_display_chars", 200)  # Max chars per line to show user (agent sees full output)

    def _truncate_for_display(self, output: str) -> str:
        import re

        lines = output.split('\n')
        result_lines = []

        # Detect read() output pattern (line numbers with arrow: "   42→")
        read_pattern = re.compile(r'^\s*\d+→')

        # Process lines, detecting and truncating read() output blocks
        i = 0
        max_read_lines = 30  # Threshold for truncation
        head_tail = 10  # Lines to keep at head and tail

        while i < len(lines):
            line = lines[i]

            # Check if this starts a read() output block
            if read_pattern.match(line):
                # Find the extent of this block
                block_start = i
                while i < len(lines) and read_pattern.match(lines[i]):
                    i += 1
                block = lines[block_start:i]

                # Truncate if too long
                if len(block) > max_read_lines:
                    omitted = len(block) - 3
                    result_lines.extend(block[:3])
                    result_lines.append(f"    ... ({omitted} lines omitted for display)")
                else:
                    result_lines.extend(block)
            else:
                result_lines.append(line)
                i += 1

        # Truncate long individual lines
        truncated_lines = []
        for line in result_lines:
            if len(line) <= self.max_display_chars:
                truncated_lines.append(line)
            else:
                truncated_lines.append(line[:self.max_display_chars] + '...')

        return '\n'.join(truncated_lines)

    def on_statement_output(self, statement_chunks: list) -> None:
        """Display per-statement summary after each statement completes."""
        # Suppress display during direct user REPL mode
        if getattr(self, '_in_user_repl', False):
            return
        # Collect error output for this statement
        error_chunks = []
        for msg_type, chunk in statement_chunks:
            if msg_type == "error":
                error_chunks.append(chunk)

        error_display = "".join(error_chunks)

        # Display error output
        if error_display.strip():
            for line in error_display.rstrip('\n').split('\n'):
                print(f"\x1b[91m{line}\x1b[0m", flush=True)  # Red for errors
                self._capture_display_line(line)
            self._repl_has_output = True

        if getattr(self, '_suppress_next_edit_result', False):
            # A file_diff was displayed for a statement whose return value was
            # not shown (e.g. assignment or multi-statement exec). Clear the
            # suppression so it cannot affect later unrelated output.
            self._suppress_next_edit_result = False

    def on_repl_output(self, output_chunks: list) -> None:
        """Called at end of turn. Updates thinking message for next turn."""
        # Show thinking for next turn
        self._turn_number = getattr(self, '_turn_number', 1) + 1
        self._flush_edit_echo_buffer()
        self._turn_output_started = False  # Reset for next turn's clear_line
        thinking = getattr(self, 'thinking_message', 'Thinking...')
        self.console.clear_line()  # Clear previous status text
        print(f"{DIM}{thinking} (turn {self._turn_number}){RESET}", end="", flush=True)

    def on_retry(self, kind: str, retry_num: int) -> None:
        if kind == "syntax":
            status = f"Syntax Retry #{retry_num}... (turn {getattr(self, '_turn_number', 1)})"
        elif kind == "max_tokens":
            status = f"Max Tokens Retry #{retry_num}... (turn {getattr(self, '_turn_number', 1)})"
        else:
            return
        self.console.clear_line()  # Clear previous status text
        self._turn_output_started = False  # Reset so next output clears this status
        print(f"{DIM}{status}{RESET}", end="", flush=True)

    def user_repl_session(self, history):
        """Drop into the REPL for direct user interaction."""
        from agentlib.cli.prompt import prompt as raw_prompt
        from codeop import compile_command

        repl = self._get_tool_repl()
        self.complete = False
        transcript = []
        buffer = []
        repl_history = []  # Separate history for REPL session

        # Get altmode for history navigation (if stdout capture available)
        altmode = getattr(self, 'altmode', None)

        print(f"{DIM}Entering REPL. Ctrl+D to exit.{RESET}")

        pending_lines = []  # Lines queued from pasted input

        while True:
            prompt_str = "... " if buffer else ">>> "

            # Get next line: from pending queue or from user input
            if pending_lines:
                line = pending_lines.pop(0)
                # Echo the line since it came from paste
                print(f"{prompt_str}{line}")
            else:
                # Auto-indent for continuation lines
                auto_indent = ''
                if buffer:
                    last_line = buffer[-1]
                    indent = len(last_line) - len(last_line.lstrip(' '))
                    stripped = last_line.rstrip()
                    if stripped.endswith(':'):
                        indent += 4
                    elif stripped == '' and indent >= 4:
                        indent -= 4
                    auto_indent = ' ' * indent

                try:
                    line = raw_prompt(
                        prompt_str,
                        history=repl_history,
                        add_to_history=False,
                        altmode=altmode,
                        initial_text=auto_indent,
                    )
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print()
                    buffer = []
                    continue

                # If pasted content has multiple lines, queue them
                if '\n' in line:
                    lines = line.split('\n')
                    line = lines[0]
                    pending_lines.extend(lines[1:])

            buffer.append(line)
            source = "\n".join(buffer)

            try:
                result = compile_command(source)
                if result is not None:
                    # Complete statement - execute with tool handling
                    # Suppress on_repl_chunk display during direct REPL mode
                    self._in_user_repl = True
                    try:
                        output, _, _, _ = self._execute_with_tool_handling(repl, source)
                    except KeyboardInterrupt:
                        self._in_user_repl = False
                        print()
                        buffer = []
                        continue
                    finally:
                        self._in_user_repl = False
                    processed = self.process_repl_output(output)
                    # Strip echo for display (user already typed it)
                    display_lines = []
                    for ln in processed.split('\n'):
                        if not ln.startswith('>>> ') and not ln.startswith('... '):
                            display_lines.append(ln)
                    display = '\n'.join(display_lines).strip()
                    if display:
                        print(f"\x1b[92m{display}\x1b[0m")
                    transcript.append(processed)
                    if source.strip():
                        repl_history.append(source)
                    buffer = []
                # else: incomplete, continue accumulating
            except SyntaxError as e:
                print(f"\x1b[91mSyntaxError: {e}\x1b[0m")
                buffer = []

        if transcript:
            # Strip trailing newlines from each entry to avoid double spacing
            cleaned = [t.rstrip('\n') for t in transcript]
            self.usermsg("##### USER REPL SESSION #####\n" + "\n".join(cleaned) + "\n##### END SESSION #####")

        return bool(transcript)

    def save_session(self, filename: str):
        raise NotImplementedError("Session export has been removed; use /resume for persisted sessions.")

    def load_session(self, filename: str):
        raise NotImplementedError("Session import has been removed; use /resume for persisted sessions.")

    def resume_session(self, session_id: str):
        session = self._session_store.get_session(session_id)
        if not session:
            print(f"{DIM}Session not found: {session_id}{RESET}")
            return False
        if hasattr(self, '_conversation'):
            del self._conversation
        if hasattr(self, '_tool_repl'):
            self._tool_repl.close()
            del self._tool_repl
        if hasattr(self, '_repl_startup_injected'):
            del self._repl_startup_injected
        _ = self.conversation
        self._suspend_persistence = True
        try:
            missing = replay_session_into_agent(self, session_id, self._session_store)
            self._session_id = session_id
            self._next_event_seq = self._session_store.get_next_seq(session_id)
            self._explicit_attachment_refs = {}
            self._pending_explicit_attachment_refs = {}
            for msg in self.conversation.messages:
                for name, path in (msg.get('_attachment_refs') or {}).items():
                    self._explicit_attachment_refs[name] = path
            self._replay_display_output()
            self.usermsg(">>> system_reset()\nREPL session has been reset\n")
            if missing:
                lines = ["[Resume warning: attachment file missing and detached]"]
                lines.extend(f"- {name}: {path}" for name, path in missing)
                self.usermsg("\n".join(lines))
        finally:
            self._suspend_persistence = False
        print(f"{DIM}Session resumed: {session_id}{RESET}")
        return True

    def _quiet_replay_session(self):
        """Re-replay the persisted session into this agent without UI noise."""
        if not self._session_id:
            return
        if hasattr(self, '_conversation'):
            del self._conversation
        _ = self.conversation
        self._suspend_persistence = True
        try:
            replay_session_into_agent(self, self._session_id, self._session_store)
            self._next_event_seq = self._session_store.get_next_seq(self._session_id)
            self._explicit_attachment_refs = {}
            self._pending_explicit_attachment_refs = {}
            for msg in self.conversation.messages:
                for name, path in (msg.get('_attachment_refs') or {}).items():
                    self._explicit_attachment_refs[name] = path
        finally:
            self._suspend_persistence = False

    def _synthetic_exchange(self):
        self._ensure_setup()
        repl = self._get_tool_repl()
        repl._inject_code("from datetime import date")
        today = __import__('datetime').date.today().isoformat()
        parsed_today = __import__('datetime').date.fromisoformat(today) if today else None
        formatted_today = (
            f"{parsed_today.strftime('%A, %B')} {parsed_today.day}, {parsed_today.year}"
            if parsed_today else today
        )
        assistant_probe = 'from datetime import date\nemit("Checking today\'s date...")\nprint(date.today().isoformat())'
        assistant_emit = f'emit("Today is {formatted_today}.", release=True)'
        user_probe = f"What's today's date?\n"
        user_probe_output = (
            user_probe
            + '>>> emit("Checking today\'s date...")\n'
            + "Checking today's date...\n"
            + f">>> print(date.today().isoformat())\n"
            + f"{today}\n"
        )
        user_emit_output = (
            f">>> {assistant_emit}\n"
            f"Today is {formatted_today}.\n"
        )
        for role, content in (
            ('user', user_probe),
            ('assistant', assistant_probe),
            ('user', user_probe_output),
            ('assistant', assistant_emit),
            ('user', user_emit_output),
        ):
            self.conversation.messages.append({"role": role, "content": content, "_synthetic": True})
        self._last_was_repl_output = True

    def cli_run(self, max_turns: int | None = None, resume: str | bool = False):
        """Run CLI loop with Python block delimiters."""
        from agentlib.cli.mixin import SQLiteHistory, InputSession
        from agentlib.cli.altmode import AltMode

        self._ensure_setup()

        if max_turns is None:
            max_turns = getattr(self, 'max_turns', 10)

        # Set up stdout capture for alt-buffer replay
        altmode = AltMode()
        altmode.install()
        self.altmode = altmode  # Make available to user_repl_session

        # Set up history
        history_path = getattr(self, 'history_db', None)
        history = SQLiteHistory(history_path)
        session = InputSession(history, altmode=altmode)

        # Display welcome banner with model and sandbox info
        welcome = getattr(self, 'welcome_message', '')
        if welcome:
            # Add model and sandbox status
            # Resolve alias to full name for display
            from agentlib.llm_registry import resolve_model_name
            full_model_name = resolve_model_name(self.model)
            model_name = full_model_name.split('/')[-1] if '/' in full_model_name else full_model_name
            sandbox_status = "[green]sandbox[/green]" if SandboxMixin in type(self).__mro__ else "[dim]no sandbox[/dim]"
            banner = f"{welcome}\n[dim]{model_name}[/dim] · {sandbox_status}"
            self.console.print(Panel.fit(banner, border_style="cyan"))

        prompt_str = getattr(self, 'cli_prompt', '> ')
        thinking = getattr(self, 'thinking_message', 'Thinking...')

        self.console.print("[dim]Enter = submit | Alt+Enter = newline | Ctrl+O = transcript | Esc Esc = rewind | Ctrl+C = interrupt | Ctrl+D = quit[/dim]")
        self.console.print("[dim]Commands: /repl, /rewind, /resume [session_id], /skills [name], /subagents [model], /attach <file>, /detach <file>, /attachments, /model [name], /tokens[/dim]")

        resumed_on_start = False
        if resume:
            if resume is True:
                from agentlib.cli.sessions import select_session_ui
                session_id = select_session_ui(altmode, self._session_store, str(Path.cwd()))
            else:
                session_id = resume
            if session_id:
                resumed_on_start = self.resume_session(session_id)

        if not resumed_on_start:
            if files := gather_auto_attach_files():
                self._display_text(f"Loading {', '.join(files)}", kind="status", create_session=False)
                self._auto_context_attachment_names.update(files)
                for filename in files:
                    self.attach_file_ref(filename, filename)
        synth = not resumed_on_start and not bool(resume)

        try:
            preload_input = ""
            while True:
                rewind_shortcut = False

                def open_transcript(_buffer: str, _cursor: int):
                    from agentlib.cli.transcript import transcript_viewer_ui
                    self._ensure_live_session()
                    self._flush_pending_session_events()
                    events = self._session_store.get_events(self._session_id) if self._session_id else []
                    transcript_viewer_ui(altmode, events)

                def trigger_rewind():
                    nonlocal rewind_shortcut
                    rewind_shortcut = True
                    return "/rewind"

                try:
                    user_input = session.prompt(
                        f"\n{prompt_str}",
                        initial_text=preload_input,
                        on_ctrl_o=open_transcript,
                        on_esc_esc=trigger_rewind,
                    )
                except KeyboardInterrupt:
                    print()
                    preload_input = ""
                    continue
                except EOFError:
                    if not self._run_pre_exit_hooks():
                        self.console.print("[yellow]Returning to prompt. Try Ctrl+D again to exit.[/yellow]")
                        continue
                    break
                preload_input = ""

                if not user_input.strip():
                    continue

                self._ensure_live_session()
                self._flush_pending_session_events()
                if user_input.strip() == "/repl":
                    self._display_input_block(user_input)
                    try:
                        self.user_repl_session(history)
                    except Exception as e:
                        print(f"\n{DIM}Error: {type(e).__name__}: {e}{RESET}", file=sys.stderr)
                    continue

                if user_input.strip() == "/rewind":
                    if not rewind_shortcut:
                        self._display_input_block(user_input)
                    from agentlib.cli.rewind import rewind_ui
                    self._ensure_live_session()
                    self._flush_pending_session_events()
                    events = self._session_store.get_events(self._session_id) if self._session_id else []
                    rewind_result = rewind_ui(altmode, events)
                    if rewind_result is not None:
                        if rewind_shortcut:
                            sys.stdout.write("\x1b[1A\r\x1b[K")
                            sys.stdout.flush()
                        target_seq = rewind_result.get("target_seq")
                        if target_seq is not None:
                            self._append_session_event("rewind", {"target_seq": target_seq})
                        self._quiet_replay_session()
                        self._replay_display_output()
                        self._display_text(f"{DIM}Conversation rewound.{RESET}", kind="status")
                        last = self.conversation.messages[-1] if self.conversation.messages else None
                        self._last_was_repl_output = bool(last and last.get('role') == 'user')
                        preload_input = rewind_result.get("preload_input", "") or ""
                    elif rewind_shortcut:
                        sys.stdout.write("\x1b[1A\r\x1b[K")
                        sys.stdout.flush()
                    continue

                if user_input.strip().startswith("/resume"):
                    self._display_input_block(user_input)
                    resumed = False
                    parts = user_input.strip().split(None, 1)
                    if len(parts) == 1:
                        from agentlib.cli.sessions import select_session_ui
                        session_id = select_session_ui(altmode, self._session_store, str(Path.cwd()))
                        if session_id:
                            resumed = self.resume_session(session_id)
                    else:
                        resumed = self.resume_session(parts[1].strip())
                    if resumed:
                        synth = False
                    continue

                if user_input.strip().startswith("/skills"):
                    self._display_input_block(user_input)
                    parts = user_input.strip().split(None, 1)
                    if len(parts) == 1:
                        from agentlib.cli.skills import select_skills_ui
                        skill_items = self.list_skills()
                        result = select_skills_ui(altmode, skill_items)
                        if result is not None:
                            changes = self.apply_skill_selection(result)
                            if changes:
                                for line in changes:
                                    self._display_text(f"{DIM}{line}{RESET}", kind="status")
                            else:
                                self._display_text(f"{DIM}No skill changes{RESET}", kind="status")
                    else:
                        ok, msg = self.attach_skill(parts[1].strip())
                        self._display_text(f"{DIM}{msg}{RESET}", kind="status")
                    continue

                if user_input.strip().startswith("/attach "):
                    self._display_input_block(user_input)
                    filename = user_input.strip()[8:].strip()
                    if filename:
                        try:
                            content = Path(filename).expanduser().read_text()
                            self.attach_file_ref(filename, filename)
                            size_kb = len(content) / 1000
                            self._display_text(f"{DIM}Attached {filename} ({size_kb:.1f}KB){RESET}", kind="status")
                        except Exception as e:
                            self._display_text(f"{DIM}Error attaching {filename}: {e}{RESET}", kind="status")
                    continue

                if user_input.strip().startswith("/detach "):
                    self._display_input_block(user_input)
                    filename = user_input.strip()[8:].strip()
                    if filename:
                        self.detach_file_ref(filename)
                        self._display_text(f"{DIM}Detached {filename}{RESET}", kind="status")
                    continue

                if user_input.strip() == "/attachments":
                    self._display_input_block(user_input)
                    attachments = self.list_attachments()
                    if not attachments:
                        self._display_text(f"{DIM}No attachments{RESET}", kind="status")
                    else:
                        self._display_text(f"{DIM}Current attachments:{RESET}", kind="status")
                        for name, content in attachments.items():
                            size_kb = len(content) / 1000
                            self._display_text(f"{DIM}  {name} ({size_kb:.1f}KB){RESET}", kind="status")
                    continue

                if user_input.strip().startswith("/model"):
                    self._display_input_block(user_input)
                    parts = user_input.strip().split(None, 1)
                    if len(parts) == 1:
                        # No argument - show current model
                        from agentlib.llm_registry import resolve_model_name
                        full_name = resolve_model_name(self.model)
                        if full_name != self.model:
                            self._display_text(f"{DIM}Current model: {self.model} → {full_name}{RESET}", kind="status")
                        else:
                            self._display_text(f"{DIM}Current model: {self.model}{RESET}", kind="status")
                    else:
                        # Set new model
                        new_model = parts[1].strip()
                        old_model = self.model
                        try:
                            # Validate the model exists by trying to get its config
                            from agentlib.llm_registry import get_model_config
                            get_model_config(new_model)
                            self.model = new_model
                            # Clear cached client so new model takes effect
                            if hasattr(self, '_llm_client'):
                                delattr(self, '_llm_client')
                            # Update conversation's client reference (preserves message history)
                            if hasattr(self, '_conversation'):
                                self._conversation.llm_client = self.llm_client
                            self._display_text(f"{DIM}Model changed: {old_model} → {new_model}{RESET}", kind="status")
                        except ModelNotFoundError as e:
                            self._display_text(f"{DIM}{str(e)}{RESET}", kind="status")
                    continue

                if user_input.strip() == "/tokens":
                    self._display_input_block(user_input)
                    tracker = self.llm_client.usage_tracker
                    if not tracker.history:
                        self._display_text(f"{DIM}No API calls yet{RESET}", kind="status")
                    else:
                        n = tracker._normalize(*tracker.history[-1])
                        total = n['prompt_tokens'] + n['cached_tokens'] + n['completion_tokens'] + n['reasoning_tokens']
                        parts = [p for p in [
                            f"{n['prompt_tokens']:,} in" if n['prompt_tokens'] else None,
                            f"{n['cached_tokens']:,} cached" if n['cached_tokens'] else None,
                            f"{n['reasoning_tokens']:,} reasoning" if n['reasoning_tokens'] else None,
                            f"{n['completion_tokens']:,} out" if n['completion_tokens'] else None,
                        ] if p]
                        self._display_text(f"{DIM}[Last request: {total:,} tokens ({', '.join(parts)})]{RESET}", kind="status")
                    continue

                if user_input.strip().startswith("/subagents"):
                    self._display_input_block(user_input)
                    try:
                        # Import subagent module into REPL and show docstring to agent
                        # Optional model parameter: /subagents [model]
                        parts = user_input.strip().split(None, 1)
                        if len(parts) > 1:
                            # Model specified administratively
                            subagent_model = parts[1].strip()
                            model_locked = True
                        else:
                            # Inherit parent's model
                            subagent_model = self.model
                            model_locked = False

                        repl = self._get_tool_repl()
                        # Import and set default model (silent injection)
                        repl._inject_code(f"from agentlib.subagent import Subagent, SubagentError, SubagentResponse, _subagents; Subagent.default_model = {repr(subagent_model)}")

                        # Only show docstring on first load; subsequent calls just update model
                        already_loaded = getattr(self, '_subagents_loaded', False)
                        if already_loaded:
                            self._display_text(f"{DIM}Subagent default model changed to: {subagent_model}{RESET}", kind="status")
                        else:
                            self._subagents_loaded = True
                            # Build docstring, optionally hiding model config section
                            from agentlib import subagent
                            docstring = subagent.__doc__
                            if model_locked:
                                # Strip "## Model Configuration" section so agent doesn't try to override
                                import re
                                docstring = re.sub(r'## Model Configuration\n.*?(?=\n## |\n"""|\Z)', '', docstring, flags=re.DOTALL)
                            self.usermsg(f">>> # Subagent module loaded (model: {subagent_model})\n{docstring}")
                            self._display_text(f"{DIM}Subagent module loaded into REPL (model: {subagent_model}){RESET}", kind="status")
                    except Exception as e:
                        print(f"\n{DIM}Error: {type(e).__name__}: {e}{RESET}", file=sys.stderr)
                    continue

                self._display_input_block(user_input)

                if synth:
                    try:
                        self._synthetic_exchange()
                    except Exception as e:
                        print(f"\n{DIM}Error: {type(e).__name__}: {e}{RESET}", file=sys.stderr)
                    synth = False

                self.usermsg(user_input, _user_content=user_input)

                # Reset state for new user interaction
                self._repl_printed_header = False
                self._repl_has_output = False
                self._turn_number = 1
                self._turn_output_started = False
                self._header_pending = False
                self._in_emit_echo = False
                self._reset_display_capture()
                print()  # Blank line after user input
                print(f"{DIM}{thinking} (turn 1){RESET}", end="", flush=True)

                try:
                    response = self.run_loop(max_turns=max_turns)
                except KeyboardInterrupt:
                    self.console.clear_line()
                    print()
                    continue
                except Exception as e:
                    self.console.clear_line()
                    print(f"\n{DIM}Error: {type(e).__name__}: {e}{RESET}", file=sys.stderr)
                    continue

                self.console.clear_line()  # Clear thinking message

                # Close Python block if we had output
                if getattr(self, '_repl_printed_header', False):
                    print("\x1b[34m"+("─"*34)+"\x1b[0m")
                    self._capture_display_line("──────────────────────────────────")
                self._flush_display_capture()

                # Display response
                response_str = str(response) if response is not None else ""
                formatted = self.format_response(response_str)
                if formatted:
                    print(formatted)
        finally:
            altmode.uninstall()
            # Save conversation on crash
            if sys.exc_info()[1] is not None:
                import tempfile
                crash_file = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.json', prefix='repl_crash_', delete=False
                )
                json.dump(self.conversation._messages(), crash_file, indent=2)
                crash_file.close()
                print(f"\n*** Conversation saved to: {crash_file.name} ***", file=sys.stderr)
            # Clean up temp files from truncated output
            for path in getattr(self, '_temp_files', []):
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self.console.print("\n[dim]Session ended. Goodbye![/dim]")
            session_id = getattr(self, "_session_id", None)
            if session_id:
                self.console.print(f"[dim]Resume session: code-agent --resume {session_id}[/dim]")


class CodeAgent(JinaMixin, MCPMixin, CodeAgentBase):
    """Code agent with native tools.

    Inherits web_fetch and web_search from JinaMixin.
    """

    mcp_servers = []

    _preview_counter = 0  # Shared counter for _vN variable names

    @REPLAgent.tool(inject=True)
    def think(self, content: str = "All relevant observations and reasoning"):
        """Think through the problem and yield to a new turn.

        Call this when you're uncertain how to proceed or need to reason
        through a problem. Write down your observations, hypotheses,
        open questions, and options you're considering.
        """
        return "[Continuing...]"

    @REPLAgent.tool(inject=True)
    def preview(self, value: object = "Value to preview"):
        """Print a value with head/tail summary for long values.

        Non-strings are previewed via repr(value). Short values are printed in
        full. Long values print first and last few lines with omitted counts.
        """
        if not isinstance(value, str):
            value = repr(value)

        lines = value.split('\n')
        nlines = len(lines)
        nchars = len(value)
        uri = None

        # Short enough to show in full
        if nlines <= 20 and nchars <= 2000:
            rendered = value
        else:
            HEAD = 8
            TAIL = 4
            omitted = nlines - HEAD - TAIL
            key = __import__("hashlib").sha256(value.encode("utf-8")).hexdigest()[:16]
            uri = f"session://preview/{key}"
            import json as _json
            global _request_id
            _request_id += 1
            _req_id = _request_id
            _send_tool_request(_json.dumps({
                "tool": "__preview_blob_save__",
                "args": {"key": key, "content": value},
                "request_id": _req_id,
            }))
            _wait_for_ack(_req_id)
            parts = [f"({nlines} lines, {nchars} chars)"]
            parts.extend(lines[:HEAD])
            parts.append(f"  ... ({omitted} lines omitted)")
            parts.extend(lines[-TAIL:])
            parts.append(f"[full output saved to {uri}]")
            rendered = '\n'.join(parts)

        _send_output("preview", rendered.rstrip('\n') + "\n")


    # Target tool names whose bare expressions get rewritten to assignment + preview
    _preview_targets = frozenset({'grep', 'bash', 'read'})

    def preprocess_code(self, code: str) -> str:
        """Apply base preprocessing then rewrite bare tool calls to assignment + preview."""
        if getattr(self, '_in_user_repl', False):
            return code

        code = super().preprocess_code(code)

        code, self._preview_counter = preprocess_code_agent(
            code,
            preview_targets=self._preview_targets,
            preview_counter=getattr(self, '_preview_counter', 0),
        )
        return code

    def _handle_tool_request(self, repl, req: dict) -> None:
        tool_name = req.get('tool')
        if tool_name in {'__preview_blob_save__', '__preview_blob_read__', '__line_patch_is_attached__'}:
            request_id = req.get('request_id')
            args = req.get('args', {})
            try:
                if tool_name == '__line_patch_is_attached__':
                    repl.send_reply(request_id, result=self._is_attached(args.get('path')))
                    repl.send_ack(request_id)
                    return
                if getattr(self, '_session_id', None) is None:
                    self._ensure_live_session()
                    self._flush_pending_session_events()
                if tool_name == '__preview_blob_save__':
                    self._session_store.save_preview_blob(self._session_id, args.get('key'), args.get('content', ''))
                    repl.send_reply(request_id, result=True)
                else:
                    repl.send_reply(request_id, result=self._session_store.get_preview_blob(self._session_id, args.get('key')))
            except Exception as e:
                repl.send_reply(request_id, error=str(e))
            finally:
                repl.send_ack(request_id)
            return
        return super()._handle_tool_request(repl, req)


    @REPLAgent.tool(inject=True)
    def grep(self,
            pattern: str = "Regex pattern to search for",
            path: Optional[str] = "File or directory to search in",
            glob: Optional[str] = "Glob pattern to filter files (e.g., '*.js')",
            file_type: Optional[str] = "File type to search (e.g., 'py', 'js', 'rust')",
            files_only: Optional[bool] = "Only return filenames, not matching lines",
            context: Optional[int] = "Lines of context around matches (-C)",
            case_insensitive: Optional[bool] = "Case insensitive search (-i)",
            multiline: Optional[bool] = "Enable multiline matching (-U)"
        ):
        """Search file contents with ripgrep."""
        import subprocess
        cmd = ['rg', '--color=never', '-n']  # Always show line numbers

        if files_only:
            cmd.append('-l')
        if case_insensitive:
            cmd.append('-i')
        if multiline:
            cmd.append('-U')
        if context:
            cmd.extend(['-C', str(context)])
        if glob:
            cmd.extend(['--glob', glob])
        if file_type:
            cmd.extend(['--type', file_type])

        cmd.append(pattern)
        if path:
            cmd.append(path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout.strip()

        if not output:
            return "No matches found"
        if files_only:
            files = output.split('\n')
            return files
        return output

    @REPLAgent.tool(inject=True)
    def read(self,
            file_path: str = "Path to the file",
            offset: Optional[int] = "Line number to start from (1-indexed)",
            limit: Optional[int] = "Number of lines to read (default: 5000)"
        ):
        """Read a file or session://preview/... URI and return its contents as text.

        Use read() when you want contents as a Python value:
            content = read("file.py")
            lines = read("file.py").splitlines()
            snippet = read("file.py", offset=100, limit=20)

        Long preview() output is saved to a session://preview/... URI:
            full_output = read("session://preview/abc123")

        Use view() when you want numbered file output:
            view("file.py")
            view("file.py", offset=100, limit=20)
            view("session://preview/abc123", offset=100, limit=20)
        """
        prefix = "session://preview/"
        if isinstance(file_path, str) and file_path.startswith(prefix):
            key = file_path[len(prefix):]
            import json as _json
            global _request_id
            _request_id += 1
            _req_id = _request_id
            _send_tool_request(_json.dumps({
                "tool": "__preview_blob_read__",
                "args": {"key": key},
                "request_id": _req_id,
            }))
            content = _wait_for_ack(_req_id)
            if content is None:
                raise FileNotFoundError(file_path)
        else:
            content = Path(file_path).expanduser().read_text()
        if offset is None and limit is None:
            return content
        all_lines = content.split('\n')
        start = (offset or 1) - 1
        end = start + (limit or 5000)
        return '\n'.join(all_lines[start:end])

    @REPLAgent.tool(inject=True)
    def view(self,
            file_path: str = "Path to the file",
            offset: Optional[int] = "Line number to start from (1-indexed)",
            limit: Optional[int] = "Number of lines to read (default: 5000)",
            **kwargs
        ):
        """Display a file or session://preview/... URI with line numbers.

        Use view() for inspection with numbered lines:
            view("file.py")
            view("file.py", offset=100, limit=20)
            view("session://preview/abc123", offset=100, limit=20)

        Preview URI reads are for inspecting saved preview() output and are
        not filesystem paths.

        WRONG — view() is not a value:
            content = view("file.py")
            print(view("file.py"))
            preview(view("file.py"))

        Use read() if you need contents as text:
            content = read("file.py")
            full_output = read("session://preview/abc123")
        """
        unexpected_kwargs = set(kwargs) - {"_force_partial"}
        if unexpected_kwargs:
            unexpected = ", ".join(sorted(unexpected_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
        force_partial = bool(kwargs.get("_force_partial", False))

        prefix = "session://preview/"
        is_preview_uri = isinstance(file_path, str) and file_path.startswith(prefix)
        path = Path(file_path).expanduser()
        if is_preview_uri:
            key = file_path[len(prefix):]
            import json as _json
            global _request_id
            _request_id += 1
            _req_id = _request_id
            _send_tool_request(_json.dumps({
                "tool": "__preview_blob_read__",
                "args": {"key": key},
                "request_id": _req_id,
            }))
            content = _wait_for_ack(_req_id)
            if content is None:
                raise FileNotFoundError(file_path)
        else:
            content = path.read_text()
        all_lines = content.split('\n')
        total_lines = len(all_lines)
        is_partial = offset is not None or limit is not None
        source_extensions = {
            ".py", ".pyi", ".pyx", ".pxd", ".pxi",
            ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts",
            ".html", ".htm", ".xhtml", ".css", ".scss", ".sass", ".less",
            ".vue", ".svelte", ".astro",
            ".java", ".kt", ".kts", ".scala", ".groovy",
            ".c", ".h", ".cc", ".hh", ".cpp", ".cxx", ".hpp", ".hxx",
            ".cs", ".go", ".rs", ".swift", ".m", ".mm",
            ".php", ".rb", ".rake", ".pl", ".pm", ".t", ".lua",
            ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
            ".sql", ".graphql", ".gql",
            ".xml", ".xsl", ".xslt", ".svg",
            ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
            ".md", ".mdx", ".rst", ".tex",
            ".dockerfile", ".containerfile",
            ".vim", ".el", ".clj", ".cljs", ".cljc", ".ex", ".exs", ".erl", ".hrl",
            ".fs", ".fsx", ".fsi", ".hs", ".lhs", ".ml", ".mli", ".nim", ".zig",
            ".r", ".R", ".jl", ".dart", ".sol", ".tf", ".tfvars", ".hcl",
        }
        source_names = {
            "Dockerfile", "Containerfile", "Makefile", "Rakefile", "Gemfile",
            "Podfile", "Brewfile", "Justfile", "Taskfile", "Jenkinsfile",
            "BUILD", "WORKSPACE", "CMakeLists.txt",
        }
        if not is_preview_uri and is_partial and not force_partial and (path.suffix in source_extensions or path.name in source_names):
            raise ValueError(
                "Partial view denied for source file. Prefer one full "
                "view(file_path) when inspecting normal source files you may "
                "need to reason about or edit across turns; use partial views only "
                "for genuinely huge, generated, vendored, or minified files, or "
                "for narrow line-number checks after the full file is already in "
                "context. To override this denial when a partial view is truly "
                "appropriate, call view(file_path, offset=..., limit=..., "
                "_force_partial=True)."
            )

        start = (offset or 1) - 1
        end = start + (limit or 5000)
        lines = all_lines[start:end]
        start_line = start + 1

        formatted = [f"{start_line + i:>5}→{line}" for i, line in enumerate(lines)]
        output = '\n'.join(formatted)

        remaining = total_lines - end
        if remaining > 0 and limit is None:
            output += f"\n... ({remaining} more lines)"

        if offset is None and limit is None:
            if not is_preview_uri:
                import hashlib
                snapshots = globals().setdefault("_line_patch_snapshots", {})
                snapshots[file_path] = {
                    "path": file_path,
                    "resolved_path": str(path.resolve()),
                    "content": content,
                    "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "line_count": len(all_lines),
                    "line_patch_stale": False,
                }
            _send_output("read_attach", file_path + "\n")
        else:
            _send_output("read_partial", file_path + "\n")

        _send_output("read", output + "\n")

    @REPLAgent.tool
    def unview(self,
            file_path: str = "Path to a file previously viewed with view()"
        ):
        """Remove a previously viewed file from future context.

        Use this if you viewed the wrong file with view() or no longer
        need it in context. This only affects future turns.
        """
        attachments = self.list_attachments(include_session_blobs=True)
        explicit_refs = getattr(self, '_explicit_attachment_refs', {})
        if file_path in explicit_refs:
            self.detach_file_ref(file_path)
        elif file_path in attachments:
            self.detach(file_path)
        globals().get("_line_patch_snapshots", {}).pop(file_path, None)
        self._pending_unviewed_files.add(file_path)
        return f"Removed from future context: {file_path}"

    @REPLAgent.tool(inject=True)
    def edit(self,
            file_path: str = "Path to the file",
            old_string: str = "Text to replace (must be unique unless replace_all)",
            new_string: str = "Replacement text",
            replace_all: Optional[bool] = "Replace all occurrences"
        ):
        """Edit a file by replacing text."""
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError("File does not exist.")
        if old_string == new_string:
            raise ValueError("No changes to make: old_string and new_string are exactly the same.")

        content = path.read_text()
        count = content.count(old_string)

        if count == 0:
            raise ValueError(f"String to replace not found in file.")
        if count > 1 and not replace_all:
            raise ValueError(f"Found {count} matches of the string to replace, but replace_all is false.")

        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)
        path.write_text(new_content)
        import difflib
        diff = ''.join(difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=file_path,
            tofile=file_path,
        ))
        if diff:
            _send_output("file_diff", diff.rstrip('\n') + "\n")
        _send_output("file_written", str(path.resolve()) + "\n")

        if replace_all and count > 1:
            return f"All {count} occurrences replaced."
        return "Edit applied."

    @REPLAgent.tool(inject=True)
    def line_patch(self,
            file_path: str = "Path to an existing file",
            patch: str = "Line patch operations"
        ):
        """Edit an existing file by line number.

        Prefer a full view(file_path) first. If no current view snapshot exists,
        line_patch uses the file's current on-disk contents as the line-number
        baseline and attaches the edited file for future context.
        Multiple operations in one call are applied atomically.

        You may call line_patch() repeatedly after one full view(file_path).

        Operations:
          replace START:END
          delete START:END
          insert before LINE
          insert after LINE

        Operation headers must start at column 1 in the patch string. Body lines
        are literal file content and continue until the next operation header; do
        not indent body lines unless the file should contain that indentation.

        Use Python file APIs such as Path.write_text(), Path.rename(), or
        Path.unlink() for create, move, and delete.
        """
        import hashlib
        import json as _json
        import re

        if isinstance(file_path, str) and file_path.startswith("session://"):
            raise ValueError("line_patch cannot edit session:// preview URIs.")

        global _request_id
        _request_id += 1
        _req_id = _request_id
        _send_tool_request(_json.dumps({
            "tool": "__line_patch_is_attached__",
            "args": {"path": file_path},
            "request_id": _req_id,
        }))
        was_attached = bool(_wait_for_ack(_req_id))

        path = Path(file_path).expanduser()
        old_text = path.read_text()
        old_hash = hashlib.sha256(old_text.encode()).hexdigest()

        snapshots = globals().setdefault("_line_patch_snapshots", {})
        snapshot = snapshots.get(file_path)
        if snapshot is None:
            all_lines = old_text.split('\n')
            snapshot = {
                "path": file_path,
                "resolved_path": str(path.resolve()),
                "content": old_text,
                "sha256": old_hash,
                "line_count": len(all_lines),
                "line_patch_stale": False,
            }
            snapshots[file_path] = snapshot
            was_attached = False
        if snapshot.get("line_patch_stale"):
            raise ValueError(f"Call view({file_path!r}) before using line_patch().")
        if old_hash != snapshot.get("sha256"):
            if not was_attached:
                snapshot.update({
                    "content": old_text,
                    "sha256": old_hash,
                    "line_count": len(old_text.split('\n')),
                    "line_patch_stale": False,
                })
            else:
                raise ValueError(f"{file_path} changed on disk since it was viewed. Call view({file_path!r}) again before line_patch().")

        header_re = re.compile(r'^(replace) (\d+):(\d+)$|^(delete) (\d+):(\d+)$|^(insert) (before|after) (\d+)$')
        raw_lines = patch.split('\n')
        ops = []
        current = None

        def finish_current():
            if current is not None:
                ops.append(current)

        for raw in raw_lines:
            match = header_re.match(raw)
            if match:
                finish_current()
                if match.group(1):
                    current = {
                        "kind": "replace",
                        "start": int(match.group(2)),
                        "end": int(match.group(3)),
                        "body": [],
                        "header": raw,
                    }
                elif match.group(4):
                    current = {
                        "kind": "delete",
                        "start": int(match.group(5)),
                        "end": int(match.group(6)),
                        "body": [],
                        "header": raw,
                    }
                else:
                    current = {
                        "kind": "insert",
                        "where": match.group(8),
                        "line": int(match.group(9)),
                        "body": [],
                        "header": raw,
                    }
                continue
            if current is None:
                if raw.strip():
                    raise ValueError(f"Expected line_patch operation header, got: {raw!r}")
                continue
            current["body"].append(raw)
        finish_current()

        if raw_lines and raw_lines[-1] == "":
            for op in reversed(ops):
                if op.get("body") and op["body"][-1] == "":
                    op["body"].pop()
                    break

        if not ops:
            raise ValueError("line_patch has no operations.")

        lines = old_text.split('\n')
        line_count = len(lines)
        range_ops = []
        insertion_points = set()

        for op in ops:
            kind = op["kind"]
            if kind in {"replace", "delete"}:
                start = op["start"]
                end = op["end"]
                if start < 1 or end < start or end > line_count:
                    raise ValueError(f"Invalid range {start}:{end} for {file_path} with {line_count} lines.")
                range_ops.append((start, end, op))
                if kind == "delete" and any(line.strip() for line in op["body"]):
                    raise ValueError(f"{op['header']} does not accept a body.")
            else:
                line = op["line"]
                where = op["where"]
                if where == "before":
                    valid = 1 <= line <= line_count + 1
                    point = ("before", line)
                else:
                    valid = 0 <= line <= line_count
                    point = ("after", line)
                if not valid:
                    raise ValueError(f"Invalid insert {where} {line} for {file_path} with {line_count} lines.")
                if point in insertion_points:
                    raise ValueError(f"Duplicate line_patch insertion point: insert {where} {line}.")
                insertion_points.add(point)
                if not op["body"]:
                    raise ValueError(f"insert {where} {line} has no body.")

        for i, (start_a, end_a, op_a) in enumerate(range_ops):
            for start_b, end_b, op_b in range_ops[i + 1:]:
                if start_a <= end_b and start_b <= end_a:
                    raise ValueError(f"Overlapping line_patch operations: {op_a['header']} conflicts with {op_b['header']}.")
        for op in ops:
            if op["kind"] != "insert":
                continue
            line = op["line"]
            for start, end, range_op in range_ops:
                inside = (
                    (op["where"] == "before" and start <= line <= end)
                    or (op["where"] == "after" and start <= line < end)
                )
                if inside:
                    raise ValueError(f"Overlapping line_patch operations: {op['header']} conflicts with {range_op['header']}.")

        by_start = {start: op for start, end, op in range_ops}
        by_before = {}
        by_after = {}
        for op in ops:
            if op["kind"] == "insert":
                if op["where"] == "before":
                    by_before[op["line"]] = op
                else:
                    by_after[op["line"]] = op

        new_lines = []
        if 0 in by_after:
            new_lines.extend(by_after[0]["body"])

        i = 1
        while i <= line_count:
            if i in by_before:
                new_lines.extend(by_before[i]["body"])
            replace_op = by_start.get(i)
            if replace_op is not None:
                if replace_op["kind"] == "replace":
                    new_lines.extend(replace_op["body"])
                i = replace_op["end"] + 1
                if i - 1 in by_after:
                    new_lines.extend(by_after[i - 1]["body"])
                continue
            new_lines.append(lines[i - 1])
            if i in by_after:
                new_lines.extend(by_after[i]["body"])
            i += 1

        if line_count + 1 in by_before:
            new_lines.extend(by_before[line_count + 1]["body"])

        new_text = '\n'.join(new_lines)
        if new_text == old_text:
            raise ValueError("line_patch produced no changes.")

        path.write_text(new_text)
        snapshot.update({
            "content": new_text,
            "sha256": hashlib.sha256(new_text.encode()).hexdigest(),
            "line_count": len(new_text.split('\n')),
            "line_patch_stale": False,
        })
        import difflib
        diff = ''.join(difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=file_path,
            tofile=file_path,
        ))
        if diff:
            _send_output("file_diff", diff.rstrip('\n') + "\n")
        _send_output("file_written", str(path.resolve()) + "\n")
        if not was_attached:
            lines = new_text.split('\n')
            formatted = '\n'.join(f"{i+1:>5}→{line}" for i, line in enumerate(lines))
            _send_output("read_attach", file_path + "\n")
            _send_output("read", formatted + "\n")
        return "Line patch applied."

    @REPLAgent.tool(inject=True)
    def bash(self,
            command: str = "The command to execute",
            timeout: Optional[int] = "Timeout in seconds (default: 120)",
            bg: bool = "Run in background (returns BashProcess object)"
        ):
        """Execute a bash command.

        Returns string output if successful within timeout.
        Returns BashProcess object if bg=True or if command times out.
        
        The BashProcess object allows interaction (read/write/kill) and 
        is automatically registered in global `_bash_procs` for recovery.
        """
        import subprocess
        import os
        import fcntl
        import signal
        import time

        # Initialize global registry and atexit handler if missing
        if '_bash_procs' not in globals():
            globals()['_bash_procs'] = {}

            # Register atexit handler to kill orphaned processes on Python exit
            import atexit
            def _cleanup_bash_procs():
                for pid, bp in list(globals().get('_bash_procs', {}).items()):
                    if bp.returncode is None:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
            atexit.register(_cleanup_bash_procs)

        class BashProcess:
            def __init__(self, popen, command, timeout=None):
                self.proc = popen
                self.command = command
                self.pid = popen.pid
                self.timeout = timeout  # None or number
                self._output = ""
                self._register()
                self._set_nonblocking(self.proc.stdout)
            
            def _register(self):
                globals()['_bash_procs'][self.pid] = self

            def _set_nonblocking(self, f):
                if f:
                    fd = f.fileno()
                    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

            def read(self, timeout=-1):
                """Read output.

                Blocks until process completes or timeout expires.
                Default: uses self.timeout if set, else 120s.
                """
                if timeout == -1:
                    timeout = self.timeout if self.timeout is not None else 120
                output = []
                start = time.time()
                
                # Helper to read all available
                def _read_chunk():
                    found_any = False
                    while True:
                        try:
                            chunk = self.proc.stdout.read(4096)
                            if not chunk:
                                break
                            output.append(chunk)
                            found_any = True
                        except Exception:
                            break
                    return found_any

                if timeout is None:
                    _read_chunk()
                else:
                    # With timeout: loop until done or timeout
                    while True:
                        _read_chunk()

                        if self.proc.poll() is not None:
                            # Finished, ensure we get everything
                            while _read_chunk(): pass
                            break

                        if time.time() - start > timeout:
                            break

                        time.sleep(0.01)

                # Prepend any buffered output, then clear buffer
                new_output = b"".join(output).decode('utf-8', errors='replace')
                if self._output:
                    result = self._output + new_output
                    self._output = ""
                    return result
                return new_output

            def write(self, text):
                """Write text to stdin."""
                if self.proc.stdin:
                    self.proc.stdin.write(text.encode())
                    self.proc.stdin.flush()
            
            def kill(self):
                """Kill the process group."""
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return f"Sent SIGKILL to process group {self.pid}"

            def wait(self, timeout=-1):
                """Wait for process to exit, draining output to prevent deadlock.

                Output is stored in .output property. Prints notice if output was captured.
                Default: uses self.timeout if set, else waits forever.
                """
                if timeout == -1:
                    timeout = self.timeout  # None means forever

                # Loop with internal timeout to handle timeout=None (wait forever)
                start = time.time()
                while True:
                    # Use 60s chunks to drain output while waiting
                    chunk_timeout = 60 if timeout is None else min(60, timeout - (time.time() - start))
                    if chunk_timeout <= 0:
                        break
                    self._output += self.read(timeout=chunk_timeout)  # Buffer for later read()
                    if self.returncode is not None:
                        break
                    if timeout is not None and time.time() - start >= timeout:
                        break

                if self.returncode is None:
                    elapsed = time.time() - start
                    output_info = f", {len(self._output)} bytes captured" if self._output else ""
                    print(f"[wait() timed out after {elapsed:.1f}s{output_info}]")
                return self.returncode is not None

            @property
            def output(self):
                """Output captured during wait(). Call read() for new output."""
                return self._output

            @property
            def returncode(self):
                return self.proc.poll()

            def __repr__(self):
                status = "running" if self.returncode is None else f"exited code={self.returncode}"
                output_info = f" output={len(self._output)}B" if self._output else ""
                return f"[BashProcess pid={self.pid} status={status}{output_info} cmd={self.command!r}]"

        # Ensure bare 'python' works in subprocesses
        from agentlib.tools.subshell import ensure_python_on_path
        ensure_python_on_path()

        # Set up preexec_fn for Linux to kill child when parent dies
        # PR_SET_PDEATHSIG makes kernel send signal to child on parent death
        def _set_pdeathsig():
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                PR_SET_PDEATHSIG = 1
                libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
            except Exception:
                pass  # Non-Linux or ctypes unavailable

        # Start process
        # start_new_session=True creates a new process group, so we can kill the whole tree
        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            start_new_session=True,
            preexec_fn=_set_pdeathsig
        )
        
        bp = BashProcess(proc, command, timeout=timeout)

        # Immediate return if background requested
        if bg:
            return bp

        # Foreground: wait for completion (read uses configured/default timeout)
        output = bp.read()
        
        if bp.returncode is None:
             # Timeout occurred
             return bp
        
        if bp.returncode != 0:
            return f"[Exit code {bp.returncode}]\n{output}"
        
        return output.strip()


def main():
    """CLI entry point for code-agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Code Agent - Python REPL-based coding assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  code-agent                          # Start with default settings
  code-agent --model sonnet           # Use Claude
  code-agent --max-turns 50           # Limit conversation turns
  code-agent --resume                 # Open session picker on startup
  code-agent --resume <session_id>    # Resume specific session directly
"""
    )
    parser.add_argument(
        "--model", "-m",
        default=_get_config_value("code_agent_model", "sonnet"),
        help="LLM model to use (default from config or sonnet)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=_get_config_value("code_agent_max_turns", 100),
        help="Maximum turns per interaction (default from config or 100)"
    )
    parser.add_argument(
        "--sandbox", "-s",
        action="store_true",
        help="Execute agent in a sandboxed filesystem and review changes or save diff"
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Disable sandbox mode (overrides config default)"
    )
    parser.add_argument(
        "--resume", "-r",
        nargs="?",
        const=True,
        default=False,
        metavar="SESSION_ID",
        help="Resume a session. With no argument, opens the session picker."
    )
    args = parser.parse_args()

    try:
        from agentlib.llm_registry import get_model_config
        get_model_config(args.model)
    except ModelNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # Determine sandbox mode: flags override config default
    use_sandbox = _get_config_value("code_agent_sandbox", False)
    if args.sandbox:
        use_sandbox = True
    if args.no_sandbox:
        use_sandbox = False

    if use_sandbox:
        class ConfiguredAgent(SandboxMixin, CodeAgent):
            model = args.model
            max_turns = args.max_turns
    else:
        class ConfiguredAgent(CodeAgent):
            model = args.model
            max_turns = args.max_turns

    try:
        with ConfiguredAgent() as agent:
            agent.cli_run(resume=args.resume)
    except ModelNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
